/**
 * List tool implementation — project file and folder structure listing.
 *
 * Reads from the daemon's tracked_files SQLite table to provide
 * tree, summary, and flat views of project structure. Detects submodules
 * from watch_folders and marks them with [submodule: repoName].
 */

import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { TrackedFileEntry, SubmoduleEntry } from '../clients/tracked-files-queries.js';
import type {
  ListOptions,
  ListFormat,
  ListResponse,
  ListStats,
  ComponentSummary,
  FolderNode,
} from './list-files-types.js';
import {
  DEFAULT_DEPTH,
  MAX_DEPTH,
  DEFAULT_LIMIT,
  MAX_LIMIT,
} from './list-files-types.js';
import {
  detectComponents,
  assignComponent,
  componentMatchesFilter,
  type ComponentMap,
} from '../utils/component-detector.js';

// Re-export types for consumers
export type { ListOptions, ListResponse } from './list-files-types.js';

/**
 * List tool for project file and folder structure
 */
export class ListFilesTool {
  private readonly stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;

  constructor(stateManager: SqliteStateManager, projectDetector: ProjectDetector) {
    this.stateManager = stateManager;
    this.projectDetector = projectDetector;
  }

  async list(options: ListOptions): Promise<ListResponse> {
    const format: ListFormat = options.format ?? 'tree';
    const depth = Math.min(Math.max(options.depth ?? DEFAULT_DEPTH, 1), MAX_DEPTH);
    const limit = Math.min(Math.max(options.limit ?? DEFAULT_LIMIT, 1), MAX_LIMIT);
    const basePath = options.path ?? '';

    // Resolve project
    const projectId = options.projectId ?? await this.resolveProjectId();
    if (!projectId) {
      return errorResponse('Could not detect project. Use projectId parameter.', basePath, format);
    }

    const watchFolderId = this.stateManager.getWatchFolderIdByTenantId(projectId);
    if (!watchFolderId) {
      return errorResponse('Project not found in database. Has the daemon indexed it?', basePath, format);
    }

    // Resolve project path for the response
    const projectResult = this.stateManager.getProjectById(projectId);
    const projectPath = projectResult.data?.project_path ?? null;

    // Query tracked files — build options conditionally for exactOptionalPropertyTypes
    const queryOpts: Parameters<typeof this.stateManager.listTrackedFiles>[0] = { watchFolderId };
    if (basePath) queryOpts.path = basePath;
    if (options.fileType) queryOpts.fileType = options.fileType;
    if (options.language) queryOpts.language = options.language;
    if (options.extension) queryOpts.extension = options.extension;
    if (options.includeTests !== undefined) queryOpts.includeTests = options.includeTests;

    const filesResult = this.stateManager.listTrackedFiles({ ...queryOpts, limit: MAX_LIMIT });
    if (filesResult.status === 'degraded') {
      return errorResponse(
        filesResult.message ?? 'Database unavailable',
        basePath,
        format,
      );
    }

    const totalMatching = this.stateManager.countTrackedFiles(queryOpts);

    // Apply glob pattern filter if specified
    let files = filesResult.data;
    if (options.pattern) {
      files = filterByGlob(files, options.pattern);
    }

    // Load components from daemon's SQLite table (populated during file processing).
    // Falls back to filesystem detection if the table is empty or unavailable.
    let components: ComponentMap | undefined;
    let componentSummaries: ComponentSummary[] | undefined;
    const dbComponents = this.stateManager.listProjectComponents(watchFolderId);
    if (dbComponents.status === 'ok' && dbComponents.data.length > 0) {
      components = new Map();
      for (const entry of dbComponents.data) {
        components.set(entry.componentName, {
          id: entry.componentName,
          basePath: entry.basePath,
          patterns: [`${entry.basePath}/**`],
          source: entry.source as 'cargo' | 'npm' | 'directory',
        });
      }
      componentSummaries = dbComponents.data.map(c => ({
        id: c.componentName,
        basePath: c.basePath,
        source: c.source as ComponentSummary['source'],
      }));
    } else if (projectPath) {
      // Fallback: detect from filesystem (daemon hasn't indexed yet)
      components = detectComponents(projectPath);
      if (components.size > 0) {
        componentSummaries = Array.from(components.values()).map(c => ({
          id: c.id,
          basePath: c.basePath,
          source: c.source,
        }));
      }
    }

    // Filter by component if requested
    if (options.component && components && components.size > 0) {
      files = files.filter(f => {
        const assigned = assignComponent(f.relativePath, components!);
        if (!assigned) return false;
        return componentMatchesFilter(assigned.id, options.component!);
      });
    }

    // Query submodules
    const submodulesResult = this.stateManager.listSubmodules(watchFolderId);
    const submodules = submodulesResult.data;

    // Collect language stats
    const languageSet = new Set<string>();
    for (const f of files) {
      if (f.language) languageSet.add(f.language);
    }

    // Build tree and render
    const root = buildTree(files, submodules, basePath);
    let listing: string;
    let renderedCount: number;

    switch (format) {
      case 'summary':
        ({ text: listing, count: renderedCount } = renderSummary(root, depth, limit));
        break;
      case 'flat':
        ({ text: listing, count: renderedCount } = renderFlat(files, limit));
        break;
      default:
        ({ text: listing, count: renderedCount } = renderTree(root, depth, limit));
        break;
    }

    const truncated = renderedCount < files.length;
    if (truncated) {
      listing += `\n... (truncated, ${totalMatching} total files match)`;
    }

    const stats: ListStats = {
      files: files.length,
      folders: countFolders(root),
      languages: Array.from(languageSet).sort(),
      truncated,
      totalMatching,
    };
    if (componentSummaries) stats.components = componentSummaries;

    return {
      success: true,
      projectPath,
      basePath: basePath || '.',
      format,
      listing,
      stats,
    };
  }

  private async resolveProjectId(): Promise<string | undefined> {
    const cwd = process.cwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
    return projectInfo?.projectId;
  }
}

// ── Tree construction ────────────────────────────────────────────────────

/**
 * Build a folder tree from flat file paths.
 *
 * If basePath is set, strips it from relativePath before inserting.
 */
export function buildTree(
  files: TrackedFileEntry[],
  submodules: SubmoduleEntry[],
  basePath: string,
): FolderNode {
  const root: FolderNode = {
    name: basePath || '.',
    children: new Map(),
    files: [],
    totalFiles: 0,
  };

  // Build set of submodule paths for fast lookup
  const submoduleMap = new Map<string, SubmoduleEntry>();
  for (const sm of submodules) {
    submoduleMap.set(sm.submodulePath, sm);
  }

  for (const file of files) {
    let relPath = file.relativePath;

    // Strip basePath prefix
    if (basePath && relPath.startsWith(basePath + '/')) {
      relPath = relPath.slice(basePath.length + 1);
    }

    const segments = relPath.split('/');
    const fileName = segments.pop()!;
    let current = root;

    // Walk/create folder path
    let pathSoFar = basePath;
    for (const segment of segments) {
      pathSoFar = pathSoFar ? `${pathSoFar}/${segment}` : segment;

      if (!current.children.has(segment)) {
        const node: FolderNode = {
          name: segment,
          children: new Map(),
          files: [],
          totalFiles: 0,
        };

        // Check if this folder is a submodule root
        const sm = submoduleMap.get(pathSoFar);
        if (sm) {
          node.submodule = { repoName: sm.repoName };
        }

        current.children.set(segment, node);
      }

      current = current.children.get(segment)!;

      // If we hit a submodule, don't go deeper
      if (current.submodule) break;
    }

    // Only add the file if we didn't stop at a submodule
    if (!current.submodule) {
      current.files.push({
        name: fileName,
        extension: file.extension,
        language: file.language,
        isTest: file.isTest,
      });
    }
  }

  // Compute totalFiles bottom-up
  computeTotalFiles(root);

  return root;
}

function computeTotalFiles(node: FolderNode): number {
  let total = node.files.length;
  for (const child of node.children.values()) {
    total += computeTotalFiles(child);
  }
  node.totalFiles = total;
  return total;
}

// ── Rendering: tree mode ─────────────────────────────────────────────────

interface RenderResult {
  text: string;
  count: number;
}

export function renderTree(
  root: FolderNode,
  maxDepth: number,
  limit: number,
): RenderResult {
  const lines: string[] = [];
  let count = 0;

  function walk(node: FolderNode, indent: number, currentDepth: number): boolean {
    const prefix = '  '.repeat(indent);

    // Render child folders first (sorted)
    const sortedChildren = Array.from(node.children.entries()).sort((a, b) =>
      a[0].localeCompare(b[0]),
    );

    for (const [, child] of sortedChildren) {
      if (count >= limit) return false;

      if (child.submodule) {
        lines.push(`${prefix}${child.name}/ [submodule: ${child.submodule.repoName}]`);
        count++;
        continue;
      }

      if (currentDepth >= maxDepth) {
        lines.push(`${prefix}${child.name}/ (${child.totalFiles} files)`);
        count++;
        continue;
      }

      lines.push(`${prefix}${child.name}/`);
      count++;
      if (!walk(child, indent + 1, currentDepth + 1)) return false;
    }

    // Render files (sorted)
    const sortedFiles = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
    for (const file of sortedFiles) {
      if (count >= limit) return false;
      const tag = file.extension ? ` [${file.extension}]` : '';
      lines.push(`${prefix}${file.name}${tag}`);
      count++;
    }

    return true;
  }

  // Start rendering from root's children (skip the root node name itself)
  walk(root, 0, 1);

  return { text: lines.join('\n'), count };
}

// ── Rendering: summary mode ──────────────────────────────────────────────

export function renderSummary(
  root: FolderNode,
  maxDepth: number,
  limit: number,
): RenderResult {
  const lines: string[] = [];
  let count = 0;

  function walk(node: FolderNode, indent: number, currentDepth: number, chainPrefix: string): boolean {
    // Get sorted children
    const sortedChildren = Array.from(node.children.entries()).sort((a, b) =>
      a[0].localeCompare(b[0]),
    );

    for (const [name, child] of sortedChildren) {
      if (count >= limit) return false;

      const childPath = chainPrefix ? `${chainPrefix}${name}` : name;

      if (child.submodule) {
        const prefix = '  '.repeat(indent);
        lines.push(`${prefix}${childPath}/ [submodule: ${child.submodule.repoName}]`);
        count++;
        continue;
      }

      // Single-child chain collapsing: if this folder has exactly one child folder
      // and no files, collapse into the child's display
      if (
        child.children.size === 1 &&
        child.files.length === 0 &&
        currentDepth < maxDepth
      ) {
        // Continue the chain
        if (!walk(child, indent, currentDepth + 1, `${childPath}/`)) return false;
        continue;
      }

      const prefix = '  '.repeat(indent);
      const extCounts = aggregateExtensions(child);
      const summary = formatExtensionSummary(child.totalFiles, extCounts);

      if (currentDepth >= maxDepth) {
        lines.push(`${prefix}${childPath}/ ${summary}`);
        count++;
        continue;
      }

      lines.push(`${prefix}${childPath}/ ${summary}`);
      count++;

      if (!walk(child, indent + 1, currentDepth + 1, '')) return false;
    }

    return true;
  }

  walk(root, 0, 1, '');

  return { text: lines.join('\n'), count };
}

function aggregateExtensions(node: FolderNode): Map<string, number> {
  const counts = new Map<string, number>();

  function collect(n: FolderNode): void {
    for (const file of n.files) {
      const key = file.extension ?? 'other';
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    for (const child of n.children.values()) {
      if (!child.submodule) collect(child);
    }
  }

  collect(node);
  return counts;
}

function formatExtensionSummary(
  totalFiles: number,
  extCounts: Map<string, number>,
): string {
  if (totalFiles === 0) return '(empty)';

  // Sort by count descending, show top 4
  const sorted = Array.from(extCounts.entries()).sort((a, b) => b[1] - a[1]);
  const shown = sorted.slice(0, 4);
  const parts = shown.map(([ext, n]) => `${n} ${ext}`);

  if (sorted.length > 4) {
    const remaining = totalFiles - shown.reduce((sum, [, n]) => sum + n, 0);
    if (remaining > 0) parts.push(`${remaining} other`);
  }

  return `(${totalFiles} files: ${parts.join(', ')})`;
}

// ── Rendering: flat mode ─────────────────────────────────────────────────

export function renderFlat(
  files: TrackedFileEntry[],
  limit: number,
): RenderResult {
  const lines: string[] = [];
  let count = 0;

  for (const file of files) {
    if (count >= limit) break;
    lines.push(file.relativePath);
    count++;
  }

  return { text: lines.join('\n'), count };
}

// ── Helpers ──────────────────────────────────────────────────────────────

function countFolders(node: FolderNode): number {
  let count = 0;
  for (const child of node.children.values()) {
    count += 1 + countFolders(child);
  }
  return count;
}

/**
 * Simple glob filter on relative paths.
 * Supports * (any non-/ chars) and ** (any path segment including /).
 */
function filterByGlob(files: TrackedFileEntry[], pattern: string): TrackedFileEntry[] {
  const regex = globToRegex(pattern);
  return files.filter((f) => regex.test(f.relativePath));
}

export function globToRegex(pattern: string): RegExp {
  let result = '';
  let i = 0;

  while (i < pattern.length) {
    const c = pattern.charAt(i);

    if (c === '*' && i + 1 < pattern.length && pattern.charAt(i + 1) === '*') {
      // ** matches anything including /
      result += '.*';
      i += 2;
      // Skip trailing /
      if (i < pattern.length && pattern.charAt(i) === '/') i++;
    } else if (c === '*') {
      // * matches anything except /
      result += '[^/]*';
      i++;
    } else if (c === '?') {
      result += '[^/]';
      i++;
    } else if ('.+^${}()|[]\\'.includes(c)) {
      result += '\\' + c;
      i++;
    } else {
      result += c;
      i++;
    }
  }

  return new RegExp(`^${result}$`);
}

function errorResponse(
  message: string,
  basePath: string,
  format: ListFormat,
): ListResponse {
  return {
    success: false,
    projectPath: null,
    basePath: basePath || '.',
    format,
    listing: '',
    stats: { files: 0, folders: 0, languages: [], truncated: false, totalMatching: 0 },
    message,
  };
}
