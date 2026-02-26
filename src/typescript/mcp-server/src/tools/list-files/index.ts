/**
 * List tool implementation — project file and folder structure listing.
 *
 * Reads from the daemon's tracked_files SQLite table to provide
 * tree, summary, and flat views of project structure. Detects submodules
 * from watch_folders and marks them with [submodule: repoName].
 */

import type { SqliteStateManager } from '../../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../utils/project-detector.js';
import type {
  ListOptions,
  ListFormat,
  ListResponse,
  ListStats,
  ComponentSummary,
} from '../list-files-types.js';
import {
  DEFAULT_DEPTH,
  MAX_DEPTH,
  DEFAULT_LIMIT,
  MAX_LIMIT,
} from '../list-files-types.js';
import {
  detectComponents,
  assignComponent,
  componentMatchesFilter,
  type ComponentMap,
} from '../../utils/component-detector.js';
import { buildTree } from './tree-builder.js';
import { renderTree, renderSummary, renderFlat } from './renderers.js';
import { filterByGlob, countFolders } from './filters.js';

// Re-export types for consumers
export type { ListOptions, ListResponse } from '../list-files-types.js';

// Re-export utilities used by tests
export { buildTree } from './tree-builder.js';
export { renderTree, renderSummary, renderFlat } from './renderers.js';
export { globToRegex } from './filters.js';

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
