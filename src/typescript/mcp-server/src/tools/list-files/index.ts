/**
 * List tool implementation — project file and folder structure listing.
 *
 * Reads from the daemon's tracked_files SQLite table to provide
 * tree, summary, and flat views of project structure. Detects submodules
 * from watch_folders and marks them with [submodule: repoName].
 */

import type { SqliteStateManager } from '../../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../utils/project-detector.js';
import type { TrackedFileEntry } from '../../clients/tracked-files-queries/index.js';
import type {
  ListOptions,
  ListFormat,
  ListResponse,
  ListStats,
  ComponentSummary,
} from '../list-files-types.js';
import { DEFAULT_DEPTH, MAX_DEPTH, DEFAULT_LIMIT, MAX_LIMIT } from '../list-files-types.js';
import {
  detectComponents,
  assignComponent,
  componentMatchesFilter,
  type ComponentMap,
} from '../../utils/component-detector/index.js';
import { buildTree } from './tree-builder.js';
import type { SubmoduleEntry } from '../../clients/tracked-files-queries/index.js';
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

  private buildListStats(
    filteredFiles: TrackedFileEntry[],
    submodules: SubmoduleEntry[],
    basePath: string,
    truncated: boolean,
    totalMatching: number,
    componentSummaries: ComponentSummary[] | undefined
  ): ListStats {
    const languageSet = new Set<string>();
    for (const f of filteredFiles) {
      if (f.language) languageSet.add(f.language);
    }
    const stats: ListStats = {
      files: filteredFiles.length,
      folders: countFolders(buildTree(filteredFiles, submodules, basePath)),
      languages: Array.from(languageSet).sort(),
      truncated,
      totalMatching,
    };
    if (componentSummaries) stats.components = componentSummaries;
    return stats;
  }

  async list(options: ListOptions): Promise<ListResponse> {
    const format: ListFormat = options.format ?? 'tree';
    const depth = Math.min(Math.max(options.depth ?? DEFAULT_DEPTH, 1), MAX_DEPTH);
    const limit = Math.min(Math.max(options.limit ?? DEFAULT_LIMIT, 1), MAX_LIMIT);
    const basePath = options.path ?? '';

    const projectId = options.projectId ?? (await this.resolveProjectId());
    if (!projectId) {
      return errorResponse('Could not detect project. Use projectId parameter.', basePath, format);
    }

    const watchFolderId = this.stateManager.getWatchFolderIdByTenantId(projectId);
    if (!watchFolderId) {
      return errorResponse(
        'Project not found in database. Has the daemon indexed it?',
        basePath,
        format
      );
    }

    const projectResult = this.stateManager.getProjectById(projectId);
    const projectPath = projectResult.data?.project_path ?? null;

    const { files, totalMatching } = this.queryFiles(options, watchFolderId, basePath);
    if (!files) return errorResponse('Database unavailable', basePath, format);

    const { components, componentSummaries } = this.resolveComponents(watchFolderId, projectPath);
    const filteredFiles = filterFilesByComponent(files, options.component, components);
    const submodules = this.stateManager.listSubmodules(watchFolderId).data;

    const { listing, renderedCount } = renderFiles(
      filteredFiles,
      submodules,
      basePath,
      format,
      depth,
      limit
    );
    const truncated = renderedCount < filteredFiles.length;
    const finalListing = truncated
      ? `${listing}\n... (truncated, ${totalMatching} total files match)`
      : listing;

    return {
      success: true,
      projectPath,
      basePath: basePath || '.',
      format,
      listing: finalListing,
      stats: this.buildListStats(
        filteredFiles,
        submodules,
        basePath,
        truncated,
        totalMatching,
        componentSummaries
      ),
    };
  }

  private queryFiles(
    options: ListOptions,
    watchFolderId: string,
    basePath: string
  ): { files: TrackedFileEntry[] | null; totalMatching: number } {
    const queryOpts: Parameters<typeof this.stateManager.listTrackedFiles>[0] = { watchFolderId };
    if (basePath) queryOpts.path = basePath;
    if (options.fileType) queryOpts.fileType = options.fileType;
    if (options.language) queryOpts.language = options.language;
    if (options.extension) queryOpts.extension = options.extension;
    if (options.includeTests !== undefined) queryOpts.includeTests = options.includeTests;

    const filesResult = this.stateManager.listTrackedFiles({ ...queryOpts, limit: MAX_LIMIT });
    if (filesResult.status === 'degraded') {
      return { files: null, totalMatching: 0 };
    }

    const totalMatching = this.stateManager.countTrackedFiles(queryOpts);
    let files = filesResult.data;
    if (options.pattern) {
      files = filterByGlob(files, options.pattern);
    }
    return { files, totalMatching };
  }

  private resolveComponents(
    watchFolderId: string,
    projectPath: string | null
  ): { components: ComponentMap | undefined; componentSummaries: ComponentSummary[] | undefined } {
    const dbComponents = this.stateManager.listProjectComponents(watchFolderId);
    if (dbComponents.status === 'ok' && dbComponents.data.length > 0) {
      const components: ComponentMap = new Map();
      for (const entry of dbComponents.data) {
        components.set(entry.componentName, {
          id: entry.componentName,
          basePath: entry.basePath,
          patterns: [`${entry.basePath}/**`],
          source: entry.source as 'cargo' | 'npm' | 'directory',
        });
      }
      const componentSummaries = dbComponents.data.map((c) => ({
        id: c.componentName,
        basePath: c.basePath,
        source: c.source as ComponentSummary['source'],
      }));
      return { components, componentSummaries };
    }
    if (projectPath) {
      const components = detectComponents(projectPath);
      if (components.size > 0) {
        const componentSummaries = Array.from(components.values()).map((c) => ({
          id: c.id,
          basePath: c.basePath,
          source: c.source,
        }));
        return { components, componentSummaries };
      }
    }
    return { components: undefined, componentSummaries: undefined };
  }

  private async resolveProjectId(): Promise<string | undefined> {
    const cwd = process.cwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
    return projectInfo?.projectId;
  }
}

function errorResponse(message: string, basePath: string, format: ListFormat): ListResponse {
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

function filterFilesByComponent(
  files: TrackedFileEntry[],
  component: string | undefined,
  components: ComponentMap | undefined
): TrackedFileEntry[] {
  if (!component || !components || components.size === 0) return files;
  return files.filter((f) => {
    const assigned = assignComponent(f.relativePath, components);
    if (!assigned) return false;
    return componentMatchesFilter(assigned.id, component);
  });
}

function renderFiles(
  files: TrackedFileEntry[],
  submodules: SubmoduleEntry[],
  basePath: string,
  format: ListFormat,
  depth: number,
  limit: number
): { listing: string; renderedCount: number } {
  const root = buildTree(files, submodules, basePath);
  switch (format) {
    case 'summary': {
      const { text, count } = renderSummary(root, depth, limit);
      return { listing: text, renderedCount: count };
    }
    case 'flat': {
      const { text, count } = renderFlat(files, limit);
      return { listing: text, renderedCount: count };
    }
    default: {
      const { text, count } = renderTree(root, depth, limit);
      return { listing: text, renderedCount: count };
    }
  }
}
