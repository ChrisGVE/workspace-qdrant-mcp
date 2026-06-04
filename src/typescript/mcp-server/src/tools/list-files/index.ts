/**
 * List tool implementation — project file and folder structure listing.
 *
 * Reads from the daemon's tracked_files SQLite table to provide
 * tree, summary, and flat views of project structure. Detects submodules
 * from watch_folders and marks them with [submodule: repoName].
 */

import { randomUUID } from 'node:crypto';
import type { SqliteStateManager } from '../../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../utils/project-detector.js';
import { getEffectiveCwd } from '../../utils/request-context.js';
import type { TrackedFileEntry } from '../../clients/tracked-files-queries/index.js';
import type {
  ListOptions,
  ListFormat,
  ListResponse,
  ListStats,
  ComponentSummary,
} from '../list-files-types.js';
import { DEFAULT_DEPTH, MAX_DEPTH, DEFAULT_LIMIT, MAX_LIMIT } from '../list-files-types.js';
import { SERVER_VERSION as MCP_SERVER_VERSION } from '../../server-types.js';

/**
 * Approximate per-file byte cost the agent would pay if they ran
 * `ls -R` (or equivalent) over the project tree without the list tool.
 *
 * Per `docs/specs/20-token-economy-instrumentation.md` §3.4 this is the
 * loosest baseline among the four tools — file paths are short and the
 * estimate carries genuine uncertainty. 96 B/path is a generous middle
 * ground (typical relative paths fall in the 40-120 B range with the
 * trailing newline / indent overhead).
 */
export const LIST_BYTES_IN_PER_FILE_PROXY = 96;

/**
 * Pure helper: compute `bytes_out` / `bytes_in` for a list response.
 * `bytes_out` is the rendered listing (the actual payload shipped);
 * `bytes_in` approximates the cost of an unshaped `ls -R` over the
 * matching file set.
 */
export function computeListEconomy(
  listing: string,
  totalMatching: number
): { bytesOut: number; bytesIn: number } {
  const bytesOut = listing.length;
  const bytesIn = Math.max(bytesOut, totalMatching * LIST_BYTES_IN_PER_FILE_PROXY);
  return { bytesOut, bytesIn };
}
import {
  detectComponents,
  componentMatchesFilter,
  type ComponentMap,
} from '../../utils/component-detector/index.js';
import { buildTree } from './tree-builder.js';
import type { SubmoduleEntry } from '../../clients/tracked-files-queries/index.js';
import { renderTree, renderSummary, renderFlat } from './renderers.js';
import { countFolders } from './filters.js';

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
    pageFiles: TrackedFileEntry[],
    submodules: SubmoduleEntry[],
    basePath: string,
    truncated: boolean,
    totalMatching: number,
    componentSummaries: ComponentSummary[] | undefined
  ): ListStats {
    const languageSet = new Set<string>();
    for (const f of pageFiles) {
      if (f.language) languageSet.add(f.language);
    }
    const stats: ListStats = {
      files: pageFiles.length,
      folders: countFolders(buildTree(pageFiles, submodules, basePath)),
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

    const startTime = Date.now();
    const eventId = randomUUID();
    // Log start. queryText captures the most distinctive parameters
    // (base path + format) so list events are self-describing under
    // `wqm admin token-savings`.
    this.stateManager.logSearchEvent({
      id: eventId,
      actor: 'claude',
      tool: 'mcp_qdrant',
      op: 'list',
      queryText: basePath || '.',
      topK: limit,
      projectId: options.projectId,
      filters: format !== 'tree' ? JSON.stringify({ format }) : undefined,
    });

    const projectId = options.projectId ?? (await this.resolveProjectId());
    if (!projectId) {
      const response = errorResponse(
        'Could not detect the project. Pass `cwd` (to auto-detect the project) or the `projectId` parameter.',
        basePath,
        format
      );
      this.finishList(eventId, response, startTime, 'unresolved_project');
      return response;
    }

    const watchFolderId = this.stateManager.getWatchFolderIdByTenantId(projectId);
    if (!watchFolderId) {
      const response = errorResponse(
        'Project not found in database. Has the daemon indexed it?',
        basePath,
        format
      );
      this.finishList(eventId, response, startTime, 'unknown_project');
      return response;
    }

    const response = this.buildListResult(
      options,
      projectId,
      watchFolderId,
      basePath,
      format,
      depth,
      limit
    );
    this.finishList(eventId, response, startTime);
    return response;
  }

  /** Record post-execution metrics for a list call. */
  private finishList(
    eventId: string,
    response: ListResponse,
    startTime: number,
    outcome?: string
  ): void {
    const totalMatching = response.stats?.totalMatching ?? 0;
    const economy = computeListEconomy(response.listing, totalMatching);
    const latencyMs = Date.now() - startTime;
    this.stateManager.updateSearchEvent(eventId, {
      resultCount: response.stats?.files ?? 0,
      latencyMs,
      ...(outcome !== undefined ? { outcome } : {}),
    });
    this.stateManager.updateSearchEventEconomy(eventId, {
      bytesIn: economy.bytesIn,
      bytesOut: economy.bytesOut,
      hitsTruncated: 0,
      shapeMode: 'none',
      toolVersion: MCP_SERVER_VERSION,
    });
  }

  private buildListResult(
    options: ListOptions,
    projectId: string,
    watchFolderId: string,
    basePath: string,
    format: ListFormat,
    depth: number,
    limit: number
  ): ListResponse {
    const projectPath = this.stateManager.getProjectById(projectId).data?.project_path ?? null;

    const { components, componentSummaries } = this.resolveComponents(watchFolderId, projectPath);

    // Resolve component filter to SQL-level base paths before querying
    const componentBasePaths = resolveComponentBasePaths(options.component, components);

    const { files, totalMatching } = this.queryFiles(
      options,
      watchFolderId,
      basePath,
      componentBasePaths
    );
    if (!files) return errorResponse('Database unavailable', basePath, format);

    const submodules = this.stateManager.listSubmodules(watchFolderId).data;

    return this.assembleResponse(
      files,
      submodules,
      basePath,
      format,
      depth,
      limit,
      totalMatching,
      projectPath,
      componentSummaries,
      options
    );
  }

  private assembleResponse(
    pageFiles: TrackedFileEntry[],
    submodules: SubmoduleEntry[],
    basePath: string,
    format: ListFormat,
    depth: number,
    limit: number,
    totalMatching: number,
    projectPath: string | null,
    componentSummaries: ComponentSummary[] | undefined,
    options: ListOptions
  ): ListResponse {
    const { listing, renderedCount } = renderFiles(
      pageFiles,
      submodules,
      basePath,
      format,
      depth,
      limit
    );
    // truncated: rendered fewer than the page (render limit hit within page)
    const truncated = renderedCount < pageFiles.length;
    // next_token: present when there are more pages beyond the current fetch window
    const pageSize = options.pageSize ?? limit;
    const hasNextPage = pageFiles.length >= pageSize;
    const lastFile = pageFiles.at(-1);
    const nextToken =
      hasNextPage && lastFile !== undefined
        ? Buffer.from(lastFile.relativePath).toString('base64')
        : undefined;
    const finalListing =
      truncated || nextToken
        ? `${listing}\n... (truncated, ${totalMatching} total files match)`
        : listing;

    const response: ListResponse = {
      success: true,
      projectPath,
      basePath: basePath || '.',
      format,
      listing: finalListing,
      stats: this.buildListStats(
        pageFiles,
        submodules,
        basePath,
        truncated || !!nextToken,
        totalMatching,
        componentSummaries
      ),
    };
    if (nextToken) response.next_token = nextToken;
    return response;
  }

  private queryFiles(
    options: ListOptions,
    watchFolderId: string,
    basePath: string,
    componentBasePaths: string[] | undefined
  ): { files: TrackedFileEntry[] | null; totalMatching: number } {
    // Decode cursor: it's a base64-encoded relativePath from a prior response.
    const afterPath = options.cursor
      ? Buffer.from(options.cursor, 'base64').toString('utf8')
      : undefined;

    const pageSize = Math.min(
      Math.max(options.pageSize ?? options.limit ?? DEFAULT_LIMIT, 1),
      MAX_LIMIT
    );

    const baseOpts: Parameters<typeof this.stateManager.listTrackedFiles>[0] = { watchFolderId };
    if (basePath) baseOpts.path = basePath;
    if (options.fileType) baseOpts.fileType = options.fileType;
    if (options.language) baseOpts.language = options.language;
    if (options.extension) baseOpts.extension = options.extension;
    if (options.includeTests !== undefined) baseOpts.includeTests = options.includeTests;
    if (options.pattern) baseOpts.glob = options.pattern;
    if (componentBasePaths && componentBasePaths.length > 0)
      baseOpts.componentBasePaths = componentBasePaths;

    // Accurate total: COUNT(*) with all filters except the cursor
    const totalMatching = this.stateManager.countTrackedFiles(baseOpts);

    // Paginated fetch: add cursor and page-size limit
    const pageOpts = { ...baseOpts, limit: pageSize };
    if (afterPath) pageOpts.afterPath = afterPath;

    const filesResult = this.stateManager.listTrackedFiles(pageOpts);
    if (filesResult.status === 'degraded') {
      return { files: null, totalMatching: 0 };
    }

    return { files: filesResult.data, totalMatching };
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
    const cwd = getEffectiveCwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false, {
      fallbackToSoleProject: true,
    });
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

/**
 * Resolve component filter to a list of base paths for SQL pushdown.
 *
 * Returns `undefined` when no component filter applies (all files pass).
 * Returns an empty array when the component name is provided but no
 * matching component is found (zero files should match).
 */
function resolveComponentBasePaths(
  component: string | undefined,
  components: ComponentMap | undefined
): string[] | undefined {
  if (!component) return undefined;
  if (!components || components.size === 0) return [];
  const basePaths: string[] = [];
  for (const info of components.values()) {
    if (componentMatchesFilter(info.id, component)) {
      basePaths.push(info.basePath);
    }
  }
  return basePaths;
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
