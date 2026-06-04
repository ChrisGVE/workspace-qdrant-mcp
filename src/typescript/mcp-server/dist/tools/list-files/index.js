/**
 * List tool implementation — project file and folder structure listing.
 *
 * Reads from the daemon's tracked_files SQLite table to provide
 * tree, summary, and flat views of project structure. Detects submodules
 * from watch_folders and marks them with [submodule: repoName].
 */
import { DEFAULT_DEPTH, MAX_DEPTH, DEFAULT_LIMIT, MAX_LIMIT } from '../list-files-types.js';
import { detectComponents, componentMatchesFilter, } from '../../utils/component-detector/index.js';
import { buildTree } from './tree-builder.js';
import { renderTree, renderSummary, renderFlat } from './renderers.js';
import { countFolders } from './filters.js';
// Re-export utilities used by tests
export { buildTree } from './tree-builder.js';
export { renderTree, renderSummary, renderFlat } from './renderers.js';
export { globToRegex } from './filters.js';
/**
 * List tool for project file and folder structure
 */
export class ListFilesTool {
    stateManager;
    projectDetector;
    constructor(stateManager, projectDetector) {
        this.stateManager = stateManager;
        this.projectDetector = projectDetector;
    }
    buildListStats(pageFiles, submodules, basePath, truncated, totalMatching, componentSummaries) {
        const languageSet = new Set();
        for (const f of pageFiles) {
            if (f.language)
                languageSet.add(f.language);
        }
        const stats = {
            files: pageFiles.length,
            folders: countFolders(buildTree(pageFiles, submodules, basePath)),
            languages: Array.from(languageSet).sort(),
            truncated,
            totalMatching,
        };
        if (componentSummaries)
            stats.components = componentSummaries;
        return stats;
    }
    async list(options) {
        const format = options.format ?? 'tree';
        const depth = Math.min(Math.max(options.depth ?? DEFAULT_DEPTH, 1), MAX_DEPTH);
        const limit = Math.min(Math.max(options.limit ?? DEFAULT_LIMIT, 1), MAX_LIMIT);
        const basePath = options.path ?? '';
        const projectId = options.projectId ?? (await this.resolveProjectId());
        if (!projectId) {
            return errorResponse('Could not detect project. Use projectId parameter.', basePath, format);
        }
        const watchFolderId = this.stateManager.getWatchFolderIdByTenantId(projectId);
        if (!watchFolderId) {
            return errorResponse('Project not found in database. Has the daemon indexed it?', basePath, format);
        }
        return this.buildListResult(options, projectId, watchFolderId, basePath, format, depth, limit);
    }
    buildListResult(options, projectId, watchFolderId, basePath, format, depth, limit) {
        const projectPath = this.stateManager.getProjectById(projectId).data?.project_path ?? null;
        const { components, componentSummaries } = this.resolveComponents(watchFolderId, projectPath);
        // Resolve component filter to SQL-level base paths before querying
        const componentBasePaths = resolveComponentBasePaths(options.component, components);
        const { files, totalMatching } = this.queryFiles(options, watchFolderId, basePath, componentBasePaths);
        if (!files)
            return errorResponse('Database unavailable', basePath, format);
        const submodules = this.stateManager.listSubmodules(watchFolderId).data;
        return this.assembleResponse(files, submodules, basePath, format, depth, limit, totalMatching, projectPath, componentSummaries, options);
    }
    assembleResponse(pageFiles, submodules, basePath, format, depth, limit, totalMatching, projectPath, componentSummaries, options) {
        const { listing, renderedCount } = renderFiles(pageFiles, submodules, basePath, format, depth, limit);
        // truncated: rendered fewer than the page (render limit hit within page)
        const truncated = renderedCount < pageFiles.length;
        // next_token: present when there are more pages beyond the current fetch window
        const pageSize = options.pageSize ?? limit;
        const hasNextPage = pageFiles.length >= pageSize;
        const lastFile = pageFiles.at(-1);
        const nextToken = hasNextPage && lastFile !== undefined
            ? Buffer.from(lastFile.relativePath).toString('base64')
            : undefined;
        const finalListing = truncated || nextToken
            ? `${listing}\n... (truncated, ${totalMatching} total files match)`
            : listing;
        const response = {
            success: true,
            projectPath,
            basePath: basePath || '.',
            format,
            listing: finalListing,
            stats: this.buildListStats(pageFiles, submodules, basePath, truncated || !!nextToken, totalMatching, componentSummaries),
        };
        if (nextToken)
            response.next_token = nextToken;
        return response;
    }
    queryFiles(options, watchFolderId, basePath, componentBasePaths) {
        // Decode cursor: it's a base64-encoded relativePath from a prior response.
        const afterPath = options.cursor
            ? Buffer.from(options.cursor, 'base64').toString('utf8')
            : undefined;
        const pageSize = Math.min(Math.max(options.pageSize ?? options.limit ?? DEFAULT_LIMIT, 1), MAX_LIMIT);
        const baseOpts = { watchFolderId };
        if (basePath)
            baseOpts.path = basePath;
        if (options.fileType)
            baseOpts.fileType = options.fileType;
        if (options.language)
            baseOpts.language = options.language;
        if (options.extension)
            baseOpts.extension = options.extension;
        if (options.includeTests !== undefined)
            baseOpts.includeTests = options.includeTests;
        if (options.pattern)
            baseOpts.glob = options.pattern;
        if (componentBasePaths && componentBasePaths.length > 0)
            baseOpts.componentBasePaths = componentBasePaths;
        // "*" means cross-branch — omit the filter to return all branches.
        if (options.branch && options.branch !== '*')
            baseOpts.branch = options.branch;
        // Accurate total: COUNT(*) with all filters except the cursor
        const totalMatching = this.stateManager.countTrackedFiles(baseOpts);
        // Paginated fetch: add cursor and page-size limit
        const pageOpts = { ...baseOpts, limit: pageSize };
        if (afterPath)
            pageOpts.afterPath = afterPath;
        const filesResult = this.stateManager.listTrackedFiles(pageOpts);
        if (filesResult.status === 'degraded') {
            return { files: null, totalMatching: 0 };
        }
        return { files: filesResult.data, totalMatching };
    }
    resolveComponents(watchFolderId, projectPath) {
        const dbComponents = this.stateManager.listProjectComponents(watchFolderId);
        if (dbComponents.status === 'ok' && dbComponents.data.length > 0) {
            const components = new Map();
            for (const entry of dbComponents.data) {
                components.set(entry.componentName, {
                    id: entry.componentName,
                    basePath: entry.basePath,
                    patterns: [`${entry.basePath}/**`],
                    source: entry.source,
                });
            }
            const componentSummaries = dbComponents.data.map((c) => ({
                id: c.componentName,
                basePath: c.basePath,
                source: c.source,
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
    async resolveProjectId() {
        const cwd = process.cwd();
        const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
        return projectInfo?.projectId;
    }
}
function errorResponse(message, basePath, format) {
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
function resolveComponentBasePaths(component, components) {
    if (!component)
        return undefined;
    if (!components || components.size === 0)
        return [];
    const basePaths = [];
    for (const info of components.values()) {
        if (componentMatchesFilter(info.id, component)) {
            basePaths.push(info.basePath);
        }
    }
    return basePaths;
}
function renderFiles(files, submodules, basePath, format, depth, limit) {
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
//# sourceMappingURL=index.js.map