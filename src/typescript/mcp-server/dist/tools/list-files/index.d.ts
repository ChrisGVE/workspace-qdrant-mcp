/**
 * List tool implementation — project file and folder structure listing.
 *
 * Reads from the daemon's tracked_files SQLite table to provide
 * tree, summary, and flat views of project structure. Detects submodules
 * from watch_folders and marks them with [submodule: repoName].
 */
import type { SqliteStateManager } from '../../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../utils/project-detector.js';
import type { ListOptions, ListResponse } from '../list-files-types.js';
export type { ListOptions, ListResponse } from '../list-files-types.js';
export { buildTree } from './tree-builder.js';
export { renderTree, renderSummary, renderFlat } from './renderers.js';
export { globToRegex } from './filters.js';
/**
 * List tool for project file and folder structure
 */
export declare class ListFilesTool {
    private readonly stateManager;
    private readonly projectDetector;
    constructor(stateManager: SqliteStateManager, projectDetector: ProjectDetector);
    private buildListStats;
    list(options: ListOptions): Promise<ListResponse>;
    private buildListResult;
    private assembleResponse;
    private queryFiles;
    private resolveComponents;
    private resolveProjectId;
}
//# sourceMappingURL=index.d.ts.map