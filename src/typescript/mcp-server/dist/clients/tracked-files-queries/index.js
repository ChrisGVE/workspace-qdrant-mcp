/**
 * Tracked files query operations for SqliteStateManager.
 *
 * Reads from the daemon-owned tracked_files and watch_folders tables
 * to provide file listing data for the list MCP tool.
 *
 * All functions accept the db handle as first parameter for delegation.
 */
export { listTrackedFiles, countTrackedFiles } from './tracked-files.js';
export { listSubmodules, extractRepoName } from './submodules.js';
export { listProjectComponents } from './components.js';
//# sourceMappingURL=index.js.map