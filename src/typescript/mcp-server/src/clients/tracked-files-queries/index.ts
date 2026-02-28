/**
 * Tracked files query operations for SqliteStateManager.
 *
 * Reads from the daemon-owned tracked_files and watch_folders tables
 * to provide file listing data for the list MCP tool.
 *
 * All functions accept the db handle as first parameter for delegation.
 */

export type { TrackedFileEntry, ListTrackedFilesOptions } from './tracked-files.js';
export { listTrackedFiles, countTrackedFiles } from './tracked-files.js';

export type { SubmoduleEntry } from './submodules.js';
export { listSubmodules, extractRepoName } from './submodules.js';

export type { ComponentEntry } from './components.js';
export { listProjectComponents } from './components.js';
