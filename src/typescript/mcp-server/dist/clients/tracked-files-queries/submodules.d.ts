/**
 * Query operations for submodule data from watch_folders.
 *
 * Queries watch_folders where parent_watch_id matches the project's
 * watch_id to enumerate Git submodules.
 */
import type { Database as DatabaseType } from 'better-sqlite3';
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
export interface SubmoduleEntry {
    submodulePath: string;
    repoName: string;
}
/**
 * List submodules for a project.
 *
 * Queries watch_folders where parent_watch_id matches the project's watch_id.
 * Extracts repo name from git_remote_url (last path segment minus .git suffix).
 */
export declare function listSubmodules(db: DatabaseType | null, watchFolderId: string): DegradedQueryResult<SubmoduleEntry[]>;
/**
 * Extract repository name from a git remote URL.
 *
 * Examples:
 *   https://github.com/user/repo.git  → repo
 *   git@github.com:user/repo.git      → repo
 *   https://github.com/user/repo      → repo
 *
 * Falls back to the last segment of submodulePath if URL is null or unparseable.
 */
export declare function extractRepoName(gitRemoteUrl: string | null, submodulePath: string): string;
//# sourceMappingURL=submodules.d.ts.map