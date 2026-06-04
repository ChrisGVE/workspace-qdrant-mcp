/**
 * Query operations for the tracked_files table.
 *
 * Reads from the daemon-owned tracked_files table to provide
 * file listing data for the list MCP tool.
 */
import type { Database as DatabaseType } from 'better-sqlite3';
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
export interface TrackedFileEntry {
    relativePath: string;
    fileType: string | null;
    language: string | null;
    extension: string | null;
    isTest: boolean;
}
export interface ListTrackedFilesOptions {
    watchFolderId: string;
    path?: string;
    fileType?: string;
    language?: string;
    extension?: string;
    includeTests?: boolean;
    branch?: string;
    limit?: number;
    /** Glob pattern (e.g. "*.rs") — translated to SQLite GLOB */
    glob?: string;
    /** Component base-path prefixes (OR logic) — each entry is a basePath like "src/rust/daemon" */
    componentBasePaths?: string[];
    /** Keyset pagination cursor: return rows with relative_path > cursor */
    afterPath?: string;
}
/**
 * List tracked files for a project, with optional filtering.
 *
 * Returns minimal fields needed for tree construction.
 */
export declare function listTrackedFiles(db: DatabaseType | null, options: ListTrackedFilesOptions): DegradedQueryResult<TrackedFileEntry[]>;
/**
 * Count total tracked files matching the same filters (ignoring limit).
 *
 * Used to report accurate totals when results are truncated.
 */
export declare function countTrackedFiles(db: DatabaseType | null, options: Omit<ListTrackedFilesOptions, 'limit'>): number;
//# sourceMappingURL=tracked-files.d.ts.map