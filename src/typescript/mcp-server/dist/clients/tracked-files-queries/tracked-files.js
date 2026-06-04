/**
 * Query operations for the tracked_files table.
 *
 * Reads from the daemon-owned tracked_files table to provide
 * file listing data for the list MCP tool.
 */
import { handleTableNotFound } from './helpers.js';
/** Build WHERE conditions and params from filter options. */
function buildFilterClause(options) {
    const conditions = ['watch_folder_id = ?'];
    const params = [options.watchFolderId];
    const { path, fileType, language, extension, branch, glob, componentBasePaths, afterPath } = options;
    const includeTests = options.includeTests ?? true;
    if (path) {
        conditions.push('relative_path LIKE ?');
        params.push(`${path}/%`);
    }
    if (fileType) {
        conditions.push('file_type = ?');
        params.push(fileType);
    }
    if (language) {
        conditions.push('language = ?');
        params.push(language);
    }
    if (extension) {
        conditions.push('extension = ?');
        params.push(extension);
    }
    if (!includeTests) {
        conditions.push('is_test = 0');
    }
    if (branch) {
        conditions.push('EXISTS (SELECT 1 FROM json_each(branches) WHERE json_each.value = ?)');
        params.push(branch);
    }
    if (glob) {
        // SQLite GLOB uses * for multi-char and ? for single-char, same as shell globs.
        // The caller passes a pattern like "*.rs" or "src/**/*.ts"; translate ** → * for SQLite.
        const sqliteGlob = glob.replace(/\*\*/g, '*');
        conditions.push('relative_path GLOB ?');
        params.push(sqliteGlob);
    }
    if (componentBasePaths && componentBasePaths.length > 0) {
        // Build OR clause: each base path matches exact or prefix (with /)
        const clauses = componentBasePaths.map(() => '(relative_path = ? OR relative_path LIKE ?)');
        conditions.push(`(${clauses.join(' OR ')})`);
        for (const bp of componentBasePaths) {
            params.push(bp, `${bp}/%`);
        }
    }
    if (afterPath) {
        conditions.push('relative_path > ?');
        params.push(afterPath);
    }
    return { conditions, params };
}
// ── Queries ──────────────────────────────────────────────────────────────
/**
 * List tracked files for a project, with optional filtering.
 *
 * Returns minimal fields needed for tree construction.
 */
export function listTrackedFiles(db, options) {
    if (!db) {
        return {
            data: [],
            status: 'degraded',
            reason: 'database_not_found',
            message: 'Database not initialized',
        };
    }
    try {
        const { conditions, params } = buildFilterClause(options);
        const limit = options.limit ?? 500;
        params.push(limit);
        const sql = `
      SELECT relative_path, file_type, language, extension, is_test
      FROM tracked_files
      WHERE ${conditions.join(' AND ')}
      ORDER BY relative_path ASC
      LIMIT ?
    `;
        const rows = db.prepare(sql).all(...params);
        return { data: rows.map(mapTrackedFileRow), status: 'ok' };
    }
    catch (error) {
        return handleTableNotFound(error, [], 'tracked_files');
    }
}
/**
 * Count total tracked files matching the same filters (ignoring limit).
 *
 * Used to report accurate totals when results are truncated.
 */
export function countTrackedFiles(db, options) {
    if (!db)
        return 0;
    try {
        const { conditions, params } = buildFilterClause(options);
        const sql = `
      SELECT COUNT(*) as cnt
      FROM tracked_files
      WHERE ${conditions.join(' AND ')}
    `;
        const row = db.prepare(sql).get(...params);
        return row.cnt;
    }
    catch {
        return 0;
    }
}
// ── Helpers ──────────────────────────────────────────────────────────────
function mapTrackedFileRow(row) {
    return {
        relativePath: row.relative_path,
        fileType: row.file_type,
        language: row.language,
        extension: row.extension,
        isTest: row.is_test === 1,
    };
}
//# sourceMappingURL=tracked-files.js.map