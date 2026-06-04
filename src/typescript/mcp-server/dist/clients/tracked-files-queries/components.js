/**
 * Query operations for project components from the project_components table.
 *
 * Returns detected workspace components (Cargo, npm, directory fallback)
 * that the daemon persists during file processing.
 */
import { handleTableNotFound } from './helpers.js';
// ── Queries ──────────────────────────────────────────────────────────────
/**
 * List project components from the daemon's project_components table.
 *
 * Returns detected workspace components (Cargo, npm, directory fallback)
 * that the daemon persists during file processing.
 */
export function listProjectComponents(db, watchFolderId) {
    if (!db) {
        return {
            data: [],
            status: 'degraded',
            reason: 'database_not_found',
            message: 'Database not initialized',
        };
    }
    try {
        const rows = db
            .prepare(`SELECT component_name, base_path, source
         FROM project_components
         WHERE watch_folder_id = ?
         ORDER BY component_name ASC`)
            .all(watchFolderId);
        return {
            data: rows.map(row => ({
                componentName: row.component_name,
                basePath: row.base_path,
                source: row.source,
            })),
            status: 'ok',
        };
    }
    catch (error) {
        return handleTableNotFound(error, [], 'project_components');
    }
}
//# sourceMappingURL=components.js.map