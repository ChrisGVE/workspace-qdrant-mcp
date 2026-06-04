/**
 * Instance-aware search queries for SqliteStateManager.
 *
 * Provides base_point filtering for multi-clone project scenarios (Task 15).
 */
/**
 * Get the watch_id for a project by its tenant_id.
 * Returns null if not found or database unavailable.
 */
export function getWatchFolderIdByTenantId(db, tenantId) {
    if (!db)
        return null;
    try {
        const row = db.prepare(`SELECT watch_id FROM watch_folders
       WHERE tenant_id = ? AND collection = 'projects' AND parent_watch_id IS NULL
       LIMIT 1`).get(tenantId);
        return row?.watch_id ?? null;
    }
    catch {
        return null;
    }
}
/**
 * Get all distinct base_point values for files tracked under a watch folder
 * (and optionally its submodules via junction table).
 *
 * Used to filter Qdrant search results to the correct instance in
 * multi-clone scenarios.
 */
export function getActiveBasePoints(db, watchFolderId, includeSubmodules = false) {
    if (!db)
        return [];
    try {
        let sql;
        if (includeSubmodules) {
            sql = `SELECT DISTINCT base_point FROM tracked_files
             WHERE base_point IS NOT NULL AND (
                 watch_folder_id = ?
                 OR watch_folder_id IN (
                     SELECT child_watch_id FROM watch_folder_submodules
                     WHERE parent_watch_id = ?
                 )
             )`;
            return db.prepare(sql).all(watchFolderId, watchFolderId)
                .map(r => r.base_point);
        }
        else {
            sql = `SELECT DISTINCT base_point FROM tracked_files
             WHERE base_point IS NOT NULL AND watch_folder_id = ?`;
            return db.prepare(sql).all(watchFolderId)
                .map(r => r.base_point);
        }
    }
    catch {
        return [];
    }
}
//# sourceMappingURL=instance-queries.js.map