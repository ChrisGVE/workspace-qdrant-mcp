/**
 * Instance-aware search queries for SqliteStateManager.
 *
 * Provides base_point filtering for multi-clone project scenarios (Task 15).
 */

import type { Database as DatabaseType } from 'better-sqlite3';

/**
 * Get the watch_id for a project by its tenant_id.
 * Returns null if not found or database unavailable.
 */
export function getWatchFolderIdByTenantId(
  db: DatabaseType | null,
  tenantId: string,
): string | null {
  if (!db) return null;

  try {
    const row = db.prepare(
      `SELECT watch_id FROM watch_folders
       WHERE tenant_id = ? AND collection = 'projects' AND parent_watch_id IS NULL
       LIMIT 1`
    ).get(tenantId) as { watch_id: string } | undefined;

    return row?.watch_id ?? null;
  } catch {
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
export function getActiveBasePoints(
  db: DatabaseType | null,
  watchFolderId: string,
  includeSubmodules = false,
): string[] {
  if (!db) return [];

  try {
    let sql: string;
    if (includeSubmodules) {
      sql = `SELECT DISTINCT base_point FROM tracked_files
             WHERE base_point IS NOT NULL AND (
                 watch_folder_id = ?
                 OR watch_folder_id IN (
                     SELECT child_watch_id FROM watch_folder_submodules
                     WHERE parent_watch_id = ?
                 )
             )`;
      return (db.prepare(sql).all(watchFolderId, watchFolderId) as Array<{ base_point: string }>)
        .map(r => r.base_point);
    } else {
      sql = `SELECT DISTINCT base_point FROM tracked_files
             WHERE base_point IS NOT NULL AND watch_folder_id = ?`;
      return (db.prepare(sql).all(watchFolderId) as Array<{ base_point: string }>)
        .map(r => r.base_point);
    }
  } catch {
    return [];
  }
}
