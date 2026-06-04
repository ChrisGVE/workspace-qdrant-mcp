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
export declare function getWatchFolderIdByTenantId(db: DatabaseType | null, tenantId: string): string | null;
/**
 * Get all distinct base_point values for files tracked under a watch folder
 * (and optionally its submodules via junction table).
 *
 * Used to filter Qdrant search results to the correct instance in
 * multi-clone scenarios.
 */
export declare function getActiveBasePoints(db: DatabaseType | null, watchFolderId: string, includeSubmodules?: boolean): string[];
//# sourceMappingURL=instance-queries.d.ts.map