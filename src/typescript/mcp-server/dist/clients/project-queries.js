/**
 * Project query operations for SqliteStateManager.
 *
 * All functions accept the db handle as first parameter for delegation.
 */
import { COLLECTION_PROJECTS } from '../common/native-bridge.js';
// ── Shared SQL and types ──────────────────────────────────────────────────
const PROJECT_SELECT_FIELDS = `
  SELECT tenant_id, path, git_remote_url, remote_hash,
         disambiguation_path, is_active,
         created_at, updated_at, last_activity_at
  FROM watch_folders
`;
function mapProjectRow(row) {
    const containerFolder = row.path.split('/').filter(Boolean).at(-1) ?? row.path;
    return {
        project_id: row.tenant_id,
        project_path: row.path,
        git_remote_url: row.git_remote_url ?? undefined,
        remote_hash: row.remote_hash ?? undefined,
        disambiguation_path: row.disambiguation_path ?? undefined,
        container_folder: containerFolder,
        is_active: row.is_active === 1,
        created_at: row.created_at,
        last_seen_at: row.updated_at ?? undefined,
        last_activity_at: row.last_activity_at ?? undefined,
    };
}
function handleTableNotFound(error, fallbackData, tableName) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    if (errorMessage.includes('no such table')) {
        return {
            data: fallbackData,
            status: 'degraded',
            reason: 'table_not_found',
            message: `Table ${tableName} not found. Daemon has not initialized database.`,
        };
    }
    throw error;
}
function noDatabaseResult(data) {
    return {
        data,
        status: 'degraded',
        reason: 'database_not_found',
        message: 'Database not initialized',
    };
}
// ── Query helpers ─────────────────────────────────────────────────────────
function queryProjectByPath(db, projectPath) {
    return db
        .prepare(`${PROJECT_SELECT_FIELDS}
      WHERE collection = ? AND (? = path OR ? LIKE path || '/' || '%')
      ORDER BY length(path) DESC
      LIMIT 1`)
        .get(COLLECTION_PROJECTS, projectPath, projectPath);
}
function queryProjectById(db, projectId) {
    return db
        .prepare(`${PROJECT_SELECT_FIELDS}
      WHERE tenant_id = ? AND collection = ?`)
        .get(projectId, COLLECTION_PROJECTS);
}
// ── Public API ────────────────────────────────────────────────────────────
/**
 * Get project by path from watch_folders table.
 *
 * Uses longest-prefix matching to find the closest enclosing project.
 */
export function getProjectByPath(db, projectPath) {
    if (!db)
        return noDatabaseResult(null);
    try {
        const row = queryProjectByPath(db, projectPath);
        return { data: row ? mapProjectRow(row) : null, status: 'ok' };
    }
    catch (error) {
        return handleTableNotFound(error, null, 'watch_folders');
    }
}
/**
 * Get project by tenant_id from watch_folders table.
 */
export function getProjectById(db, projectId) {
    if (!db)
        return noDatabaseResult(null);
    try {
        const row = queryProjectById(db, projectId);
        return { data: row ? mapProjectRow(row) : null, status: 'ok' };
    }
    catch (error) {
        return handleTableNotFound(error, null, 'watch_folders');
    }
}
/**
 * List all active projects from watch_folders table.
 */
export function listActiveProjects(db) {
    if (!db)
        return noDatabaseResult([]);
    try {
        const projects = db
            .prepare(`${PROJECT_SELECT_FIELDS}
        WHERE is_active = 1 AND collection = ?
        ORDER BY last_activity_at DESC`)
            .all(COLLECTION_PROJECTS);
        return { data: projects.map(mapProjectRow), status: 'ok' };
    }
    catch (error) {
        return handleTableNotFound(error, [], 'watch_folders');
    }
}
//# sourceMappingURL=project-queries.js.map