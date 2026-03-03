/**
 * Project query operations for SqliteStateManager.
 *
 * All functions accept the db handle as first parameter for delegation.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import type { RegisteredProject } from '../types/state.js';
import { COLLECTION_PROJECTS } from '../common/native-bridge.js';
import type { DegradedQueryResult } from './sqlite-state-manager.js';

// ── Shared SQL and types ──────────────────────────────────────────────────

const PROJECT_SELECT_FIELDS = `
  SELECT tenant_id, path, git_remote_url, remote_hash,
         disambiguation_path, is_active,
         created_at, updated_at, last_activity_at
  FROM watch_folders
`;

interface WatchFolderRow {
  tenant_id: string;
  path: string;
  git_remote_url: string | null;
  remote_hash: string | null;
  disambiguation_path: string | null;
  is_active: number;
  created_at: string;
  updated_at: string | null;
  last_activity_at: string | null;
}

function mapProjectRow(row: WatchFolderRow): RegisteredProject {
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

function handleTableNotFound<T>(
  error: unknown,
  fallbackData: T,
  tableName: string,
): DegradedQueryResult<T> {
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

function noDatabaseResult<T>(data: T): DegradedQueryResult<T> {
  return {
    data,
    status: 'degraded',
    reason: 'database_not_found',
    message: 'Database not initialized',
  };
}

// ── Query helpers ─────────────────────────────────────────────────────────

function queryProjectByPath(
  db: DatabaseType,
  projectPath: string,
): WatchFolderRow | undefined {
  return db
    .prepare(
      `${PROJECT_SELECT_FIELDS}
      WHERE collection = ? AND (? = path OR ? LIKE path || '/' || '%')
      ORDER BY length(path) DESC
      LIMIT 1`
    )
    .get(COLLECTION_PROJECTS, projectPath, projectPath) as WatchFolderRow | undefined;
}

function queryProjectById(
  db: DatabaseType,
  projectId: string,
): WatchFolderRow | undefined {
  return db
    .prepare(
      `${PROJECT_SELECT_FIELDS}
      WHERE tenant_id = ? AND collection = ?`
    )
    .get(projectId, COLLECTION_PROJECTS) as WatchFolderRow | undefined;
}

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Get project by path from watch_folders table.
 *
 * Uses longest-prefix matching to find the closest enclosing project.
 */
export function getProjectByPath(
  db: DatabaseType | null,
  projectPath: string,
): DegradedQueryResult<RegisteredProject | null> {
  if (!db) return noDatabaseResult(null);

  try {
    const row = queryProjectByPath(db, projectPath);
    return { data: row ? mapProjectRow(row) : null, status: 'ok' };
  } catch (error) {
    return handleTableNotFound<RegisteredProject | null>(error, null, 'watch_folders');
  }
}

/**
 * Get project by tenant_id from watch_folders table.
 */
export function getProjectById(
  db: DatabaseType | null,
  projectId: string,
): DegradedQueryResult<RegisteredProject | null> {
  if (!db) return noDatabaseResult(null);

  try {
    const row = queryProjectById(db, projectId);
    return { data: row ? mapProjectRow(row) : null, status: 'ok' };
  } catch (error) {
    return handleTableNotFound<RegisteredProject | null>(error, null, 'watch_folders');
  }
}

/**
 * List all active projects from watch_folders table.
 */
export function listActiveProjects(
  db: DatabaseType | null,
): DegradedQueryResult<RegisteredProject[]> {
  if (!db) return noDatabaseResult([]);

  try {
    const projects = db
      .prepare(
        `${PROJECT_SELECT_FIELDS}
        WHERE is_active = 1 AND collection = ?
        ORDER BY last_activity_at DESC`
      )
      .all(COLLECTION_PROJECTS) as WatchFolderRow[];

    return { data: projects.map(mapProjectRow), status: 'ok' };
  } catch (error) {
    return handleTableNotFound<RegisteredProject[]>(error, [], 'watch_folders');
  }
}
