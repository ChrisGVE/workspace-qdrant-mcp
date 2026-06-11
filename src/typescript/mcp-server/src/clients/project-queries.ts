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
    // `watch_folders.is_active` is a session counter (incremented by
    // RegisterProject, decremented by DeprioritizeProject), not a boolean.
    // `> 0` matches the semantics callers want from this flag: "the
    // project has at least one live session".
    is_active: row.is_active > 0,
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

/**
 * Canonicalize a host/container path so the same physical location compares
 * equal across path namespaces.
 *
 * The MCP server (often a Docker container) stores project roots in whatever
 * form the daemon observed — e.g. Docker Desktop's `/run/desktop/mnt/host/c/…`
 * — while a client reports its CWD as a Windows path (`C:\…`), a WSL mount
 * (`/mnt/c/…`) or an MSYS/Git-Bash path (`/c/…`). All four denote drive C, so
 * they fold to `/c/…`. A Windows host editing a repo that lives on the WSL ext4
 * filesystem reports a UNC share CWD (`\\wsl.localhost\<distro>\home\…`); the
 * daemon runs inside the distro and stores the native POSIX path (`/home/…`),
 * so the UNC view folds to that same root.
 *
 * Pure prefix normalization: separators to `/`, the drive/mount/UNC prefix
 * unified, duplicate and trailing slashes trimmed. Case is preserved (callers
 * compare case-insensitively where appropriate). Native POSIX paths and
 * multi-letter mount dirs (e.g. `/mnt/data`) are left untouched.
 */
export function canonicalizeHostPath(p: string): string {
  let s = p.replace(/\\/g, '/');
  // WSL UNC share from a Windows host: "\\wsl.localhost\<distro>\…" or the
  // legacy "\\wsl$\<distro>\…" (slash-normalized above to
  // "//wsl.localhost/<distro>/…"). Drop the share + distro segments so the
  // path folds to the native ext4 root the daemon stores ("/home/…").
  s = s.replace(/^\/+wsl(?:\.localhost|\$)\/[^/]+/i, '');
  // Windows drive: "C:/…" or bare "C:" → "/c/…".
  s = s.replace(/^([A-Za-z]):(?=\/|$)/, (_m, drive: string) => `/${drive.toLowerCase()}`);
  // Docker Desktop host mount: "/run/desktop/mnt/host/c/…" → "/c/…".
  s = s.replace(/^\/run\/desktop\/mnt\/host(?=\/[A-Za-z](?:\/|$))/i, '');
  // WSL drive mount: "/mnt/c/…" → "/c/…".
  s = s.replace(/^\/mnt(?=\/[A-Za-z](?:\/|$))/i, '');
  // Collapse duplicate slashes; drop a trailing slash (but keep root "/").
  s = s.replace(/\/{2,}/g, '/');
  if (s.length > 1 && s.endsWith('/')) s = s.slice(0, -1);
  return s;
}

function queryProjectByPath(
  db: DatabaseType,
  projectPath: string,
): WatchFolderRow | undefined {
  const rows = db
    .prepare(`${PROJECT_SELECT_FIELDS} WHERE collection = ?`)
    .all(COLLECTION_PROJECTS) as WatchFolderRow[];
  if (rows.length === 0) return undefined;

  // Longest canonical-prefix match. Canonicalizing both sides bridges the
  // host/container path-namespace gap (see canonicalizeHostPath); the length
  // tiebreak mirrors the previous SQL `ORDER BY length(path) DESC`. Compared
  // case-insensitively to tolerate Windows path-case drift.
  const target = canonicalizeHostPath(projectPath).toLowerCase();
  let best: WatchFolderRow | undefined;
  let bestLen = -1;
  for (const row of rows) {
    const stored = canonicalizeHostPath(row.path).toLowerCase();
    if ((target === stored || target.startsWith(`${stored}/`)) && stored.length > bestLen) {
      best = row;
      bestLen = stored.length;
    }
  }
  return best;
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

/**
 * List every registered project regardless of activity state.
 *
 * Distinct from `listActiveProjects` which filters on `is_active = 1`.
 * `is_active` is actually a session counter (incremented by
 * `RegisterProject`, decremented by `DeprioritizeProject`), so projects
 * with 2+ overlapping sessions or any currently-deactivated state would
 * be hidden by the strict-equality filter. The admin UI needs the full
 * inventory to render status pills correctly.
 */
export function listAllProjects(
  db: DatabaseType | null,
): DegradedQueryResult<RegisteredProject[]> {
  if (!db) return noDatabaseResult([]);

  try {
    const projects = db
      .prepare(
        `${PROJECT_SELECT_FIELDS}
        WHERE collection = ?
        ORDER BY is_active DESC, last_activity_at DESC NULLS LAST, path ASC`
      )
      .all(COLLECTION_PROJECTS) as WatchFolderRow[];

    return { data: projects.map(mapProjectRow), status: 'ok' };
  } catch (error) {
    return handleTableNotFound<RegisteredProject[]>(error, [], 'watch_folders');
  }
}
