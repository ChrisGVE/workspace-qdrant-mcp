/**
 * Query operations for project components from the project_components table.
 *
 * Returns detected workspace components (Cargo, npm, directory fallback)
 * that the daemon persists during file processing.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
import { handleTableNotFound } from './helpers.js';

// ── Types ────────────────────────────────────────────────────────────────

export interface ComponentEntry {
  componentName: string;
  basePath: string;
  source: string;
}

// ── Queries ──────────────────────────────────────────────────────────────

/**
 * List project components from the daemon's project_components table.
 *
 * Returns detected workspace components (Cargo, npm, directory fallback)
 * that the daemon persists during file processing.
 */
export function listProjectComponents(
  db: DatabaseType | null,
  watchFolderId: string,
): DegradedQueryResult<ComponentEntry[]> {
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
      .prepare(
        `SELECT component_name, base_path, source
         FROM project_components
         WHERE watch_folder_id = ?
         ORDER BY component_name ASC`,
      )
      .all(watchFolderId) as Array<{
      component_name: string;
      base_path: string;
      source: string;
    }>;

    return {
      data: rows.map(row => ({
        componentName: row.component_name,
        basePath: row.base_path,
        source: row.source,
      })),
      status: 'ok',
    };
  } catch (error) {
    return handleTableNotFound(error, [], 'project_components');
  }
}
