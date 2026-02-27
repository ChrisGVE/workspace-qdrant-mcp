/**
 * Query operations for the tracked_files table.
 *
 * Reads from the daemon-owned tracked_files table to provide
 * file listing data for the list MCP tool.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
import { handleTableNotFound } from './helpers.js';

// ── Types ────────────────────────────────────────────────────────────────

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
}

// ── Query Building ───────────────────────────────────────────────────────

interface FilterClause {
  conditions: string[];
  params: (string | number)[];
}

/** Build WHERE conditions and params from filter options. */
function buildFilterClause(
  options: Omit<ListTrackedFilesOptions, 'limit'>,
): FilterClause {
  const conditions: string[] = ['watch_folder_id = ?'];
  const params: (string | number)[] = [options.watchFolderId];
  const { path, fileType, language, extension, branch } = options;
  const includeTests = options.includeTests ?? true;

  if (path) { conditions.push('relative_path LIKE ?'); params.push(`${path}/%`); }
  if (fileType) { conditions.push('file_type = ?'); params.push(fileType); }
  if (language) { conditions.push('language = ?'); params.push(language); }
  if (extension) { conditions.push('extension = ?'); params.push(extension); }
  if (!includeTests) { conditions.push('is_test = 0'); }
  if (branch) { conditions.push('branch = ?'); params.push(branch); }

  return { conditions, params };
}

// ── Queries ──────────────────────────────────────────────────────────────

/**
 * List tracked files for a project, with optional filtering.
 *
 * Returns minimal fields needed for tree construction.
 */
export function listTrackedFiles(
  db: DatabaseType | null,
  options: ListTrackedFilesOptions,
): DegradedQueryResult<TrackedFileEntry[]> {
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

    const rows = db.prepare(sql).all(...params) as Array<{
      relative_path: string;
      file_type: string | null;
      language: string | null;
      extension: string | null;
      is_test: number;
    }>;

    return { data: rows.map(mapTrackedFileRow), status: 'ok' };
  } catch (error) {
    return handleTableNotFound(error, [], 'tracked_files');
  }
}

/**
 * Count total tracked files matching the same filters (ignoring limit).
 *
 * Used to report accurate totals when results are truncated.
 */
export function countTrackedFiles(
  db: DatabaseType | null,
  options: Omit<ListTrackedFilesOptions, 'limit'>,
): number {
  if (!db) return 0;

  try {
    const { conditions, params } = buildFilterClause(options);
    const sql = `
      SELECT COUNT(*) as cnt
      FROM tracked_files
      WHERE ${conditions.join(' AND ')}
    `;
    const row = db.prepare(sql).get(...params) as { cnt: number };
    return row.cnt;
  } catch {
    return 0;
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────

function mapTrackedFileRow(row: {
  relative_path: string;
  file_type: string | null;
  language: string | null;
  extension: string | null;
  is_test: number;
}): TrackedFileEntry {
  return {
    relativePath: row.relative_path,
    fileType: row.file_type,
    language: row.language,
    extension: row.extension,
    isTest: row.is_test === 1,
  };
}
