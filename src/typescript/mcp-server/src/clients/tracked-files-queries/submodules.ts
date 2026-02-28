/**
 * Query operations for submodule data from watch_folders.
 *
 * Queries watch_folders where parent_watch_id matches the project's
 * watch_id to enumerate Git submodules.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import type { DegradedQueryResult } from '../sqlite-state-manager.js';
import { handleTableNotFound } from './helpers.js';

// ── Types ────────────────────────────────────────────────────────────────

export interface SubmoduleEntry {
  submodulePath: string;
  repoName: string;
}

// ── Queries ──────────────────────────────────────────────────────────────

/**
 * List submodules for a project.
 *
 * Queries watch_folders where parent_watch_id matches the project's watch_id.
 * Extracts repo name from git_remote_url (last path segment minus .git suffix).
 */
export function listSubmodules(
  db: DatabaseType | null,
  watchFolderId: string,
): DegradedQueryResult<SubmoduleEntry[]> {
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
        `SELECT submodule_path, git_remote_url
         FROM watch_folders
         WHERE parent_watch_id = ?
         ORDER BY submodule_path ASC`,
      )
      .all(watchFolderId) as Array<{
      submodule_path: string | null;
      git_remote_url: string | null;
    }>;

    const entries: SubmoduleEntry[] = [];
    for (const row of rows) {
      if (!row.submodule_path) continue;
      entries.push({
        submodulePath: row.submodule_path,
        repoName: extractRepoName(row.git_remote_url, row.submodule_path),
      });
    }

    return { data: entries, status: 'ok' };
  } catch (error) {
    return handleTableNotFound(error, [], 'watch_folders');
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/**
 * Extract repository name from a git remote URL.
 *
 * Examples:
 *   https://github.com/user/repo.git  → repo
 *   git@github.com:user/repo.git      → repo
 *   https://github.com/user/repo      → repo
 *
 * Falls back to the last segment of submodulePath if URL is null or unparseable.
 */
export function extractRepoName(
  gitRemoteUrl: string | null,
  submodulePath: string,
): string {
  if (gitRemoteUrl) {
    // Remove trailing slash and .git suffix
    const cleaned = gitRemoteUrl.replace(/\/+$/, '').replace(/\.git$/, '');
    const lastSegment = cleaned.split('/').pop();
    // Handle git@host:user/repo format
    if (lastSegment) {
      const colonSplit = lastSegment.split(':').pop();
      if (colonSplit) return colonSplit;
    }
  }

  // Fallback: last segment of the submodule path
  return submodulePath.split('/').filter(Boolean).pop() ?? submodulePath;
}
