/**
 * Read-only client for the FTS5 / file_metadata search database.
 *
 * Backs the admin UI's "largest files" view (Task #4 of the FTS5
 * size-guard series). The Rust daemon owns search.db writes — this
 * client opens it `readonly` so accidental schema drift / mutation
 * surfaces immediately. The daemon writes via WAL, so concurrent
 * reads from this handle don't interfere.
 *
 * NOT a `SqliteStateManager` extension because that class is wired
 * for state.db and a single dbPath; mixing the two would muddy ADR-003
 * ownership semantics. Both stay separate and read-only.
 */

import Database, { type Database as DatabaseType } from 'better-sqlite3';
import { existsSync } from 'node:fs';

import { getSearchDatabasePath } from '../utils/paths.js';

export interface SearchDbReaderConfig {
  dbPath?: string;
}

export interface LargeFileRow {
  file_id: number;
  tenant_id: string;
  /** Null in storage maps to the literal `"(none)"` so the JSON stays uniform. */
  branch: string;
  file_path: string;
  /** May be null for rows ingested before search.db v7. */
  size_bytes: number | null;
  /** Always 0 or 1 (search.db v8 `INTEGER NOT NULL DEFAULT 0`). */
  fts5_skipped: number;
}

export interface ListLargeFilesOptions {
  limit?: number;
  tenantId?: string;
  branch?: string;
  /** When true, return only rows where `fts5_skipped = 1`. */
  skippedOnly?: boolean;
}

export interface ChurnFileRow {
  file_id: number;
  tenant_id: string;
  /** Null in storage maps to the literal `"(none)"` so the JSON stays uniform. */
  branch: string;
  file_path: string;
  /** Number of times the daemon has (re)indexed this file's content (search.db v9). */
  reindex_count: number;
  /** RFC3339 UTC of first index; null for rows written before search.db v9. */
  first_indexed_at: string | null;
  /** May be null for rows ingested before search.db v7. */
  size_bytes: number | null;
}

export interface ListChurnFilesOptions {
  limit?: number;
  tenantId?: string;
  branch?: string;
  /** Only return files re-indexed at least this many times. Default 2 (≥1 re-index). */
  minReindexCount?: number;
}

export type ReaderStatus =
  | { status: 'ok' }
  | { status: 'degraded'; reason: 'database_not_found' | 'database_error'; message: string };

const DEFAULT_LIMIT = 50;
const MAX_LIMIT = 500;

export class SearchDbReader {
  private db: DatabaseType | null = null;
  private readonly dbPath: string;
  private initialized = false;

  constructor(config: SearchDbReaderConfig = {}) {
    this.dbPath = config.dbPath ?? getSearchDatabasePath();
  }

  /** Lazy open; returns degraded status if search.db is missing or won't open. */
  initialize(): ReaderStatus {
    if (this.initialized) return { status: 'ok' };
    if (!existsSync(this.dbPath)) {
      return {
        status: 'degraded',
        reason: 'database_not_found',
        message: `search.db not found at ${this.dbPath}. Daemon has not initialized yet.`,
      };
    }
    try {
      this.db = new Database(this.dbPath, { readonly: true, fileMustExist: true });
      this.initialized = true;
      return { status: 'ok' };
    } catch (error) {
      return {
        status: 'degraded',
        reason: 'database_error',
        message: `Failed to open search.db: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
      this.initialized = false;
    }
  }

  isConnected(): boolean {
    return this.initialized && this.db !== null;
  }

  getDatabasePath(): string {
    return this.dbPath;
  }

  /**
   * Return the N largest files in `file_metadata`, optionally filtered by
   * tenant_id and/or branch. Sort: `size_bytes DESC NULLS LAST`.
   *
   * Bounded by MAX_LIMIT to keep the admin UI responsive (and discourage
   * "give me everything" queries that the Prometheus gauges already cover
   * in aggregate form).
   */
  listLargestFiles(options: ListLargeFilesOptions = {}): LargeFileRow[] {
    const status = this.initialize();
    if (status.status !== 'ok' || !this.db) {
      return [];
    }
    const limit = Math.min(Math.max(options.limit ?? DEFAULT_LIMIT, 1), MAX_LIMIT);

    // Build WHERE clause dynamically — better-sqlite3 binds named params
    // safely. Branch=null is filterable via the special "(none)" sentinel
    // to match the Prometheus / Rust convention.
    const where: string[] = [];
    const params: Record<string, string | number> = { limit };
    if (options.tenantId) {
      where.push('tenant_id = @tenantId');
      params['tenantId'] = options.tenantId;
    }
    if (options.branch === '(none)') {
      where.push('branch IS NULL');
    } else if (options.branch) {
      where.push('branch = @branch');
      params['branch'] = options.branch;
    }
    if (options.skippedOnly) {
      where.push('fts5_skipped = 1');
    }
    const whereSql = where.length > 0 ? `WHERE ${where.join(' AND ')}` : '';

    const sql = `
      SELECT
        file_id,
        tenant_id,
        COALESCE(branch, '(none)') AS branch,
        file_path,
        size_bytes,
        fts5_skipped
      FROM file_metadata
      ${whereSql}
      ORDER BY size_bytes DESC NULLS LAST, file_id DESC
      LIMIT @limit
    `;

    try {
      return this.db.prepare(sql).all(params) as LargeFileRow[];
    } catch (error) {
      // file_metadata may not exist on a fresh search.db that hasn't
      // run migration v4 yet. Returning [] is friendlier than throwing
      // because the admin UI polls this on every snapshot refresh.
      return [];
    }
  }

  /**
   * Return the N most-churned files in `file_metadata`, ranked by
   * `reindex_count DESC` (search.db v9). High counts flag files whose
   * content changes constantly — typically IDE/build-generated artifacts
   * (`.idea/`, `target/`, lockfiles, codegen output) that are good ignore
   * candidates. The caller pairs `reindex_count` with `first_indexed_at`
   * to derive a churn rate.
   *
   * Returns [] on any error (e.g. a pre-v9 search.db lacking the column),
   * matching `listLargestFiles` so the admin UI degrades gracefully.
   */
  listChurnFiles(options: ListChurnFilesOptions = {}): ChurnFileRow[] {
    const status = this.initialize();
    if (status.status !== 'ok' || !this.db) {
      return [];
    }
    const limit = Math.min(Math.max(options.limit ?? DEFAULT_LIMIT, 1), MAX_LIMIT);
    const minCount = Math.max(options.minReindexCount ?? 2, 1);

    const where: string[] = ['reindex_count >= @minCount'];
    const params: Record<string, string | number> = { limit, minCount };
    if (options.tenantId) {
      where.push('tenant_id = @tenantId');
      params['tenantId'] = options.tenantId;
    }
    if (options.branch === '(none)') {
      where.push('branch IS NULL');
    } else if (options.branch) {
      where.push('branch = @branch');
      params['branch'] = options.branch;
    }

    const sql = `
      SELECT
        file_id,
        tenant_id,
        COALESCE(branch, '(none)') AS branch,
        file_path,
        reindex_count,
        first_indexed_at,
        size_bytes
      FROM file_metadata
      WHERE ${where.join(' AND ')}
      ORDER BY reindex_count DESC, file_id DESC
      LIMIT @limit
    `;

    try {
      return this.db.prepare(sql).all(params) as ChurnFileRow[];
    } catch (error) {
      // Pre-v9 search.db has no `reindex_count` column — degrade to empty
      // rather than throwing, since the admin UI polls this.
      return [];
    }
  }
}
