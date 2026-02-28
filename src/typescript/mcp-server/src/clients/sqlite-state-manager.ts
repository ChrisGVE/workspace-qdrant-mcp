/**
 * SQLite state manager for TypeScript MCP server
 *
 * Thin facade over domain-specific query modules. Provides access to the
 * unified_queue and watch_folders tables with graceful degradation.
 *
 * IMPORTANT: Per ADR-003, the Rust daemon owns the SQLite database and schema.
 * This client may read/write to tables but must NOT create tables or run migrations.
 */

import Database, { type Database as DatabaseType } from 'better-sqlite3';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

import type {
  UnifiedQueueItem,
  QueueItemType,
  QueueOperation,
  QueueStatus,
  QueueStats,
  RegisteredProject,
  ContentPayload,
  RulesPayload,
  LibraryPayload,
} from '../types/state.js';
import { PRIORITY_LOW } from '../common/native-bridge.js';

// Re-export types for consumers
export type {
  UnifiedQueueItem,
  QueueItemType,
  QueueOperation,
  QueueStatus,
  QueueStats,
  RegisteredProject,
  ContentPayload,
  RulesPayload,
  LibraryPayload,
};

// Re-export payload builder functions
export {
  generateIdempotencyKey,
  buildContentPayload,
  buildRulesPayload,
  buildLibraryPayload,
  VALID_ITEM_TYPES,
  VALID_OPERATIONS,
} from './queue-payload-builders.js';

// Import delegate modules
import * as queueOps from './queue-operations.js';
import * as projectQueries from './project-queries.js';
import * as searchEventQueries from './search-event-queries.js';
import * as tagQueries from './tag-queries.js';
import * as instanceQueries from './instance-queries.js';
import * as rulesMirrorQueries from './rules-mirror-queries.js';
import * as trackedFilesQueries from './tracked-files-queries/index.js';

// Re-export delegate types
export type { SearchEventInput, SearchEventUpdate } from './search-event-queries.js';
export type { RulesMirrorEntry } from './rules-mirror-queries.js';
export type { TrackedFileEntry, SubmoduleEntry, ComponentEntry, ListTrackedFilesOptions } from './tracked-files-queries/index.js';

// Default database path
const DEFAULT_DB_PATH = join(homedir(), '.workspace-qdrant', 'state.db');

export interface SqliteStateManagerConfig {
  dbPath?: string;
}

export interface EnqueueResult {
  queueId: string;
  isNew: boolean;
  idempotencyKey: string;
}

export interface DegradedQueryResult<T> {
  data: T;
  status: 'ok' | 'degraded';
  reason?: 'database_not_found' | 'table_not_found' | 'database_error';
  message?: string;
}

/**
 * SQLite state manager for MCP server
 *
 * Provides synchronous access to the daemon's SQLite database.
 * Uses better-sqlite3 for fast, synchronous operations.
 */
export class SqliteStateManager {
  private db: DatabaseType | null = null;
  private readonly dbPath: string;
  private initialized = false;

  constructor(config: SqliteStateManagerConfig = {}) {
    this.dbPath = config.dbPath ?? DEFAULT_DB_PATH;
  }

  // ── Core lifecycle ────────────────────────────────────────────────────

  initialize(): DegradedQueryResult<boolean> {
    if (this.initialized) return { data: true, status: 'ok' };

    if (!existsSync(this.dbPath)) {
      return {
        data: false,
        status: 'degraded',
        reason: 'database_not_found',
        message: `Database not found at ${this.dbPath}. Daemon has not initialized yet.`,
      };
    }

    try {
      this.db = new Database(this.dbPath, { readonly: false, fileMustExist: true });
      this.db.pragma('journal_mode = WAL');
      this.initialized = true;
      return { data: true, status: 'ok' };
    } catch (error) {
      return {
        data: false,
        status: 'degraded',
        reason: 'database_error',
        message: `Failed to open database: ${error instanceof Error ? error.message : 'Unknown error'}`,
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

  // ── Unified Queue (delegated) ─────────────────────────────────────────

  enqueueUnified(
    itemType: QueueItemType,
    op: QueueOperation,
    tenantId: string,
    collection: string,
    payload: Record<string, unknown>,
    priority = PRIORITY_LOW,
    branch = 'main',
    metadata?: Record<string, unknown>,
  ): DegradedQueryResult<EnqueueResult | null> {
    return queueOps.enqueueUnified(
      this.db, itemType, op, tenantId, collection, payload, priority, branch, metadata,
    );
  }

  getQueueStats(): DegradedQueryResult<QueueStats | null> {
    return queueOps.getQueueStats(this.db);
  }

  // ── Project queries (delegated) ───────────────────────────────────────

  getProjectByPath(projectPath: string) {
    return projectQueries.getProjectByPath(this.db, projectPath);
  }

  getProjectById(projectId: string) {
    return projectQueries.getProjectById(this.db, projectId);
  }

  listActiveProjects() {
    return projectQueries.listActiveProjects(this.db);
  }

  // ── Search event instrumentation (delegated) ──────────────────────────

  logSearchEvent(event: searchEventQueries.SearchEventInput): void {
    searchEventQueries.logSearchEvent(this.db, event);
  }

  updateSearchEvent(eventId: string, update: searchEventQueries.SearchEventUpdate): void {
    searchEventQueries.updateSearchEvent(this.db, eventId, update);
  }

  // ── Tag/basket queries (delegated) ────────────────────────────────────

  getMatchingTags(query: string, collection: string, tenantId?: string) {
    return tagQueries.getMatchingTags(this.db, query, collection, tenantId);
  }

  getKeywordBasketsForTags(tagIds: number[]) {
    return tagQueries.getKeywordBasketsForTags(this.db, tagIds);
  }

  listTags(collection: string, tenantId?: string, limit = 50) {
    return tagQueries.listTags(this.db, collection, tenantId, limit);
  }

  getTagHierarchy(collection: string, tenantId?: string) {
    return tagQueries.getTagHierarchy(this.db, collection, tenantId);
  }

  // ── Instance-aware queries (delegated) ────────────────────────────────

  getWatchFolderIdByTenantId(tenantId: string) {
    return instanceQueries.getWatchFolderIdByTenantId(this.db, tenantId);
  }

  getActiveBasePoints(watchFolderId: string, includeSubmodules = false) {
    return instanceQueries.getActiveBasePoints(this.db, watchFolderId, includeSubmodules);
  }

  // ── Rules mirror (delegated) ──────────────────────────────────────────

  upsertRulesMirror(entry: rulesMirrorQueries.RulesMirrorEntry): void {
    rulesMirrorQueries.upsertRulesMirror(this.db, entry);
  }

  deleteRulesMirror(ruleId: string): void {
    rulesMirrorQueries.deleteRulesMirror(this.db, ruleId);
  }

  listRulesMirror(scope?: string, tenantId?: string, limit = 50) {
    return rulesMirrorQueries.listRulesMirror(this.db, scope, tenantId, limit);
  }

  // ── Tracked files (delegated) ──────────────────────────────────────────

  listTrackedFiles(options: trackedFilesQueries.ListTrackedFilesOptions) {
    return trackedFilesQueries.listTrackedFiles(this.db, options);
  }

  countTrackedFiles(options: Omit<trackedFilesQueries.ListTrackedFilesOptions, 'limit'>) {
    return trackedFilesQueries.countTrackedFiles(this.db, options);
  }

  listSubmodules(watchFolderId: string) {
    return trackedFilesQueries.listSubmodules(this.db, watchFolderId);
  }

  listProjectComponents(watchFolderId: string) {
    return trackedFilesQueries.listProjectComponents(this.db, watchFolderId);
  }
}
