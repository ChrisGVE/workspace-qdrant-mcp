/**
 * SQLite state manager for TypeScript MCP server
 *
 * Provides access to the unified_queue and watch_folders tables.
 * Implements graceful handling when daemon hasn't initialized the database.
 *
 * IMPORTANT: Per ADR-003, the Rust daemon owns the SQLite database and schema.
 * This client may read/write to tables but must NOT create tables or run migrations.
 *
 * NOTE: Project lookups query watch_folders WHERE collection = 'projects'.
 * The old registered_projects table has been consolidated into watch_folders.
 */

import Database, { type Database as DatabaseType } from 'better-sqlite3';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';

import type {
  UnifiedQueueItem,
  QueueItemType,
  QueueOperation,
  QueueStatus,
  QueueStats,
  RegisteredProject,
  ContentPayload,
  MemoryPayload,
  LibraryPayload,
} from '../types/state.js';

// Re-export types
export type {
  UnifiedQueueItem,
  QueueItemType,
  QueueOperation,
  QueueStatus,
  QueueStats,
  RegisteredProject,
  ContentPayload,
  MemoryPayload,
  LibraryPayload,
};

// Default database path
const DEFAULT_DB_PATH = join(homedir(), '.workspace-qdrant', 'state.db');

// Valid item types
const VALID_ITEM_TYPES: QueueItemType[] = [
  'content',
  'file',
  'folder',
  'project',
  'library',
  'memory',
];

// Valid operations per item type
const VALID_OPERATIONS: Record<QueueItemType, QueueOperation[]> = {
  content: ['ingest', 'update', 'delete'],
  file: ['ingest', 'update', 'delete'],
  folder: ['ingest', 'delete', 'scan'],
  project: ['ingest', 'update', 'delete'],
  library: ['ingest', 'update', 'delete'],
  memory: ['ingest', 'update', 'delete'],
};

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
 * Generate idempotency key for queue deduplication
 *
 * Format matches Python and Rust implementations:
 * Input: {item_type}|{op}|{tenant_id}|{collection}|{payload_json}
 * Output: SHA256 hash truncated to 32 hex characters
 */
export function generateIdempotencyKey(
  itemType: QueueItemType,
  op: QueueOperation,
  tenantId: string,
  collection: string,
  payload: Record<string, unknown>
): string {
  // Serialize payload with sorted keys (matching Python json.dumps(sort_keys=True))
  const payloadJson = JSON.stringify(payload, Object.keys(payload).sort());

  // Construct canonical input string
  const inputString = `${itemType}|${op}|${tenantId}|${collection}|${payloadJson}`;

  // Hash and truncate to 32 hex chars
  return createHash('sha256').update(inputString, 'utf-8').digest('hex').slice(0, 32);
}

/**
 * Build content payload for queue
 */
export function buildContentPayload(
  content: string,
  sourceType: string,
  mainTag?: string,
  fullTag?: string
): ContentPayload {
  return {
    content,
    source_type: sourceType,
    main_tag: mainTag,
    full_tag: fullTag,
  };
}

/**
 * Build memory payload for queue
 */
export function buildMemoryPayload(
  label: string,
  content: string,
  scope: 'global' | 'project',
  projectId?: string
): MemoryPayload {
  return {
    label,
    content,
    scope,
    project_id: projectId,
  };
}

/**
 * Build library payload for queue
 */
export function buildLibraryPayload(
  libraryName: string,
  content?: string,
  source?: string,
  url?: string
): LibraryPayload {
  return {
    library_name: libraryName,
    content,
    source,
    url,
  };
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

  /**
   * Initialize the state manager
   *
   * Opens database connection if file exists.
   * Does NOT create the database - that's the daemon's responsibility.
   */
  initialize(): DegradedQueryResult<boolean> {
    if (this.initialized) {
      return { data: true, status: 'ok' };
    }

    if (!existsSync(this.dbPath)) {
      return {
        data: false,
        status: 'degraded',
        reason: 'database_not_found',
        message: `Database not found at ${this.dbPath}. Daemon has not initialized yet.`,
      };
    }

    try {
      this.db = new Database(this.dbPath, {
        readonly: false,
        fileMustExist: true,
      });

      // Enable WAL mode for better concurrent access
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

  /**
   * Close the database connection
   */
  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
      this.initialized = false;
    }
  }

  /**
   * Check if connected to database
   */
  isConnected(): boolean {
    return this.initialized && this.db !== null;
  }

  /**
   * Get database path
   */
  getDatabasePath(): string {
    return this.dbPath;
  }

  // ============================================================================
  // Unified Queue Methods
  // ============================================================================

  /**
   * Enqueue an item to the unified queue with idempotency support
   *
   * @param itemType Type of queue item
   * @param op Operation type
   * @param tenantId Project/tenant identifier
   * @param collection Target collection name
   * @param payload Payload dictionary
   * @param priority Priority level 0-10 (default 5)
   * @param branch Git branch (default 'main')
   * @param metadata Optional additional metadata
   */
  enqueueUnified(
    itemType: QueueItemType,
    op: QueueOperation,
    tenantId: string,
    collection: string,
    payload: Record<string, unknown>,
    priority = 5,
    branch = 'main',
    metadata?: Record<string, unknown>
  ): DegradedQueryResult<EnqueueResult | null> {
    if (!this.db) {
      return {
        data: null,
        status: 'degraded',
        reason: 'database_not_found',
        message: 'Database not initialized. Start daemon first.',
      };
    }

    // Validate inputs
    if (!VALID_ITEM_TYPES.includes(itemType)) {
      throw new Error(`Invalid item type: ${itemType}`);
    }

    const validOps = VALID_OPERATIONS[itemType];
    if (!validOps?.includes(op)) {
      throw new Error(`Invalid operation '${op}' for item type '${itemType}'`);
    }

    if (!tenantId.trim()) {
      throw new Error('tenant_id cannot be empty');
    }
    if (!collection.trim()) {
      throw new Error('collection cannot be empty');
    }
    if (priority < 0 || priority > 10) {
      throw new Error('Priority must be between 0 and 10');
    }

    try {
      // Generate idempotency key
      const idempotencyKey = generateIdempotencyKey(itemType, op, tenantId, collection, payload);
      const queueId = randomUUID();
      const now = new Date().toISOString();
      const payloadJson = JSON.stringify(payload, Object.keys(payload).sort());
      const metadataJson = metadata ? JSON.stringify(metadata) : '{}';

      // Use transaction for atomicity
      const result = this.db.transaction(() => {
        // Check for existing item with same idempotency key
        const existing = this.db!.prepare(
          'SELECT queue_id FROM unified_queue WHERE idempotency_key = ?'
        ).get(idempotencyKey) as { queue_id: string } | undefined;

        if (existing) {
          return {
            queueId: existing.queue_id,
            isNew: false,
            idempotencyKey,
          };
        }

        // Insert new queue item
        this.db!.prepare(
          `
          INSERT INTO unified_queue
          (queue_id, item_type, op, tenant_id, collection, priority, status,
           idempotency_key, payload_json, branch, metadata, created_at, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
          `
        ).run(
          queueId,
          itemType,
          op,
          tenantId,
          collection,
          priority,
          idempotencyKey,
          payloadJson,
          branch,
          metadataJson,
          now,
          now
        );

        return {
          queueId,
          isNew: true,
          idempotencyKey,
        };
      })();

      return { data: result, status: 'ok' };
    } catch (error) {
      // Check for "table not found" error
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.includes('no such table')) {
        return {
          data: null,
          status: 'degraded',
          reason: 'table_not_found',
          message: 'Table unified_queue not found. Daemon has not initialized database.',
        };
      }
      throw error;
    }
  }

  /**
   * Get queue statistics
   */
  getQueueStats(): DegradedQueryResult<QueueStats | null> {
    if (!this.db) {
      return {
        data: null,
        status: 'degraded',
        reason: 'database_not_found',
        message: 'Database not initialized',
      };
    }

    try {
      // Get counts by status
      const statusCounts = this.db
        .prepare(
          `
        SELECT status, COUNT(*) as count
        FROM unified_queue
        GROUP BY status
      `
        )
        .all() as Array<{ status: QueueStatus; count: number }>;

      // Get counts by item type
      const typeCounts = this.db
        .prepare(
          `
        SELECT item_type, COUNT(*) as count
        FROM unified_queue
        WHERE status = 'pending'
        GROUP BY item_type
      `
        )
        .all() as Array<{ item_type: QueueItemType; count: number }>;

      // Get counts by collection
      const collectionCounts = this.db
        .prepare(
          `
        SELECT collection, COUNT(*) as count
        FROM unified_queue
        WHERE status = 'pending'
        GROUP BY collection
      `
        )
        .all() as Array<{ collection: string; count: number }>;

      // Get stale items (in_progress but lease expired)
      const staleCount = this.db
        .prepare(
          `
        SELECT COUNT(*) as count
        FROM unified_queue
        WHERE status = 'in_progress'
        AND lease_expires_at < datetime('now')
      `
        )
        .get() as { count: number };

      // Build stats object
      const statusMap = new Map(statusCounts.map((r) => [r.status, r.count]));
      const typeMap: Record<QueueItemType, number> = {} as Record<QueueItemType, number>;
      for (const row of typeCounts) {
        typeMap[row.item_type] = row.count;
      }

      const stats: QueueStats = {
        total_pending: statusMap.get('pending') ?? 0,
        total_in_progress: statusMap.get('in_progress') ?? 0,
        total_done: statusMap.get('done') ?? 0,
        total_failed: statusMap.get('failed') ?? 0,
        by_item_type: typeMap,
        by_collection: collectionCounts.map((r) => ({ collection: r.collection, count: r.count })),
        stale_items_count: staleCount.count,
      };

      return { data: stats, status: 'ok' };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.includes('no such table')) {
        return {
          data: null,
          status: 'degraded',
          reason: 'table_not_found',
          message: 'Table unified_queue not found.',
        };
      }
      throw error;
    }
  }

  // ============================================================================
  // Project Methods
  // ============================================================================

  /**
   * Get project by path
   *
   * Fetches project_id from watch_folders table (collection = 'projects').
   * Used by MCP server to get project_id for the current working directory.
   *
   * NOTE: Queries watch_folders instead of the deprecated registered_projects table.
   */
  getProjectByPath(projectPath: string): DegradedQueryResult<RegisteredProject | null> {
    if (!this.db) {
      return {
        data: null,
        status: 'degraded',
        reason: 'database_not_found',
        message: 'Database not initialized',
      };
    }

    try {
      const project = this.db
        .prepare(
          `
        SELECT tenant_id, path, git_remote_url, remote_hash,
               disambiguation_path, is_active,
               created_at, updated_at, last_activity_at
        FROM watch_folders
        WHERE path = ? AND collection = 'projects'
      `
        )
        .get(projectPath) as
        | {
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
        | undefined;

      if (!project) {
        return { data: null, status: 'ok' };
      }

      // Derive container_folder from path (basename)
      const containerFolder = project.path.split('/').filter(Boolean).at(-1) ?? project.path;

      return {
        data: {
          project_id: project.tenant_id,
          project_path: project.path,
          git_remote_url: project.git_remote_url ?? undefined,
          remote_hash: project.remote_hash ?? undefined,
          disambiguation_path: project.disambiguation_path ?? undefined,
          container_folder: containerFolder,
          is_active: project.is_active === 1,
          created_at: project.created_at,
          last_seen_at: project.updated_at ?? undefined,
          last_activity_at: project.last_activity_at ?? undefined,
        },
        status: 'ok',
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.includes('no such table')) {
        return {
          data: null,
          status: 'degraded',
          reason: 'table_not_found',
          message: 'Table watch_folders not found. Daemon has not initialized database.',
        };
      }
      throw error;
    }
  }

  /**
   * Get project by ID
   *
   * NOTE: Queries watch_folders instead of the deprecated registered_projects table.
   */
  getProjectById(projectId: string): DegradedQueryResult<RegisteredProject | null> {
    if (!this.db) {
      return {
        data: null,
        status: 'degraded',
        reason: 'database_not_found',
        message: 'Database not initialized',
      };
    }

    try {
      const project = this.db
        .prepare(
          `
        SELECT tenant_id, path, git_remote_url, remote_hash,
               disambiguation_path, is_active,
               created_at, updated_at, last_activity_at
        FROM watch_folders
        WHERE tenant_id = ? AND collection = 'projects'
      `
        )
        .get(projectId) as
        | {
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
        | undefined;

      if (!project) {
        return { data: null, status: 'ok' };
      }

      // Derive container_folder from path (basename)
      const containerFolder = project.path.split('/').filter(Boolean).at(-1) ?? project.path;

      return {
        data: {
          project_id: project.tenant_id,
          project_path: project.path,
          git_remote_url: project.git_remote_url ?? undefined,
          remote_hash: project.remote_hash ?? undefined,
          disambiguation_path: project.disambiguation_path ?? undefined,
          container_folder: containerFolder,
          is_active: project.is_active === 1,
          created_at: project.created_at,
          last_seen_at: project.updated_at ?? undefined,
          last_activity_at: project.last_activity_at ?? undefined,
        },
        status: 'ok',
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.includes('no such table')) {
        return {
          data: null,
          status: 'degraded',
          reason: 'table_not_found',
          message: 'Table watch_folders not found. Daemon has not initialized database.',
        };
      }
      throw error;
    }
  }

  /**
   * List all active projects
   *
   * NOTE: Queries watch_folders instead of the deprecated registered_projects table.
   */
  listActiveProjects(): DegradedQueryResult<RegisteredProject[]> {
    if (!this.db) {
      return {
        data: [],
        status: 'degraded',
        reason: 'database_not_found',
        message: 'Database not initialized',
      };
    }

    try {
      const projects = this.db
        .prepare(
          `
        SELECT tenant_id, path, git_remote_url, remote_hash,
               disambiguation_path, is_active,
               created_at, updated_at, last_activity_at
        FROM watch_folders
        WHERE is_active = 1 AND collection = 'projects'
        ORDER BY last_activity_at DESC
      `
        )
        .all() as Array<{
        tenant_id: string;
        path: string;
        git_remote_url: string | null;
        remote_hash: string | null;
        disambiguation_path: string | null;
        is_active: number;
        created_at: string;
        updated_at: string | null;
        last_activity_at: string | null;
      }>;

      return {
        data: projects.map((p) => {
          // Derive container_folder from path (basename)
          const parts = p.path.split('/').filter(Boolean);
          const containerFolder: string = parts.length > 0 ? parts[parts.length - 1]! : p.path;

          return {
            project_id: p.tenant_id,
            project_path: p.path,
            git_remote_url: p.git_remote_url ?? undefined,
            remote_hash: p.remote_hash ?? undefined,
            disambiguation_path: p.disambiguation_path ?? undefined,
            container_folder: containerFolder,
            is_active: p.is_active === 1,
            created_at: p.created_at,
            last_seen_at: p.updated_at ?? undefined,
            last_activity_at: p.last_activity_at ?? undefined,
          };
        }),
        status: 'ok',
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.includes('no such table')) {
        return {
          data: [],
          status: 'degraded',
          reason: 'table_not_found',
          message: 'Table watch_folders not found. Daemon has not initialized database.',
        };
      }
      throw error;
    }
  }
}
