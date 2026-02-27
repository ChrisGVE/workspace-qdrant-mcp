/**
 * Unified queue enqueue and stats operations for SqliteStateManager.
 *
 * Handles queue item insertion with idempotency and statistics queries.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import { utcNow } from '../utils/timestamps.js';

import type {
  QueueItemType,
  QueueOperation,
  QueueStatus,
  QueueStats,
} from '../types/state.js';

import { generateIdempotencyKey, VALID_ITEM_TYPES, VALID_OPERATIONS } from './queue-payload-builders.js';
import type { DegradedQueryResult, EnqueueResult } from './sqlite-state-manager.js';

/** Return a degraded result for missing tables, or re-throw unknown errors. */
function handleTableNotFoundOrThrow<T>(error: unknown, message: string): DegradedQueryResult<T | null> {
  const msg = error instanceof Error ? error.message : String(error);
  if (msg.includes('no such table')) {
    return { data: null, status: 'degraded', reason: 'table_not_found', message };
  }
  throw error;
}

/** Validate enqueue parameters. Throws on invalid input. */
function validateEnqueueParams(
  itemType: QueueItemType, op: QueueOperation,
  tenantId: string, collection: string, priority: number,
): void {
  if (!VALID_ITEM_TYPES.includes(itemType)) {
    throw new Error(`Invalid item type: ${itemType}`);
  }
  const validOps = VALID_OPERATIONS[itemType];
  if (!validOps?.includes(op)) {
    throw new Error(`Invalid operation '${op}' for item type '${itemType}'`);
  }
  if (!tenantId.trim()) throw new Error('tenant_id cannot be empty');
  if (!collection.trim()) throw new Error('collection cannot be empty');
  if (priority < 0 || priority > 10) throw new Error('Priority must be between 0 and 10');
}

/** Insert-or-deduplicate inside a transaction. */
function executeEnqueueTransaction(
  db: DatabaseType,
  itemType: QueueItemType, op: QueueOperation,
  tenantId: string, collection: string, priority: number, branch: string,
  payload: Record<string, unknown>, metadata?: Record<string, unknown>,
): EnqueueResult {
  const idempotencyKey = generateIdempotencyKey(itemType, op, tenantId, collection, payload);
  const queueId = randomUUID();
  const now = utcNow();
  const payloadJson = JSON.stringify(payload, Object.keys(payload).sort());
  const metadataJson = metadata ? JSON.stringify(metadata) : '{}';

  return db.transaction(() => {
    const existing = db.prepare(
      'SELECT queue_id FROM unified_queue WHERE idempotency_key = ?'
    ).get(idempotencyKey) as { queue_id: string } | undefined;

    if (existing) {
      return { queueId: existing.queue_id, isNew: false, idempotencyKey };
    }

    db.prepare(
      `INSERT INTO unified_queue
      (queue_id, item_type, op, tenant_id, collection, priority, status,
       idempotency_key, payload_json, branch, metadata, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)`
    ).run(queueId, itemType, op, tenantId, collection, priority,
          idempotencyKey, payloadJson, branch, metadataJson, now, now);

    return { queueId, isNew: true, idempotencyKey };
  })();
}

/**
 * Enqueue an item to the unified queue with idempotency support.
 */
export function enqueueUnified(
  db: DatabaseType | null,
  itemType: QueueItemType, op: QueueOperation,
  tenantId: string, collection: string,
  payload: Record<string, unknown>, priority: number, branch: string,
  metadata?: Record<string, unknown>,
): DegradedQueryResult<EnqueueResult | null> {
  if (!db) {
    return { data: null, status: 'degraded', reason: 'database_not_found',
             message: 'Database not initialized. Start daemon first.' };
  }

  validateEnqueueParams(itemType, op, tenantId, collection, priority);

  try {
    const result = executeEnqueueTransaction(
      db, itemType, op, tenantId, collection, priority, branch, payload, metadata,
    );
    return { data: result, status: 'ok' };
  } catch (error) {
    return handleTableNotFoundOrThrow(error,
      'Table unified_queue not found. Daemon has not initialized database.');
  }
}

/** Query all raw stats from the unified_queue table. */
function queryRawQueueStats(db: DatabaseType): QueueStats {
  const statusCounts = db.prepare(
    'SELECT status, COUNT(*) as count FROM unified_queue GROUP BY status'
  ).all() as Array<{ status: QueueStatus; count: number }>;

  const typeCounts = db.prepare(
    "SELECT item_type, COUNT(*) as count FROM unified_queue WHERE status = 'pending' GROUP BY item_type"
  ).all() as Array<{ item_type: QueueItemType; count: number }>;

  const collectionCounts = db.prepare(
    "SELECT collection, COUNT(*) as count FROM unified_queue WHERE status = 'pending' GROUP BY collection"
  ).all() as Array<{ collection: string; count: number }>;

  const staleCount = db.prepare(
    "SELECT COUNT(*) as count FROM unified_queue WHERE status = 'in_progress' AND lease_expires_at < datetime('now')"
  ).get() as { count: number };

  const statusMap = new Map(statusCounts.map((r) => [r.status, r.count]));
  const typeMap: Record<QueueItemType, number> = {} as Record<QueueItemType, number>;
  for (const row of typeCounts) typeMap[row.item_type] = row.count;

  return {
    total_pending: statusMap.get('pending') ?? 0,
    total_in_progress: statusMap.get('in_progress') ?? 0,
    total_done: statusMap.get('done') ?? 0,
    total_failed: statusMap.get('failed') ?? 0,
    by_item_type: typeMap,
    by_collection: collectionCounts.map((r) => ({ collection: r.collection, count: r.count })),
    stale_items_count: staleCount.count,
  };
}

/**
 * Get queue statistics grouped by status, item type, and collection.
 */
export function getQueueStats(
  db: DatabaseType | null,
): DegradedQueryResult<QueueStats | null> {
  if (!db) {
    return { data: null, status: 'degraded', reason: 'database_not_found', message: 'Database not initialized' };
  }

  try {
    return { data: queryRawQueueStats(db), status: 'ok' };
  } catch (error) {
    return handleTableNotFoundOrThrow(error, 'Table unified_queue not found.');
  }
}
