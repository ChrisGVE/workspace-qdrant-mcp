/**
 * Unified queue enqueue (via gRPC) and stats operations (direct SQLite read).
 *
 * Enqueue operations are sent to the daemon via gRPC. Read-only queue
 * statistics remain as direct SQLite queries.
 */

import type { Database as DatabaseType } from 'better-sqlite3';

import type { QueueItemType, QueueOperation, QueueStatus, QueueStats } from '../types/state.js';

import type { DaemonClient } from './daemon-client.js';
import { VALID_ITEM_TYPES, VALID_OPERATIONS } from './queue-payload-builders.js';
import type { DegradedQueryResult, EnqueueResult } from './sqlite-state-manager.js';

/** Return a degraded result for missing tables, or re-throw unknown errors. */
function handleTableNotFoundOrThrow<T>(
  error: unknown,
  message: string
): DegradedQueryResult<T | null> {
  const msg = error instanceof Error ? error.message : String(error);
  if (msg.includes('no such table')) {
    return { data: null, status: 'degraded', reason: 'table_not_found', message };
  }
  throw error;
}

/** Validate enqueue parameters. Throws on invalid input. */
function validateEnqueueParams(
  itemType: QueueItemType,
  op: QueueOperation,
  tenantId: string,
  collection: string,
  priority: number
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

/** Build the gRPC enqueue request object. */
function buildEnqueueRequest(
  itemType: QueueItemType,
  op: QueueOperation,
  tenantId: string,
  collection: string,
  payload: Record<string, unknown>,
  branch: string,
  metadata?: Record<string, unknown>
): {
  item_type: string;
  op: string;
  tenant_id: string;
  collection: string;
  payload_json: string;
  branch: string;
  metadata_json?: string;
} {
  const request: {
    item_type: string;
    op: string;
    tenant_id: string;
    collection: string;
    payload_json: string;
    branch: string;
    metadata_json?: string;
  } = {
    item_type: itemType,
    op,
    tenant_id: tenantId,
    collection,
    payload_json: JSON.stringify(payload, Object.keys(payload).sort()),
    branch,
  };
  if (metadata) request.metadata_json = JSON.stringify(metadata);
  return request;
}

/**
 * Enqueue an item to the unified queue via daemon gRPC.
 *
 * Returns a degraded result when the daemon is unavailable rather than
 * throwing, so callers can surface a helpful message to the LLM.
 */
/** Map a successful daemon enqueue response to a DegradedQueryResult. */
function enqueueSuccess(response: {
  queue_id: string;
  is_new: boolean;
  idempotency_key: string;
}): DegradedQueryResult<EnqueueResult | null> {
  return {
    data: {
      queueId: response.queue_id,
      isNew: response.is_new,
      idempotencyKey: response.idempotency_key,
    },
    status: 'ok',
  };
}

export async function enqueueUnified(
  daemonClient: DaemonClient | null,
  itemType: QueueItemType,
  op: QueueOperation,
  tenantId: string,
  collection: string,
  payload: Record<string, unknown>,
  priority: number,
  branch: string,
  metadata?: Record<string, unknown>
): Promise<DegradedQueryResult<EnqueueResult | null>> {
  if (!daemonClient) {
    return {
      data: null,
      status: 'degraded',
      reason: 'daemon_unavailable',
      message: 'Daemon not available. Start memexd to process writes.',
    };
  }

  validateEnqueueParams(itemType, op, tenantId, collection, priority);

  try {
    const request = buildEnqueueRequest(
      itemType,
      op,
      tenantId,
      collection,
      payload,
      branch,
      metadata
    );
    const response = await daemonClient.enqueueItem(request);
    return enqueueSuccess(response);
  } catch (error) {
    return {
      data: null,
      status: 'degraded',
      reason: 'daemon_error',
      message: `Daemon write failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

/** Query all raw stats from the unified_queue table. */
function queryRawQueueStats(db: DatabaseType): QueueStats {
  const statusCounts = db
    .prepare('SELECT status, COUNT(*) as count FROM unified_queue GROUP BY status')
    .all() as Array<{ status: QueueStatus; count: number }>;

  const typeCounts = db
    .prepare(
      "SELECT item_type, COUNT(*) as count FROM unified_queue WHERE status = 'pending' GROUP BY item_type"
    )
    .all() as Array<{ item_type: QueueItemType; count: number }>;

  const collectionCounts = db
    .prepare(
      "SELECT collection, COUNT(*) as count FROM unified_queue WHERE status = 'pending' GROUP BY collection"
    )
    .all() as Array<{ collection: string; count: number }>;

  const staleCount = db
    .prepare(
      "SELECT COUNT(*) as count FROM unified_queue WHERE status = 'in_progress' AND lease_expires_at < datetime('now')"
    )
    .get() as { count: number };

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
 * Read-only — queries SQLite directly.
 */
export function getQueueStats(db: DatabaseType | null): DegradedQueryResult<QueueStats | null> {
  if (!db) {
    return {
      data: null,
      status: 'degraded',
      reason: 'database_not_found',
      message: 'Database not initialized',
    };
  }

  try {
    return { data: queryRawQueueStats(db), status: 'ok' };
  } catch (error) {
    return handleTableNotFoundOrThrow(error, 'Table unified_queue not found.');
  }
}
