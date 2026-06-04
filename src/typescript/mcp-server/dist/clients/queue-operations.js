/**
 * Unified queue enqueue (via gRPC) and stats operations (direct SQLite read).
 *
 * Enqueue operations are sent to the daemon via gRPC. Read-only queue
 * statistics remain as direct SQLite queries.
 */
import { VALID_ITEM_TYPES, VALID_OPERATIONS } from './queue-payload-builders.js';
/** Return a degraded result for missing tables, or re-throw unknown errors. */
function handleTableNotFoundOrThrow(error, message) {
    const msg = error instanceof Error ? error.message : String(error);
    if (msg.includes('no such table')) {
        return { data: null, status: 'degraded', reason: 'table_not_found', message };
    }
    throw error;
}
/**
 * Deterministic JSON stringifier with recursively sorted object keys.
 *
 * Used to produce a stable `payload_json` for queue idempotency hashing.
 * `JSON.stringify`'s replacer-array form only includes the listed keys at
 * every depth, which silently drops nested fields — we cannot rely on it
 * here (F-008).
 */
function stableStringify(value) {
    if (value === null || typeof value !== 'object') {
        return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
        return `[${value.map((v) => stableStringify(v)).join(',')}]`;
    }
    const obj = value;
    const sortedKeys = Object.keys(obj).sort();
    const entries = sortedKeys.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`);
    return `{${entries.join(',')}}`;
}
/** Validate enqueue parameters. Throws on invalid input. */
function validateEnqueueParams(itemType, op, tenantId, collection, priority) {
    if (!VALID_ITEM_TYPES.includes(itemType)) {
        throw new Error(`Invalid item type: ${itemType}`);
    }
    const validOps = VALID_OPERATIONS[itemType];
    if (!validOps?.includes(op)) {
        throw new Error(`Invalid operation '${op}' for item type '${itemType}'`);
    }
    if (!tenantId.trim())
        throw new Error('tenant_id cannot be empty');
    if (!collection.trim())
        throw new Error('collection cannot be empty');
    if (priority < 0 || priority > 10)
        throw new Error('Priority must be between 0 and 10');
}
/** Build the gRPC enqueue request object. */
function buildEnqueueRequest(itemType, op, tenantId, collection, payload, branch, metadata) {
    const request = {
        item_type: itemType,
        op,
        tenant_id: tenantId,
        collection,
        payload_json: stableStringify(payload),
        branch,
    };
    if (metadata)
        request.metadata_json = JSON.stringify(metadata);
    return request;
}
/**
 * Enqueue an item to the unified queue via daemon gRPC.
 *
 * Returns a degraded result when the daemon is unavailable rather than
 * throwing, so callers can surface a helpful message to the LLM.
 */
/** Map a successful daemon enqueue response to a DegradedQueryResult. */
function enqueueSuccess(response) {
    return {
        data: {
            queueId: response.queue_id,
            isNew: response.is_new,
            idempotencyKey: response.idempotency_key,
        },
        status: 'ok',
    };
}
export async function enqueueUnified(daemonClient, itemType, op, tenantId, collection, payload, priority, branch, metadata) {
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
        const request = buildEnqueueRequest(itemType, op, tenantId, collection, payload, branch, metadata);
        const response = await daemonClient.enqueueItem(request);
        return enqueueSuccess(response);
    }
    catch (error) {
        return {
            data: null,
            status: 'degraded',
            reason: 'daemon_error',
            message: `Daemon write failed: ${error instanceof Error ? error.message : String(error)}`,
        };
    }
}
/** Query all raw stats from the unified_queue table. */
function queryRawQueueStats(db) {
    const statusCounts = db
        .prepare('SELECT status, COUNT(*) as count FROM unified_queue GROUP BY status')
        .all();
    const typeCounts = db
        .prepare("SELECT item_type, COUNT(*) as count FROM unified_queue WHERE status = 'pending' GROUP BY item_type")
        .all();
    const collectionCounts = db
        .prepare("SELECT collection, COUNT(*) as count FROM unified_queue WHERE status = 'pending' GROUP BY collection")
        .all();
    const staleCount = db
        .prepare("SELECT COUNT(*) as count FROM unified_queue WHERE status = 'in_progress' AND lease_until < datetime('now')")
        .get();
    const statusMap = new Map(statusCounts.map((r) => [r.status, r.count]));
    const typeMap = {};
    for (const row of typeCounts)
        typeMap[row.item_type] = row.count;
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
export function getQueueStats(db) {
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
    }
    catch (error) {
        return handleTableNotFoundOrThrow(error, 'Table unified_queue not found.');
    }
}
//# sourceMappingURL=queue-operations.js.map