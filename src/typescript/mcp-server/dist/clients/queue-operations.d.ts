/**
 * Unified queue enqueue (via gRPC) and stats operations (direct SQLite read).
 *
 * Enqueue operations are sent to the daemon via gRPC. Read-only queue
 * statistics remain as direct SQLite queries.
 */
import type { Database as DatabaseType } from 'better-sqlite3';
import type { QueueItemType, QueueOperation, QueueStats } from '../types/state.js';
import type { DaemonClient } from './daemon-client.js';
import type { DegradedQueryResult, EnqueueResult } from './sqlite-state-manager.js';
export declare function enqueueUnified(daemonClient: DaemonClient | null, itemType: QueueItemType, op: QueueOperation, tenantId: string, collection: string, payload: Record<string, unknown>, priority: number, branch: string, metadata?: Record<string, unknown>): Promise<DegradedQueryResult<EnqueueResult | null>>;
/**
 * Get queue statistics grouped by status, item type, and collection.
 * Read-only — queries SQLite directly.
 */
export declare function getQueueStats(db: DatabaseType | null): DegradedQueryResult<QueueStats | null>;
//# sourceMappingURL=queue-operations.d.ts.map