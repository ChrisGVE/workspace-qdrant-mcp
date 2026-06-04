/**
 * Queue payload builder utilities and idempotency key generation.
 *
 * Free functions used by SqliteStateManager and other consumers.
 */
import type { QueueItemType, QueueOperation, ContentPayload, RulesPayload, LibraryPayload } from '../types/state.js';
export declare const VALID_ITEM_TYPES: QueueItemType[];
export declare const VALID_OPERATIONS: Record<QueueItemType, QueueOperation[]>;
/**
 * Generate idempotency key for queue deduplication
 *
 * Format matches Python and Rust implementations:
 * Input: {item_type}|{op}|{tenant_id}|{collection}|{payload_json}
 * Output: SHA256 hash truncated to 32 hex characters
 */
export declare function generateIdempotencyKey(itemType: QueueItemType, op: QueueOperation, tenantId: string, collection: string, payload: Record<string, unknown>): string;
/**
 * Build content payload for queue
 */
export declare function buildContentPayload(content: string, sourceType: string, mainTag?: string, fullTag?: string): ContentPayload;
/**
 * Build rules payload for queue
 */
export declare function buildRulesPayload(label: string, content: string, scope: 'global' | 'project', projectId?: string): RulesPayload;
/**
 * Build library payload for queue
 */
export declare function buildLibraryPayload(libraryName: string, content?: string, source?: string, url?: string): LibraryPayload;
//# sourceMappingURL=queue-payload-builders.d.ts.map