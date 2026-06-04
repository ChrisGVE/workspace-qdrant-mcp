/**
 * Queue payload builder utilities and idempotency key generation.
 *
 * Free functions used by SqliteStateManager and other consumers.
 */
import { createHash } from 'node:crypto';
// Valid item types
export const VALID_ITEM_TYPES = [
    'text',
    'file',
    'url',
    'website',
    'doc',
    'folder',
    'tenant',
    'collection',
];
// Valid operations per item type
export const VALID_OPERATIONS = {
    text: ['add', 'update', 'delete', 'uplift'],
    file: ['add', 'update', 'delete', 'rename', 'uplift'],
    url: ['add', 'update', 'delete', 'uplift'],
    website: ['add', 'update', 'delete', 'scan', 'uplift'],
    doc: ['delete', 'uplift'],
    folder: ['delete', 'scan', 'rename'],
    tenant: ['add', 'update', 'delete', 'scan', 'rename', 'uplift'],
    collection: ['uplift', 'reset'],
};
/**
 * Generate idempotency key for queue deduplication
 *
 * Format matches Python and Rust implementations:
 * Input: {item_type}|{op}|{tenant_id}|{collection}|{payload_json}
 * Output: SHA256 hash truncated to 32 hex characters
 */
export function generateIdempotencyKey(itemType, op, tenantId, collection, payload) {
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
export function buildContentPayload(content, sourceType, mainTag, fullTag) {
    return {
        content,
        source_type: sourceType,
        main_tag: mainTag,
        full_tag: fullTag,
    };
}
/**
 * Build rules payload for queue
 */
export function buildRulesPayload(label, content, scope, projectId) {
    return {
        content,
        source_type: 'rule',
        label,
        scope,
        project_id: projectId,
    };
}
/**
 * Build library payload for queue
 */
export function buildLibraryPayload(libraryName, content, source, url) {
    return {
        library_name: libraryName,
        content,
        source,
        url,
    };
}
//# sourceMappingURL=queue-payload-builders.js.map