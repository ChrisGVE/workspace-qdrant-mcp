/**
 * Queue payload builder utilities and idempotency key generation.
 *
 * Free functions used by SqliteStateManager and other consumers.
 */

import { createHash } from 'node:crypto';

import type {
  QueueItemType,
  QueueOperation,
  ContentPayload,
  RulesPayload,
  LibraryPayload,
} from '../types/state.js';

// Valid item types
export const VALID_ITEM_TYPES: QueueItemType[] = [
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
export const VALID_OPERATIONS: Record<QueueItemType, QueueOperation[]> = {
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
 * Build rules payload for queue
 */
export function buildRulesPayload(
  label: string,
  content: string,
  scope: 'global' | 'project',
  projectId?: string
): RulesPayload {
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
