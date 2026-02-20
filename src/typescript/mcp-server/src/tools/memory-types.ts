/**
 * Memory tool types, interfaces, and constants.
 */

// Canonical memory collection name from native bridge (single source of truth)
import { COLLECTION_MEMORY } from '../common/native-bridge.js';
export const MEMORY_COLLECTION = COLLECTION_MEMORY;
export const MEMORY_BASENAME = 'memory';

export type MemoryAction = 'add' | 'update' | 'remove' | 'list';
export type MemoryScope = 'global' | 'project';

export interface MemoryRule {
  id: string;
  label?: string;
  content: string;
  scope: MemoryScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface MemoryOptions {
  action: MemoryAction;
  content?: string;
  label?: string;
  scope?: MemoryScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  limit?: number;
}

export interface MemoryResponse {
  success: boolean;
  action: MemoryAction;
  label?: string;
  rules?: MemoryRule[];
  message?: string;
  fallback_mode?: 'unified_queue';
  queue_id?: string;
}

export interface MemoryToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
}
