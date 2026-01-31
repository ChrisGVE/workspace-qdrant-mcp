/**
 * SQLite state database types
 * Matches the schema owned by the Rust daemon (ADR-003)
 */

// ============================================================================
// Unified Queue Types
// ============================================================================

export type QueueItemType = 'content' | 'file' | 'folder' | 'project' | 'library' | 'memory';
export type QueueOperation = 'ingest' | 'update' | 'delete' | 'scan';
export type QueueStatus = 'pending' | 'in_progress' | 'done' | 'failed';

export interface UnifiedQueueItem {
  queue_id: string;
  idempotency_key: string;
  item_type: QueueItemType;
  op: QueueOperation;
  tenant_id: string;
  collection: string;
  priority: number;
  status: QueueStatus;
  branch?: string;
  payload_json: string;
  metadata?: string;
  created_at: string;
  updated_at: string;
  retry_count: number;
  max_retries: number;
  last_error?: string;
  leased_by?: string;
  lease_expires_at?: string;
}

export interface QueueStats {
  total_pending: number;
  total_in_progress: number;
  total_done: number;
  total_failed: number;
  by_item_type: Record<QueueItemType, number>;
  by_collection: Array<{ collection: string; count: number }>;
  stale_items_count: number;
}

// ============================================================================
// Watch Folder Types
// ============================================================================

export interface WatchFolderConfig {
  watch_id: string;
  path: string;
  collection: string;
  tenant_id: string;
  library_mode?: 'sync' | 'incremental';
  follow_symlinks: boolean;
  auto_ingest: boolean;
  enabled: boolean;
  cleanup_on_disable: boolean;
  created_at: string;
  updated_at: string;
  last_scan?: string;
}

// ============================================================================
// Registered Projects Types
// ============================================================================

export interface RegisteredProject {
  project_id: string;
  project_path: string;
  git_remote_url?: string;
  remote_hash?: string;
  disambiguation_path?: string;
  container_folder: string;
  is_active: boolean;
  created_at: string;
  last_seen_at?: string;
  last_activity_at?: string;
}

// ============================================================================
// Queue Payload Types
// ============================================================================

export interface ContentPayload {
  content: string;
  source_type: string;
  main_tag?: string;
  full_tag?: string;
}

export interface MemoryPayload {
  label: string;
  content: string;
  scope: 'global' | 'project';
  project_id?: string;
}

export interface FilePayload {
  file_path: string;
  relative_path?: string;
}

export interface FolderPayload {
  folder_path: string;
  relative_path?: string;
}

export interface LibraryPayload {
  content?: string;
  source?: string;
  url?: string;
  file_path?: string;
  library_name: string;
  relative_path?: string;
}

// ============================================================================
// Degraded Response Types
// ============================================================================

export interface DegradedResponse {
  results: never[];
  status: 'degraded';
  reason: 'database_not_initialized';
  message: string;
}

export function createDegradedResponse(message?: string): DegradedResponse {
  return {
    results: [],
    status: 'degraded',
    reason: 'database_not_initialized',
    message: message ?? 'Daemon has not run yet. Results may be incomplete.',
  };
}
