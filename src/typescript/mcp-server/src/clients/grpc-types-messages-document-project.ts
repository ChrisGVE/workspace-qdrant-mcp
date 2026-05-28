/** gRPC message types for DocumentService and ProjectService. */

// ── DocumentService ──

export interface IngestTextRequest {
  content: string;
  collection_basename: string;
  tenant_id: string;
  document_id?: string;
  metadata?: Record<string, string>;
  chunk_text?: boolean;
}

export interface IngestTextResponse {
  document_id: string;
  success: boolean;
  chunks_created: number;
  error_message?: string;
}

export interface UpdateTextRequest {
  document_id: string;
  content: string;
  collection_name?: string;
  metadata?: Record<string, string>;
}

export interface UpdateTextResponse {
  success: boolean;
  error_message?: string;
  updated_at?: { seconds: number; nanos: number };
}

export interface DeleteTextRequest {
  document_id: string;
  collection_name: string;
}

// ── ProjectService ──

export interface RegisterProjectRequest {
  path: string;
  project_id: string;
  name?: string;
  git_remote?: string;
  register_if_new?: boolean;
  priority?: string;
}

export interface RegisterProjectResponse {
  created: boolean;
  project_id: string;
  priority: string;
  is_active: boolean;
  newly_registered: boolean;
  is_worktree?: boolean;
  watch_path?: string;
}

export interface DeprioritizeProjectRequest {
  project_id: string;
  watch_path?: string;
}

export interface DeprioritizeProjectResponse {
  success: boolean;
  is_active: boolean;
  new_priority: string;
}

export interface GetProjectStatusRequest {
  project_id: string;
}

export interface GetProjectStatusResponse {
  found: boolean;
  project_id: string;
  project_name: string;
  project_root: string;
  priority: string;
  is_active: boolean;
  last_active?: { seconds: number; nanos: number };
  registered_at?: { seconds: number; nanos: number };
  git_remote?: string;
  is_worktree?: boolean;
  main_worktree_path?: string;
  // Indexing-progress block (filled by daemon's project_service).
  pending_count?: number;
  in_progress_count?: number;
  failed_count?: number;
  done_count?: number;
  total_count?: number;
  percent_complete?: number;
  // Optional ETA in seconds. Absent when the daemon doesn't have enough
  // recent activity data (cold-start) or when the rate is zero with
  // pending > 0 — callers must render "warming up" / "unknown" instead
  // of fabricating a value.
  eta_seconds?: number;
}

export interface ListProjectsRequest {
  priority_filter?: string;
  active_only?: boolean;
}

export interface ProjectInfo {
  project_id: string;
  project_name: string;
  project_root: string;
  priority: string;
  is_active: boolean;
  last_active?: { seconds: number; nanos: number };
  is_worktree?: boolean;
}

export interface ListProjectsResponse {
  projects: ProjectInfo[];
  total_count: number;
}

export interface ListWatchesRequest {
  collection?: string;
  enabled_only?: boolean;
}

export interface WatchInfo {
  watch_id: string;
  path: string;
  collection: string;
  tenant_id: string;
  enabled: boolean;
  is_active: boolean;
  is_paused: boolean;
  is_archived: boolean;
  last_scan: string;
  last_activity_at: string;
  git_remote_url?: string;
  library_mode?: string;
}

export interface ListWatchesResponse {
  watches: WatchInfo[];
  total_count: number;
}

export interface HeartbeatRequest {
  project_id: string;
}

export interface HeartbeatResponse {
  acknowledged: boolean;
  next_heartbeat_by?: { seconds: number; nanos: number };
}
