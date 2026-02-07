/**
 * TypeScript types for gRPC daemon communication
 * Matches workspace_daemon.proto definitions
 */

// ============================================================================
// Enums
// ============================================================================

export enum ServiceStatus {
  SERVICE_STATUS_UNSPECIFIED = 0,
  SERVICE_STATUS_HEALTHY = 1,
  SERVICE_STATUS_DEGRADED = 2,
  SERVICE_STATUS_UNHEALTHY = 3,
  SERVICE_STATUS_UNAVAILABLE = 4,
}

export enum QueueType {
  QUEUE_TYPE_UNSPECIFIED = 0,
  INGEST_QUEUE = 1,
  WATCHED_PROJECTS = 2,
  WATCHED_FOLDERS = 3,
  TOOLS_AVAILABLE = 4,
}

export enum ServerState {
  SERVER_STATE_UNSPECIFIED = 0,
  SERVER_STATE_UP = 1,
  SERVER_STATE_DOWN = 2,
}

// ============================================================================
// SystemService Types
// ============================================================================

export interface ComponentHealth {
  component_name: string;
  status: ServiceStatus;
  message: string;
  last_check?: { seconds: number; nanos: number };
}

export interface HealthCheckResponse {
  status: ServiceStatus;
  components: ComponentHealth[];
  timestamp?: { seconds: number; nanos: number };
}

// Alias to match daemon's proto naming
export type HealthResponse = HealthCheckResponse;

export interface SystemMetrics {
  cpu_usage_percent: number;
  memory_usage_bytes: number;
  memory_total_bytes: number;
  disk_usage_bytes: number;
  disk_total_bytes: number;
  active_connections: number;
  pending_operations: number;
}

export interface SystemStatusResponse {
  status: ServiceStatus;
  metrics: SystemMetrics;
  active_projects: string[];
  total_documents: number;
  total_collections: number;
  uptime_since?: { seconds: number; nanos: number };
}

export interface Metric {
  name: string;
  type: string;
  labels: Record<string, string>;
  value: number;
  timestamp?: { seconds: number; nanos: number };
}

export interface MetricsResponse {
  metrics: Metric[];
  collected_at?: { seconds: number; nanos: number };
}

export interface RefreshSignalRequest {
  queue_type: QueueType;
  lsp_languages?: string[];
  grammar_languages?: string[];
}

export interface ServerStatusNotification {
  state: ServerState;
  project_name?: string;
  project_root?: string;
}

// ============================================================================
// CollectionService Types
// ============================================================================

export interface CollectionConfig {
  vector_size: number;
  distance_metric: string;
  enable_indexing: boolean;
  metadata_schema: Record<string, string>;
}

export interface CreateCollectionRequest {
  collection_name: string;
  project_id?: string;
  config?: CollectionConfig;
}

export interface CreateCollectionResponse {
  success: boolean;
  error_message?: string;
  collection_id?: string;
}

export interface DeleteCollectionRequest {
  collection_name: string;
  project_id?: string;
  force?: boolean;
}

export interface CreateAliasRequest {
  alias_name: string;
  collection_name: string;
}

export interface DeleteAliasRequest {
  alias_name: string;
}

export interface RenameAliasRequest {
  old_alias_name: string;
  new_alias_name: string;
  collection_name: string;
}

// ============================================================================
// DocumentService Types
// ============================================================================

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

// ============================================================================
// ProjectService Types
// ============================================================================

export interface RegisterProjectRequest {
  path: string;
  project_id: string;
  name?: string;
  git_remote?: string;
  register_if_new?: boolean;
}

export interface RegisterProjectResponse {
  created: boolean;
  project_id: string;
  priority: string;
  is_active: boolean;
  newly_registered: boolean;
}

export interface DeprioritizeProjectRequest {
  project_id: string;
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
}

export interface ListProjectsResponse {
  projects: ProjectInfo[];
  total_count: number;
}

export interface HeartbeatRequest {
  project_id: string;
}

export interface HeartbeatResponse {
  acknowledged: boolean;
  next_heartbeat_by?: { seconds: number; nanos: number };
}

// ============================================================================
// Service Client Interfaces
// ============================================================================

export interface SystemServiceClient {
  health(
    request: Record<string, never>,
    callback: (error: Error | null, response: HealthCheckResponse) => void
  ): void;
  getStatus(
    request: Record<string, never>,
    callback: (error: Error | null, response: SystemStatusResponse) => void
  ): void;
  getMetrics(
    request: Record<string, never>,
    callback: (error: Error | null, response: MetricsResponse) => void
  ): void;
  sendRefreshSignal(
    request: RefreshSignalRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  notifyServerStatus(
    request: ServerStatusNotification,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  pauseAllWatchers(
    request: Record<string, never>,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  resumeAllWatchers(
    request: Record<string, never>,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
}

export interface CollectionServiceClient {
  createCollection(
    request: CreateCollectionRequest,
    callback: (error: Error | null, response: CreateCollectionResponse) => void
  ): void;
  deleteCollection(
    request: DeleteCollectionRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  createCollectionAlias(
    request: CreateAliasRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  deleteCollectionAlias(
    request: DeleteAliasRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  renameCollectionAlias(
    request: RenameAliasRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
}

export interface DocumentServiceClient {
  ingestText(
    request: IngestTextRequest,
    callback: (error: Error | null, response: IngestTextResponse) => void
  ): void;
  updateText(
    request: UpdateTextRequest,
    callback: (error: Error | null, response: UpdateTextResponse) => void
  ): void;
  deleteText(
    request: DeleteTextRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
}

export interface ProjectServiceClient {
  registerProject(
    request: RegisterProjectRequest,
    callback: (error: Error | null, response: RegisterProjectResponse) => void
  ): void;
  deprioritizeProject(
    request: DeprioritizeProjectRequest,
    callback: (error: Error | null, response: DeprioritizeProjectResponse) => void
  ): void;
  getProjectStatus(
    request: GetProjectStatusRequest,
    callback: (error: Error | null, response: GetProjectStatusResponse) => void
  ): void;
  listProjects(
    request: ListProjectsRequest,
    callback: (error: Error | null, response: ListProjectsResponse) => void
  ): void;
  heartbeat(
    request: HeartbeatRequest,
    callback: (error: Error | null, response: HeartbeatResponse) => void
  ): void;
}

// ============================================================================
// EmbeddingService Types
// ============================================================================

export interface EmbedTextRequest {
  text: string;
  model?: string;
}

export interface EmbedTextResponse {
  embedding: number[];
  dimensions: number;
  model_name: string;
  success: boolean;
  error_message?: string;
}

export interface SparseVectorRequest {
  text: string;
}

export interface SparseVectorResponse {
  indices_values: Record<number, number>;
  vocab_size: number;
  success: boolean;
  error_message?: string;
}

export interface EmbeddingServiceClient {
  embedText(
    request: EmbedTextRequest,
    callback: (error: Error | null, response: EmbedTextResponse) => void
  ): void;
  generateSparseVector(
    request: SparseVectorRequest,
    callback: (error: Error | null, response: SparseVectorResponse) => void
  ): void;
}
