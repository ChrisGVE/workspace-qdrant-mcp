/** gRPC message types for SystemService and CollectionService. */
import type { ServiceStatus, QueueType, ServerState } from './grpc-types-messages-enums.js';
export interface ComponentHealth {
    component_name: string;
    status: ServiceStatus;
    message: string;
    last_check?: {
        seconds: number;
        nanos: number;
    };
}
export interface HealthCheckResponse {
    status: ServiceStatus;
    components: ComponentHealth[];
    timestamp?: {
        seconds: number;
        nanos: number;
    };
}
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
    uptime_since?: {
        seconds: number;
        nanos: number;
    };
}
export interface Metric {
    name: string;
    type: string;
    labels: Record<string, string>;
    value: number;
    timestamp?: {
        seconds: number;
        nanos: number;
    };
}
export interface MetricsResponse {
    metrics: Metric[];
    collected_at?: {
        seconds: number;
        nanos: number;
    };
}
export interface GetEmbeddingProviderStatusResponse {
    provider: string;
    model: string;
    output_dim: number;
    base_url: string;
    probe_status: string;
    probe_message: string;
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
//# sourceMappingURL=grpc-types-messages-system-collection.d.ts.map