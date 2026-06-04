/** gRPC message types for DocumentService and ProjectService. */
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
    updated_at?: {
        seconds: number;
        nanos: number;
    };
}
export interface DeleteTextRequest {
    document_id: string;
    collection_name: string;
}
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
    last_active?: {
        seconds: number;
        nanos: number;
    };
    registered_at?: {
        seconds: number;
        nanos: number;
    };
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
    last_active?: {
        seconds: number;
        nanos: number;
    };
    is_worktree?: boolean;
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
    next_heartbeat_by?: {
        seconds: number;
        nanos: number;
    };
}
export interface ResolveSearchScopeRequest {
    tenant_id: string;
    scope: string;
}
export interface TenantDecay {
    tenant_id: string;
    multiplier: number;
}
export interface ResolveSearchScopeResponse {
    tenant_ids: string[];
    filter_by_tenant: boolean;
    decay_map: TenantDecay[];
}
//# sourceMappingURL=grpc-types-messages-document-project.d.ts.map