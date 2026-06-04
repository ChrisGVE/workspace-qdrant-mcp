/**
 * DaemonClientSystem — SystemService and ProjectService RPC methods.
 */
import type { HealthCheckResponse, SystemStatusResponse, MetricsResponse, GetEmbeddingProviderStatusResponse, ServerState, RegisterProjectRequest, RegisterProjectResponse, DeprioritizeProjectRequest, DeprioritizeProjectResponse, HeartbeatRequest, HeartbeatResponse, ResolveSearchScopeRequest, ResolveSearchScopeResponse } from '../grpc-types.js';
import { DaemonClientBase } from './connection.js';
export declare class DaemonClientSystem extends DaemonClientBase {
    healthCheck(): Promise<HealthCheckResponse>;
    getStatus(): Promise<SystemStatusResponse>;
    getMetrics(): Promise<MetricsResponse>;
    getEmbeddingProviderStatus(): Promise<GetEmbeddingProviderStatusResponse>;
    notifyServerStatus(state: ServerState, projectName?: string, projectRoot?: string): Promise<void>;
    registerProject(request: RegisterProjectRequest): Promise<RegisterProjectResponse>;
    deprioritizeProject(request: DeprioritizeProjectRequest): Promise<DeprioritizeProjectResponse>;
    heartbeat(request: HeartbeatRequest): Promise<HeartbeatResponse>;
    resolveSearchScope(request: ResolveSearchScopeRequest): Promise<ResolveSearchScopeResponse>;
}
//# sourceMappingURL=system-methods.d.ts.map