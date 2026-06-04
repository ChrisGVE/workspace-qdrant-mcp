/**
 * gRPC client for communicating with the Rust daemon (memexd).
 *
 * Provides type-safe wrappers around daemon RPC methods with automatic
 * retry (exponential backoff), connection health monitoring, and timeouts.
 *
 * Implementation is split across:
 *   daemon-client/connection.ts     — base class, lifecycle, retry logic
 *   daemon-client/system-methods.ts — SystemService + ProjectService RPCs
 *   daemon-client/service-methods.ts — remaining service RPCs
 */
export type { DaemonClientConfig, ConnectionState } from './daemon-client/connection.js';
export { DaemonClientService as DaemonClient } from './daemon-client/service-methods.js';
export { ServiceStatus } from './grpc-types.js';
export type { HealthCheckResponse, SystemStatusResponse, MetricsResponse, GetEmbeddingProviderStatusResponse, RegisterProjectRequest, RegisterProjectResponse, DeprioritizeProjectRequest, DeprioritizeProjectResponse, HeartbeatRequest, HeartbeatResponse, IngestTextRequest, IngestTextResponse, EmbedTextRequest, EmbedTextResponse, SparseVectorRequest, SparseVectorResponse, TextSearchRequest, TextSearchResponse, TextSearchCountResponse, TextSearchMatch, QueryRelatedRequest, QueryRelatedResponse, TraversalNodeProto, EnqueueItemRequest, EnqueueItemResponse, LogSearchEventRequest, UpdateSearchEventRequest, UpsertRuleMirrorRequest, DeleteRuleMirrorRequest, UpsertScratchpadMirrorRequest, DeleteScratchpadMirrorRequest, } from './grpc-types.js';
//# sourceMappingURL=daemon-client.d.ts.map