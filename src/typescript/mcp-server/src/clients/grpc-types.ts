/**
 * gRPC types barrel — re-exports from sub-modules.
 *
 * - grpc-types-messages.ts: Enums and data message interfaces
 * - grpc-types-clients.ts: Service client interfaces
 */

export {
  // Enums
  ServiceStatus,
  QueueType,
  ServerState,
  // SystemService
  type ComponentHealth,
  type HealthCheckResponse,
  type HealthResponse,
  type SystemMetrics,
  type SystemStatusResponse,
  type Metric,
  type MetricsResponse,
  type RefreshSignalRequest,
  type ServerStatusNotification,
  // CollectionService
  type CollectionConfig,
  type CreateCollectionRequest,
  type CreateCollectionResponse,
  type DeleteCollectionRequest,
  type CreateAliasRequest,
  type DeleteAliasRequest,
  type RenameAliasRequest,
  // DocumentService
  type IngestTextRequest,
  type IngestTextResponse,
  type UpdateTextRequest,
  type UpdateTextResponse,
  type DeleteTextRequest,
  // ProjectService
  type RegisterProjectRequest,
  type RegisterProjectResponse,
  type DeprioritizeProjectRequest,
  type DeprioritizeProjectResponse,
  type GetProjectStatusRequest,
  type GetProjectStatusResponse,
  type ListProjectsRequest,
  type ProjectInfo,
  type ListProjectsResponse,
  type HeartbeatRequest,
  type HeartbeatResponse,
  // EmbeddingService
  type EmbedTextRequest,
  type EmbedTextResponse,
  type SparseVectorRequest,
  type SparseVectorResponse,
  // TextSearchService
  type TextSearchRequest,
  type TextSearchResponse,
  type TextSearchCountResponse,
  type TextSearchMatch,
} from './grpc-types-messages.js';

export type {
  SystemServiceClient,
  CollectionServiceClient,
  DocumentServiceClient,
  ProjectServiceClient,
  EmbeddingServiceClient,
  TextSearchServiceClient,
} from './grpc-types-clients.js';
