/**
 * gRPC service client interfaces — typed wrappers for proto services.
 */

import type {
  HealthCheckResponse,
  SystemStatusResponse,
  MetricsResponse,
  RefreshSignalRequest,
  ServerStatusNotification,
  CreateCollectionRequest,
  CreateCollectionResponse,
  DeleteCollectionRequest,
  CreateAliasRequest,
  DeleteAliasRequest,
  RenameAliasRequest,
  IngestTextRequest,
  IngestTextResponse,
  UpdateTextRequest,
  UpdateTextResponse,
  DeleteTextRequest,
  RegisterProjectRequest,
  RegisterProjectResponse,
  DeprioritizeProjectRequest,
  DeprioritizeProjectResponse,
  GetProjectStatusRequest,
  GetProjectStatusResponse,
  ListProjectsRequest,
  ListProjectsResponse,
  HeartbeatRequest,
  HeartbeatResponse,
  EmbedTextRequest,
  EmbedTextResponse,
  SparseVectorRequest,
  SparseVectorResponse,
  TextSearchRequest,
  TextSearchResponse,
  TextSearchCountResponse,
} from './grpc-types-messages.js';

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

export interface TextSearchServiceClient {
  search(
    request: TextSearchRequest,
    callback: (error: Error | null, response: TextSearchResponse) => void
  ): void;
  countMatches(
    request: TextSearchRequest,
    callback: (error: Error | null, response: TextSearchCountResponse) => void
  ): void;
}
