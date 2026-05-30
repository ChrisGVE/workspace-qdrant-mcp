/**
 * gRPC service client interfaces — typed wrappers for proto services.
 */

import type {
  HealthCheckResponse,
  SystemStatusResponse,
  MetricsResponse,
  QueueStatsResponse,
  GetEmbeddingProviderStatusResponse,
  RebuildIndexRequest,
  RebuildIndexResponse,
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
  ListWatchesRequest,
  ListWatchesResponse,
  HeartbeatRequest,
  HeartbeatResponse,
  ListFailedItemsRequest,
  ListFailedItemsResponse,
  EmbedTextRequest,
  EmbedTextResponse,
  SparseVectorRequest,
  SparseVectorResponse,
  TextSearchRequest,
  TextSearchResponse,
  TextSearchCountResponse,
  QueryRelatedRequest,
  QueryRelatedResponse,
  ImpactAnalysisRequest,
  ImpactAnalysisResponse,
  PageRankRequest,
  PageRankResponse,
  GraphStatsRequest,
  GraphStatsResponse,
  CommunityRequest,
  CommunityResponse,
  BetweennessRequest,
  BetweennessResponse,
  EnqueueItemRequest,
  EnqueueItemResponse,
  RetryAllResponse,
  RetryItemRequest,
  RetryItemResponse,
  LogSearchEventRequest,
  UpdateSearchEventRequest,
  UpdateSearchEventEconomyRequest,
  UpsertRuleMirrorRequest,
  DeleteRuleMirrorRequest,
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
  getEmbeddingProviderStatus(
    request: Record<string, never>,
    callback: (error: Error | null, response: GetEmbeddingProviderStatusResponse) => void
  ): void;
  getQueueStats(
    request: Record<string, never>,
    callback: (error: Error | null, response: QueueStatsResponse) => void
  ): void;
  rebuildIndex(
    request: RebuildIndexRequest,
    callback: (error: Error | null, response: RebuildIndexResponse) => void
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
  listWatches(
    request: ListWatchesRequest,
    callback: (error: Error | null, response: ListWatchesResponse) => void
  ): void;
  heartbeat(
    request: HeartbeatRequest,
    callback: (error: Error | null, response: HeartbeatResponse) => void
  ): void;
  listFailedItems(
    request: ListFailedItemsRequest,
    callback: (error: Error | null, response: ListFailedItemsResponse) => void
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

export interface GraphServiceClient {
  queryRelated(
    request: QueryRelatedRequest,
    callback: (error: Error | null, response: QueryRelatedResponse) => void
  ): void;
  impactAnalysis(
    request: ImpactAnalysisRequest,
    callback: (error: Error | null, response: ImpactAnalysisResponse) => void
  ): void;
  computePageRank(
    request: PageRankRequest,
    callback: (error: Error | null, response: PageRankResponse) => void
  ): void;
  detectCommunities(
    request: CommunityRequest,
    callback: (error: Error | null, response: CommunityResponse) => void
  ): void;
  getGraphStats(
    request: GraphStatsRequest,
    callback: (error: Error | null, response: GraphStatsResponse) => void
  ): void;
  computeBetweenness(
    request: BetweennessRequest,
    callback: (error: Error | null, response: BetweennessResponse) => void
  ): void;
}

export interface QueueWriteServiceClient {
  enqueueItem(
    request: EnqueueItemRequest,
    callback: (error: Error | null, response: EnqueueItemResponse) => void
  ): void;
  retryAll(
    request: Record<string, never>,
    callback: (error: Error | null, response: RetryAllResponse) => void
  ): void;
  retryItem(
    request: RetryItemRequest,
    callback: (error: Error | null, response: RetryItemResponse) => void
  ): void;
}

export interface TrackingWriteServiceClient {
  logSearchEvent(
    request: LogSearchEventRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  updateSearchEvent(
    request: UpdateSearchEventRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  updateSearchEventEconomy(
    request: UpdateSearchEventEconomyRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  upsertRuleMirror(
    request: UpsertRuleMirrorRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
  deleteRuleMirror(
    request: DeleteRuleMirrorRequest,
    callback: (error: Error | null, response: Record<string, never>) => void
  ): void;
}

export interface WatchIdRequest {
  watch_id: string;
}

export interface WatchMutationResponse {
  affected_count: number;
}

export interface WatchWriteServiceClient {
  pauseWatchers(
    request: Record<string, never>,
    callback: (error: Error | null, response: WatchMutationResponse) => void
  ): void;
  resumeWatchers(
    request: Record<string, never>,
    callback: (error: Error | null, response: WatchMutationResponse) => void
  ): void;
  pauseWatch(
    request: WatchIdRequest,
    callback: (error: Error | null, response: WatchMutationResponse) => void
  ): void;
  resumeWatch(
    request: WatchIdRequest,
    callback: (error: Error | null, response: WatchMutationResponse) => void
  ): void;
}

export interface ReapplyIgnoreRulesResponse {
  projects_processed: number;
  stale_deleted: number;
  missing_added: number;
}

export interface ReembedTenantRequest {
  tenant_id: string;
}

export interface ReembedTenantResponse {
  files_enqueued: number;
  message: string;
}

export interface AdminWriteServiceClient {
  reapplyIgnoreRules(
    request: Record<string, never>,
    callback: (error: Error | null, response: ReapplyIgnoreRulesResponse) => void
  ): void;
  reembedTenant(
    request: ReembedTenantRequest,
    callback: (error: Error | null, response: ReembedTenantResponse) => void
  ): void;
}
