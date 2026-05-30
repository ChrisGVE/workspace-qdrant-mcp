/**
 * DaemonClientService — Document, Embedding, TextSearch, Graph,
 * QueueWrite, and TrackingWrite RPC methods.
 */

import type {
  IngestTextRequest,
  IngestTextResponse,
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
  UpsertScratchpadMirrorRequest,
  DeleteScratchpadMirrorRequest,
  WatchIdRequest,
  WatchMutationResponse,
  ReapplyIgnoreRulesResponse,
  ReembedTenantRequest,
  ReembedTenantResponse,
} from '../grpc-types.js';

import { DaemonClientSystem } from './system-methods.js';
import { grpcUnaryWithTimeout } from './connection.js';

export class DaemonClientService extends DaemonClientSystem {
  // ── DocumentService ──

  async ingestText(request: IngestTextRequest): Promise<IngestTextResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.documentClient,
        'ingestText',
        request,
        this.getMethodTimeout('ingestText')
      )
    );
  }

  // ── EmbeddingService ──

  async embedText(request: EmbedTextRequest): Promise<EmbedTextResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.embeddingClient,
        'embedText',
        request,
        this.getMethodTimeout('embedText')
      )
    );
  }

  async generateSparseVector(request: SparseVectorRequest): Promise<SparseVectorResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.embeddingClient,
        'generateSparseVector',
        request,
        this.getMethodTimeout('generateSparseVector')
      )
    );
  }

  // ── TextSearchService ──

  async textSearch(request: TextSearchRequest): Promise<TextSearchResponse> {
    // 'search' is the wire method name; getMethodTimeout applies the 2× ceiling.
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.textSearchClient,
        'search',
        request,
        this.getMethodTimeout('search')
      )
    );
  }

  async textSearchCount(request: TextSearchRequest): Promise<TextSearchCountResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.textSearchClient,
        'countMatches',
        request,
        this.getMethodTimeout('countMatches')
      )
    );
  }

  // ── GraphService ──

  async queryRelated(request: QueryRelatedRequest): Promise<QueryRelatedResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.graphClient,
        'queryRelated',
        request,
        this.getMethodTimeout('queryRelated')
      )
    );
  }

  async impactAnalysis(request: ImpactAnalysisRequest): Promise<ImpactAnalysisResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.graphClient,
        'impactAnalysis',
        request,
        this.getMethodTimeout('impactAnalysis')
      )
    );
  }

  async computePageRank(request: PageRankRequest): Promise<PageRankResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.graphClient,
        'computePageRank',
        request,
        this.getMethodTimeout('computePageRank')
      )
    );
  }

  async detectCommunities(request: CommunityRequest): Promise<CommunityResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.graphClient,
        'detectCommunities',
        request,
        this.getMethodTimeout('detectCommunities')
      )
    );
  }

  async getGraphStats(request: GraphStatsRequest): Promise<GraphStatsResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.graphClient,
        'getGraphStats',
        request,
        this.getMethodTimeout('getGraphStats')
      )
    );
  }

  async computeBetweenness(request: BetweennessRequest): Promise<BetweennessResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.graphClient,
        'computeBetweenness',
        request,
        this.getMethodTimeout('computeBetweenness')
      )
    );
  }

  // ── QueueWriteService ──

  async enqueueItem(request: EnqueueItemRequest): Promise<EnqueueItemResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.queueWriteClient,
        'enqueueItem',
        request,
        this.getMethodTimeout('enqueueItem')
      )
    );
  }

  /** Reset all failed queue items back to 'pending' (QueueWriteService.RetryAll). */
  async retryAll(): Promise<RetryAllResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.queueWriteClient,
        'retryAll',
        {},
        this.getMethodTimeout('retryAll')
      )
    );
  }

  /** Reset a single failed queue item to 'pending' by id/prefix (QueueWriteService.RetryItem). */
  async retryItem(request: RetryItemRequest): Promise<RetryItemResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.queueWriteClient,
        'retryItem',
        request,
        this.getMethodTimeout('retryItem')
      )
    );
  }

  // ── TrackingWriteService ──

  async logSearchEvent(request: LogSearchEventRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.trackingWriteClient,
        'logSearchEvent',
        request,
        this.getMethodTimeout('logSearchEvent')
      )
    );
  }

  async updateSearchEvent(request: UpdateSearchEventRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.trackingWriteClient,
        'updateSearchEvent',
        request,
        this.getMethodTimeout('updateSearchEvent')
      )
    );
  }

  async updateSearchEventEconomy(request: UpdateSearchEventEconomyRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.trackingWriteClient,
        'updateSearchEventEconomy',
        request,
        this.getMethodTimeout('updateSearchEventEconomy')
      )
    );
  }

  async upsertRuleMirror(request: UpsertRuleMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.trackingWriteClient,
        'upsertRuleMirror',
        request,
        this.getMethodTimeout('upsertRuleMirror')
      )
    );
  }

  async deleteRuleMirror(request: DeleteRuleMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.trackingWriteClient,
        'deleteRuleMirror',
        request,
        this.getMethodTimeout('deleteRuleMirror')
      )
    );
  }

  async upsertScratchpadMirror(request: UpsertScratchpadMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.trackingWriteClient,
        'upsertScratchpadMirror',
        request,
        this.getMethodTimeout('upsertScratchpadMirror')
      )
    );
  }

  async deleteScratchpadMirror(request: DeleteScratchpadMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.trackingWriteClient,
        'deleteScratchpadMirror',
        request,
        this.getMethodTimeout('deleteScratchpadMirror')
      )
    );
  }

  // ── WatchWriteService ──

  async pauseWatch(request: WatchIdRequest): Promise<WatchMutationResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.watchWriteClient,
        'pauseWatch',
        request,
        this.getMethodTimeout('pauseWatch')
      )
    );
  }

  async resumeWatch(request: WatchIdRequest): Promise<WatchMutationResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.watchWriteClient,
        'resumeWatch',
        request,
        this.getMethodTimeout('resumeWatch')
      )
    );
  }

  // ── AdminWriteService ──

  async reapplyIgnoreRules(): Promise<ReapplyIgnoreRulesResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.adminWriteClient,
        'reapplyIgnoreRules',
        {},
        this.getMethodTimeout('reapplyIgnoreRules')
      )
    );
  }

  async reembedTenant(request: ReembedTenantRequest): Promise<ReembedTenantResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.adminWriteClient,
        'reembedTenant',
        request,
        this.getMethodTimeout('reembedTenant')
      )
    );
  }
}
