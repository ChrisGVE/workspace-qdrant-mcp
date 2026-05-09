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
  EnqueueItemRequest,
  EnqueueItemResponse,
  LogSearchEventRequest,
  UpdateSearchEventRequest,
  UpsertRuleMirrorRequest,
  DeleteRuleMirrorRequest,
  UpsertScratchpadMirrorRequest,
  DeleteScratchpadMirrorRequest,
} from '../grpc-types.js';

import { DaemonClientSystem } from './system-methods.js';
import { grpcUnary } from './connection.js';

export class DaemonClientService extends DaemonClientSystem {
  // ── DocumentService ──

  async ingestText(request: IngestTextRequest): Promise<IngestTextResponse> {
    return this.callWithRetry(() => grpcUnary(this.documentClient, 'ingestText', request));
  }

  // ── EmbeddingService ──

  async embedText(request: EmbedTextRequest): Promise<EmbedTextResponse> {
    return this.callWithRetry(() => grpcUnary(this.embeddingClient, 'embedText', request));
  }

  async generateSparseVector(request: SparseVectorRequest): Promise<SparseVectorResponse> {
    return this.callWithRetry(() =>
      grpcUnary(this.embeddingClient, 'generateSparseVector', request)
    );
  }

  // ── TextSearchService ──

  async textSearch(request: TextSearchRequest): Promise<TextSearchResponse> {
    return this.callWithRetry(() => grpcUnary(this.textSearchClient, 'search', request));
  }

  async textSearchCount(request: TextSearchRequest): Promise<TextSearchCountResponse> {
    return this.callWithRetry(() => grpcUnary(this.textSearchClient, 'countMatches', request));
  }

  // ── GraphService ──

  async queryRelated(request: QueryRelatedRequest): Promise<QueryRelatedResponse> {
    return this.callWithRetry(() => grpcUnary(this.graphClient, 'queryRelated', request));
  }

  // ── QueueWriteService ──

  async enqueueItem(request: EnqueueItemRequest): Promise<EnqueueItemResponse> {
    return this.callWithRetry(() => grpcUnary(this.queueWriteClient, 'enqueueItem', request));
  }

  // ── TrackingWriteService ──

  async logSearchEvent(request: LogSearchEventRequest): Promise<void> {
    return this.callWithRetry(() => grpcUnary(this.trackingWriteClient, 'logSearchEvent', request));
  }

  async updateSearchEvent(request: UpdateSearchEventRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'updateSearchEvent', request)
    );
  }

  async upsertRuleMirror(request: UpsertRuleMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'upsertRuleMirror', request)
    );
  }

  async deleteRuleMirror(request: DeleteRuleMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'deleteRuleMirror', request)
    );
  }

  async upsertScratchpadMirror(request: UpsertScratchpadMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'upsertScratchpadMirror', request)
    );
  }

  async deleteScratchpadMirror(request: DeleteScratchpadMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'deleteScratchpadMirror', request)
    );
  }
}
