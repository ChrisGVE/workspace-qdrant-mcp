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
  UpdateSearchEventEconomyRequest,
  UpsertRuleMirrorRequest,
  DeleteRuleMirrorRequest,
  UpsertScratchpadMirrorRequest,
  DeleteScratchpadMirrorRequest,
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
}
