/**
 * DaemonClientService — Document, Embedding, TextSearch, Graph,
 * QueueWrite, and TrackingWrite RPC methods.
 */
import { DaemonClientSystem } from './system-methods.js';
import { grpcUnaryWithTimeout } from './connection.js';
export class DaemonClientService extends DaemonClientSystem {
    // ── DocumentService ──
    async ingestText(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.documentClient, 'ingestText', request, this.getMethodTimeout('ingestText')));
    }
    // ── EmbeddingService ──
    async embedText(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.embeddingClient, 'embedText', request, this.getMethodTimeout('embedText')));
    }
    async generateSparseVector(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.embeddingClient, 'generateSparseVector', request, this.getMethodTimeout('generateSparseVector')));
    }
    // ── TextSearchService ──
    async textSearch(request) {
        // 'search' is the wire method name; getMethodTimeout applies the 2× ceiling.
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.textSearchClient, 'search', request, this.getMethodTimeout('search')));
    }
    async textSearchCount(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.textSearchClient, 'countMatches', request, this.getMethodTimeout('countMatches')));
    }
    // ── GraphService ──
    async queryRelated(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.graphClient, 'queryRelated', request, this.getMethodTimeout('queryRelated')));
    }
    // ── QueueWriteService ──
    async enqueueItem(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.queueWriteClient, 'enqueueItem', request, this.getMethodTimeout('enqueueItem')));
    }
    // ── TrackingWriteService ──
    async logSearchEvent(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.trackingWriteClient, 'logSearchEvent', request, this.getMethodTimeout('logSearchEvent')));
    }
    async updateSearchEvent(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.trackingWriteClient, 'updateSearchEvent', request, this.getMethodTimeout('updateSearchEvent')));
    }
    async upsertRuleMirror(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.trackingWriteClient, 'upsertRuleMirror', request, this.getMethodTimeout('upsertRuleMirror')));
    }
    async deleteRuleMirror(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.trackingWriteClient, 'deleteRuleMirror', request, this.getMethodTimeout('deleteRuleMirror')));
    }
    async upsertScratchpadMirror(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.trackingWriteClient, 'upsertScratchpadMirror', request, this.getMethodTimeout('upsertScratchpadMirror')));
    }
    async deleteScratchpadMirror(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.trackingWriteClient, 'deleteScratchpadMirror', request, this.getMethodTimeout('deleteScratchpadMirror')));
    }
}
//# sourceMappingURL=service-methods.js.map