/**
 * DaemonClientService — Document, Embedding, TextSearch, Graph,
 * QueueWrite, and TrackingWrite RPC methods.
 */
import type { IngestTextRequest, IngestTextResponse, EmbedTextRequest, EmbedTextResponse, SparseVectorRequest, SparseVectorResponse, TextSearchRequest, TextSearchResponse, TextSearchCountResponse, QueryRelatedRequest, QueryRelatedResponse, EnqueueItemRequest, EnqueueItemResponse, LogSearchEventRequest, UpdateSearchEventRequest, UpsertRuleMirrorRequest, DeleteRuleMirrorRequest, UpsertScratchpadMirrorRequest, DeleteScratchpadMirrorRequest } from '../grpc-types.js';
import { DaemonClientSystem } from './system-methods.js';
export declare class DaemonClientService extends DaemonClientSystem {
    ingestText(request: IngestTextRequest): Promise<IngestTextResponse>;
    embedText(request: EmbedTextRequest): Promise<EmbedTextResponse>;
    generateSparseVector(request: SparseVectorRequest): Promise<SparseVectorResponse>;
    textSearch(request: TextSearchRequest): Promise<TextSearchResponse>;
    textSearchCount(request: TextSearchRequest): Promise<TextSearchCountResponse>;
    queryRelated(request: QueryRelatedRequest): Promise<QueryRelatedResponse>;
    enqueueItem(request: EnqueueItemRequest): Promise<EnqueueItemResponse>;
    logSearchEvent(request: LogSearchEventRequest): Promise<void>;
    updateSearchEvent(request: UpdateSearchEventRequest): Promise<void>;
    upsertRuleMirror(request: UpsertRuleMirrorRequest): Promise<void>;
    deleteRuleMirror(request: DeleteRuleMirrorRequest): Promise<void>;
    upsertScratchpadMirror(request: UpsertScratchpadMirrorRequest): Promise<void>;
    deleteScratchpadMirror(request: DeleteScratchpadMirrorRequest): Promise<void>;
}
//# sourceMappingURL=service-methods.d.ts.map