/**
 * Store tool implementation for content storage to libraries collection
 *
 * Per docs/specs/08-api-reference.md:
 * "store is for adding reference documentation to the libraries collection"
 *
 * Per spec:
 * "clarified MCP does NOT store to projects collection (daemon handles via file watching)"
 *
 * Per ADR-002, this tool ONLY uses unified_queue for writes.
 * The daemon processes the queue and writes to Qdrant.
 * MCP tools MUST NOT write to Qdrant directly.
 *
 * IMPORTANT: This tool is for LIBRARIES collection ONLY.
 * Project content is handled by daemon file watching, not this tool.
 */
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
export type SourceType = 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';
export interface StoreOptions {
    content: string;
    libraryName?: string;
    forProject?: boolean;
    projectId?: string;
    title?: string;
    url?: string;
    filePath?: string;
    sourceType?: SourceType;
    metadata?: Record<string, string>;
}
export interface StoreResponse {
    success: boolean;
    documentId?: string;
    collection: string;
    message: string;
    fallback_mode: 'unified_queue';
    queue_id?: string;
}
export interface StoreToolConfig {
}
/**
 * Store tool for content storage to libraries collection
 *
 * Per spec: "store is for adding reference documentation to the libraries collection"
 * Libraries are collections of reference information (books, documentation, papers, websites)
 * - NOT programming libraries (use context7 MCP for those)
 * - NOT project content (handled by daemon file watching)
 *
 * Per ADR-002: All writes go through unified_queue, never direct to daemon.
 */
export declare class StoreTool {
    private readonly stateManager;
    constructor(_config: StoreToolConfig, stateManager: SqliteStateManager);
    private queueAndRespond;
    store(options: StoreOptions): Promise<StoreResponse>;
    private resolveTenant;
    private buildStoreMetadata;
    /**
     * Generate document ID using SHA256 hash for idempotency
     */
    private generateDocumentId;
    /**
     * Queue store operation for daemon processing
     *
     * Per ADR-002: This is the ONLY write path for MCP store tool
     */
    private queueStoreOperation;
}
//# sourceMappingURL=store.d.ts.map