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
import { createHash } from 'node:crypto';
// Canonical collection name from native bridge (single source of truth)
import { COLLECTION_LIBRARIES, PRIORITY_HIGH, FIELD_SOURCE_TYPE, FIELD_CONTENT, FIELD_DOCUMENT_ID, FIELD_LIBRARY_NAME, FIELD_TITLE, FIELD_FILE_PATH, } from '../common/native-bridge.js';
const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
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
export class StoreTool {
    stateManager;
    constructor(_config, stateManager
    // ProjectDetector no longer needed - we only write to libraries
    ) {
        // NOTE: DaemonClient intentionally NOT accepted per ADR-002
        // All writes must go through unified_queue only
        this.stateManager = stateManager;
    }
    async queueAndRespond(content, tenantId, libraryLabel, documentId, fullMetadata, sourceType) {
        try {
            const queueResult = await this.queueStoreOperation({
                content,
                tenantId,
                libraryName: libraryLabel,
                documentId,
                metadata: fullMetadata,
                sourceType,
            });
            return {
                success: true,
                documentId,
                collection: LIBRARIES_COLLECTION,
                message: `Content queued for processing by daemon (libraries/${tenantId})`,
                fallback_mode: 'unified_queue',
                queue_id: queueResult.queueId,
            };
        }
        catch (error) {
            return {
                success: false,
                collection: LIBRARIES_COLLECTION,
                message: `Failed to queue content: ${error instanceof Error ? error.message : 'Unknown error'}`,
                fallback_mode: 'unified_queue',
            };
        }
    }
    async store(options) {
        const { content, libraryName, forProject = false, projectId, title, url, filePath, sourceType = 'user_input', metadata = {}, } = options;
        if (!content?.trim()) {
            return {
                success: false,
                collection: LIBRARIES_COLLECTION,
                message: 'Content is required for storing',
                fallback_mode: 'unified_queue',
            };
        }
        const tenantResult = this.resolveTenant(forProject, projectId, libraryName);
        if ('error' in tenantResult)
            return tenantResult.error;
        const { tenantId, libraryLabel } = tenantResult;
        const documentId = this.generateDocumentId(content, tenantId);
        const fullMetadata = this.buildStoreMetadata(metadata, sourceType, title, url, filePath);
        return this.queueAndRespond(content, tenantId, libraryLabel, documentId, fullMetadata, sourceType);
    }
    resolveTenant(forProject, projectId, libraryName) {
        if (forProject) {
            if (!projectId?.trim()) {
                return {
                    error: {
                        success: false,
                        collection: LIBRARIES_COLLECTION,
                        message: 'No active project detected. forProject requires an active project session.',
                        fallback_mode: 'unified_queue',
                    },
                };
            }
            return { tenantId: projectId.trim(), libraryLabel: libraryName?.trim() || 'project-refs' };
        }
        if (!libraryName?.trim()) {
            return {
                error: {
                    success: false,
                    collection: LIBRARIES_COLLECTION,
                    message: 'libraryName is required - this tool stores to the libraries collection only. For project content, use file watching (daemon handles this automatically).',
                    fallback_mode: 'unified_queue',
                },
            };
        }
        return { tenantId: libraryName.trim(), libraryLabel: libraryName.trim() };
    }
    buildStoreMetadata(metadata, sourceType, title, url, filePath) {
        const full = { ...metadata, [FIELD_SOURCE_TYPE]: sourceType };
        if (title)
            full[FIELD_TITLE] = title;
        if (url)
            full['url'] = url;
        if (filePath)
            full[FIELD_FILE_PATH] = filePath;
        return full;
    }
    /**
     * Generate document ID using SHA256 hash for idempotency
     */
    generateDocumentId(content, tenantId) {
        const hash = createHash('sha256');
        hash.update(tenantId);
        hash.update(content);
        // Return first 32 chars of hash as document ID
        return hash.digest('hex').substring(0, 32);
    }
    /**
     * Queue store operation for daemon processing
     *
     * Per ADR-002: This is the ONLY write path for MCP store tool
     */
    async queueStoreOperation(params) {
        const payload = {
            [FIELD_CONTENT]: params.content,
            [FIELD_DOCUMENT_ID]: params.documentId,
            [FIELD_SOURCE_TYPE]: params.sourceType,
            metadata: params.metadata,
            [FIELD_LIBRARY_NAME]: params.libraryName,
        };
        // Use state manager to enqueue to libraries collection
        const result = await this.stateManager.enqueueUnified('tenant', 'add', params.tenantId, LIBRARIES_COLLECTION, payload, PRIORITY_HIGH, // MCP-initiated content is high priority
        undefined, // No branch for library content
        { source: 'mcp_store_tool' });
        if (result.status !== 'ok' || !result.data) {
            throw new Error(result.message ?? 'Failed to enqueue store operation');
        }
        return { queueId: result.data.queueId };
    }
}
//# sourceMappingURL=store.js.map