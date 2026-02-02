/**
 * Store tool implementation for content storage to libraries collection
 *
 * Per ADR-002, this tool ONLY uses unified_queue for writes.
 * The daemon processes the queue and writes to Qdrant.
 * MCP tools MUST NOT write to Qdrant directly.
 *
 * Per spec, MCP can only store to 'libraries' collection:
 * - projects collection: populated by file watcher
 * - libraries collection: reference documentation via this tool
 * - memory collection: via the memory tool
 */

import { createHash } from 'node:crypto';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';

// Canonical collection name per ADR-001
const LIBRARIES_COLLECTION = 'libraries';

export type SourceType = 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';

// MCP store tool can only write to libraries
export type CollectionType = 'libraries';

export interface StoreOptions {
  content: string;
  libraryName: string;        // Required - target library
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
  fallback_mode: 'unified_queue';  // Always queue-based per ADR-002
  queue_id?: string;
}

export interface StoreToolConfig {
  // No configuration needed - always libraries collection
}

/**
 * Store tool for content storage to libraries collection
 *
 * Per ADR-002: All writes go through unified_queue, never direct to daemon.
 * Per spec: MCP can only store to libraries (projects via file watcher).
 */
export class StoreTool {
  private readonly stateManager: SqliteStateManager;

  constructor(
    _config: StoreToolConfig,
    stateManager: SqliteStateManager
  ) {
    // NOTE: DaemonClient intentionally NOT accepted per ADR-002
    // All writes must go through unified_queue only
    this.stateManager = stateManager;
  }

  /**
   * Store content to the libraries collection
   *
   * @param options Store options with required libraryName
   * @returns StoreResponse with queue_id (content is queued, not stored immediately)
   */
  async store(options: StoreOptions): Promise<StoreResponse> {
    const {
      content,
      libraryName,
      title,
      url,
      filePath,
      sourceType = 'user_input',
      metadata = {},
    } = options;

    // Validate content
    if (!content?.trim()) {
      return {
        success: false,
        collection: LIBRARIES_COLLECTION,
        message: 'Content is required for storing',
        fallback_mode: 'unified_queue',
      };
    }

    // Validate library name
    if (!libraryName?.trim()) {
      return {
        success: false,
        collection: LIBRARIES_COLLECTION,
        message: 'Library name is required when storing to libraries collection',
        fallback_mode: 'unified_queue',
      };
    }

    // Tenant ID is the library name for libraries collection
    const tenantId = libraryName.trim();

    // Generate document ID using content hash for idempotency
    const documentId = this.generateDocumentId(content, tenantId);

    // Build metadata
    const fullMetadata: Record<string, string> = {
      ...metadata,
      source_type: sourceType,
    };

    if (title) fullMetadata['title'] = title;
    if (url) fullMetadata['url'] = url;
    if (filePath) fullMetadata['file_path'] = filePath;

    // Per ADR-002: ONLY queue the operation, never call daemon directly
    try {
      const queueResult = await this.queueStoreOperation({
        content,
        tenantId,
        documentId,
        metadata: fullMetadata,
        sourceType,
      });

      return {
        success: true,
        documentId,
        collection: LIBRARIES_COLLECTION,
        message: 'Content queued for processing by daemon',
        fallback_mode: 'unified_queue',
        queue_id: queueResult.queueId,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        collection: LIBRARIES_COLLECTION,
        message: `Failed to queue content: ${errorMessage}`,
        fallback_mode: 'unified_queue',
      };
    }
  }

  /**
   * Generate document ID using SHA256 hash for idempotency
   */
  private generateDocumentId(content: string, tenantId: string): string {
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
  private async queueStoreOperation(params: {
    content: string;
    tenantId: string;
    documentId: string;
    metadata: Record<string, string>;
    sourceType: SourceType;
  }): Promise<{ queueId: string }> {
    const payload: Record<string, unknown> = {
      content: params.content,
      document_id: params.documentId,
      source_type: params.sourceType,
      metadata: params.metadata,
    };

    // Use state manager to enqueue
    const result = await this.stateManager.enqueueUnified(
      'content',
      'ingest',
      params.tenantId,
      LIBRARIES_COLLECTION,
      payload,
      8, // Priority 8 for MCP content (same as other MCP operations)
      undefined, // No branch for library content
      { source: 'mcp_store_tool' }
    );

    if (result.status !== 'ok' || !result.data) {
      throw new Error(result.message ?? 'Failed to enqueue store operation');
    }

    return { queueId: result.data.queueId };
  }
}
