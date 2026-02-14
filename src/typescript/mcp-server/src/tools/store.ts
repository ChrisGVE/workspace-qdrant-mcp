/**
 * Store tool implementation for content storage to libraries collection
 *
 * Per WORKSPACE_QDRANT_MCP.md spec line 1718:
 * "store is for adding reference documentation to the libraries collection"
 *
 * Per spec v1.3 changelog:
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
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';

// Canonical collection name from native bridge (single source of truth)
import { COLLECTION_LIBRARIES, PRIORITY_HIGH } from '../common/native-bridge.js';
const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;

export type SourceType = 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';

export interface StoreOptions {
  content: string;
  libraryName?: string;         // Required unless forProject is true
  forProject?: boolean;         // When true, store to libraries scoped to current project
  projectId?: string;           // Project tenant_id (required when forProject is true)
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
  // No configuration needed
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
export class StoreTool {
  private readonly stateManager: SqliteStateManager;

  constructor(
    _config: StoreToolConfig,
    stateManager: SqliteStateManager,
    // ProjectDetector no longer needed - we only write to libraries
  ) {
    // NOTE: DaemonClient intentionally NOT accepted per ADR-002
    // All writes must go through unified_queue only
    this.stateManager = stateManager;
  }

  /**
   * Store content to libraries collection
   *
   * @param options Store options with libraryName (required)
   * @returns StoreResponse with queue_id (content is queued, not stored immediately)
   */
  async store(options: StoreOptions): Promise<StoreResponse> {
    const {
      content,
      libraryName,
      forProject = false,
      projectId,
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

    // Determine tenant_id and library_name based on mode
    let tenantId: string;
    let libraryLabel: string;

    if (forProject) {
      // Project-scoped library: tenant_id = projectId, library_name = explicit or generated
      if (!projectId?.trim()) {
        return {
          success: false,
          collection: LIBRARIES_COLLECTION,
          message: 'No active project detected. forProject requires an active project session.',
          fallback_mode: 'unified_queue',
        };
      }
      tenantId = projectId.trim();
      libraryLabel = libraryName?.trim() || 'project-refs';
    } else {
      // Standalone library: tenant_id = libraryName (required)
      if (!libraryName?.trim()) {
        return {
          success: false,
          collection: LIBRARIES_COLLECTION,
          message: 'libraryName is required - this tool stores to the libraries collection only. For project content, use file watching (daemon handles this automatically).',
          fallback_mode: 'unified_queue',
        };
      }
      tenantId = libraryName.trim();
      libraryLabel = tenantId;
    }

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
    libraryName: string;
    documentId: string;
    metadata: Record<string, string>;
    sourceType: SourceType;
  }): Promise<{ queueId: string }> {
    const payload: Record<string, unknown> = {
      content: params.content,
      document_id: params.documentId,
      source_type: params.sourceType,
      metadata: params.metadata,
      library_name: params.libraryName,
    };

    // Use state manager to enqueue to libraries collection
    const result = await this.stateManager.enqueueUnified(
      'library',  // item_type per spec line 1147
      'ingest',
      params.tenantId,
      LIBRARIES_COLLECTION,
      payload,
      PRIORITY_HIGH, // MCP-initiated content is high priority
      undefined, // No branch for library content
      { source: 'mcp_store_tool' }
    );

    if (result.status !== 'ok' || !result.data) {
      throw new Error(result.message ?? 'Failed to enqueue store operation');
    }

    return { queueId: result.data.queueId };
  }
}
