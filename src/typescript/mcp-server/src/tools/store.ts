/**
 * Store tool implementation for content storage to projects or libraries collection
 *
 * Per ADR-002, this tool ONLY uses unified_queue for writes.
 * The daemon processes the queue and writes to Qdrant.
 * MCP tools MUST NOT write to Qdrant directly.
 *
 * Supports both collections:
 * - projects collection: code and documents from projects (uses projectId)
 * - libraries collection: reference documentation (uses libraryName)
 * - memory collection: via the memory tool (not this tool)
 */

import { createHash } from 'node:crypto';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';

// Canonical collection names per ADR-001
const PROJECTS_COLLECTION = 'projects';
const LIBRARIES_COLLECTION = 'libraries';

export type SourceType = 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';

// MCP store tool can write to projects or libraries
export type CollectionType = 'projects' | 'libraries';

export interface StoreOptions {
  content: string;
  collection?: CollectionType;  // Target collection (default: projects)
  projectId?: string;           // Required for projects collection
  libraryName?: string;         // Required for libraries collection
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
  // No configuration needed - project ID detected via projectDetector
}

/**
 * Store tool for content storage to projects or libraries collection
 *
 * Per ADR-002: All writes go through unified_queue, never direct to daemon.
 */
export class StoreTool {
  private readonly stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;

  constructor(
    _config: StoreToolConfig,
    stateManager: SqliteStateManager,
    projectDetector: ProjectDetector
  ) {
    // NOTE: DaemonClient intentionally NOT accepted per ADR-002
    // All writes must go through unified_queue only
    this.stateManager = stateManager;
    this.projectDetector = projectDetector;
  }

  /**
   * Store content to projects or libraries collection
   *
   * @param options Store options with projectId or libraryName
   * @returns StoreResponse with queue_id (content is queued, not stored immediately)
   */
  async store(options: StoreOptions): Promise<StoreResponse> {
    const {
      content,
      collection = 'projects',  // Default to projects collection
      projectId,
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
        collection,
        message: 'Content is required for storing',
        fallback_mode: 'unified_queue',
      };
    }

    // Determine target collection and tenant ID
    let targetCollection: string;
    let tenantId: string;

    if (collection === 'libraries') {
      // Libraries collection requires libraryName
      if (!libraryName?.trim()) {
        return {
          success: false,
          collection: LIBRARIES_COLLECTION,
          message: 'libraryName is required when storing to libraries collection',
          fallback_mode: 'unified_queue',
        };
      }
      targetCollection = LIBRARIES_COLLECTION;
      tenantId = libraryName.trim();
    } else {
      // Projects collection - use projectId or auto-detect from current project
      let effectiveProjectId = projectId?.trim();
      if (!effectiveProjectId) {
        // Try to get project ID from database first
        effectiveProjectId = await this.projectDetector.getCurrentProjectId() ?? undefined;

        // Fallback: use project root path as tenant_id (daemon will reconcile)
        if (!effectiveProjectId) {
          const projectRoot = this.projectDetector.findProjectRoot(process.cwd());
          if (projectRoot) {
            effectiveProjectId = projectRoot;
          }
        }
      }
      if (!effectiveProjectId) {
        return {
          success: false,
          collection: PROJECTS_COLLECTION,
          message: 'projectId is required when storing to projects collection (no project detected in current directory)',
          fallback_mode: 'unified_queue',
        };
      }
      targetCollection = PROJECTS_COLLECTION;
      tenantId = effectiveProjectId;
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
        collection: targetCollection,
        documentId,
        metadata: fullMetadata,
        sourceType,
      });

      return {
        success: true,
        documentId,
        collection: targetCollection,
        message: `Content queued for processing by daemon (${targetCollection})`,
        fallback_mode: 'unified_queue',
        queue_id: queueResult.queueId,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        collection: targetCollection,
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
    collection: string;
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
      params.collection,
      payload,
      8, // Priority 8 for MCP content (same as other MCP operations)
      undefined, // No branch for content
      { source: 'mcp_store_tool' }
    );

    if (result.status !== 'ok' || !result.data) {
      throw new Error(result.message ?? 'Failed to enqueue store operation');
    }

    return { queueId: result.data.queueId };
  }
}
