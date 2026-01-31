/**
 * Store tool implementation for content storage
 *
 * Provides content storage to collections with:
 * - Daemon-first approach with unified_queue fallback
 * - Support for projects and libraries collections
 * - Idempotency via SHA256 hash of content
 * - Multiple source types: user_input, web, file
 *
 * Uses unified_queue fallback when daemon unavailable (per ADR-002)
 */

import { createHash } from 'node:crypto';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';

// Canonical collection names per ADR-001
const PROJECTS_COLLECTION = 'projects';
const LIBRARIES_COLLECTION = 'libraries';

// Collection basenames for daemon ingestion
const PROJECTS_BASENAME = 'code';
const LIBRARIES_BASENAME = 'lib';

export type SourceType = 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';
export type CollectionType = 'projects' | 'libraries';

export interface StoreOptions {
  content: string;
  collection?: CollectionType;
  title?: string;
  url?: string;
  filePath?: string;
  sourceType?: SourceType;
  projectId?: string;
  libraryName?: string;
  branch?: string;
  fileType?: string;
  metadata?: Record<string, string>;
}

export interface StoreResponse {
  success: boolean;
  documentId?: string;
  collection?: string;
  message?: string;
  fallback_mode?: 'unified_queue';
  queue_id?: string;
}

export interface StoreToolConfig {
  defaultCollection?: CollectionType;
}

/**
 * Store tool for content storage to collections
 */
export class StoreTool {
  private readonly daemonClient: DaemonClient;
  private readonly stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;
  private readonly defaultCollection: CollectionType;

  constructor(
    config: StoreToolConfig,
    daemonClient: DaemonClient,
    stateManager: SqliteStateManager,
    projectDetector: ProjectDetector
  ) {
    this.daemonClient = daemonClient;
    this.stateManager = stateManager;
    this.projectDetector = projectDetector;
    this.defaultCollection = config.defaultCollection ?? 'projects';
  }

  /**
   * Store content to a collection
   */
  async store(options: StoreOptions): Promise<StoreResponse> {
    const {
      content,
      collection = this.defaultCollection,
      title,
      url,
      filePath,
      sourceType = 'user_input',
      projectId,
      libraryName,
      branch,
      fileType,
      metadata = {},
    } = options;

    // Validate content
    if (!content?.trim()) {
      return {
        success: false,
        message: 'Content is required for storing',
      };
    }

    // Validate library name for libraries collection
    if (collection === 'libraries' && !libraryName) {
      return {
        success: false,
        message: 'Library name is required when storing to libraries collection',
      };
    }

    // Resolve tenant ID based on collection type
    let tenantId: string;
    let collectionBasename: string;

    if (collection === 'libraries') {
      tenantId = libraryName!;
      collectionBasename = LIBRARIES_BASENAME;
    } else {
      // For projects, use provided projectId or resolve from cwd
      if (projectId) {
        tenantId = projectId;
      } else {
        const cwd = process.cwd();
        const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
        tenantId = projectInfo?.projectId ?? 'default';
      }
      collectionBasename = PROJECTS_BASENAME;
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
    if (branch) fullMetadata['branch'] = branch;
    if (fileType) fullMetadata['file_type'] = fileType;

    // Try daemon first
    try {
      const response = await this.daemonClient.ingestText({
        content,
        collection_basename: collectionBasename,
        tenant_id: tenantId,
        document_id: documentId,
        metadata: fullMetadata,
      });

      if (response.success) {
        return {
          success: true,
          documentId: response.document_id,
          collection: collection === 'libraries' ? LIBRARIES_COLLECTION : PROJECTS_COLLECTION,
          message: `Content stored successfully (${response.chunks_created} chunks)`,
        };
      }

      // Daemon returned failure - fall back to queue
    } catch {
      // Daemon unavailable - fall back to queue
    }

    // Fallback: queue the operation
    const queueResult = this.queueStoreOperation({
      content,
      collection,
      tenantId,
      documentId,
      metadata: fullMetadata,
      sourceType,
      branch: branch ?? 'main',
    });

    return {
      success: true,
      documentId,
      collection: collection === 'libraries' ? LIBRARIES_COLLECTION : PROJECTS_COLLECTION,
      message: 'Content queued for processing',
      fallback_mode: 'unified_queue',
      queue_id: queueResult.queueId,
    };
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
   */
  private queueStoreOperation(params: {
    content: string;
    collection: CollectionType;
    tenantId: string;
    documentId: string;
    metadata: Record<string, string>;
    sourceType: SourceType;
    branch: string;
  }): { queueId: string } {
    const payload: Record<string, unknown> = {
      content: params.content,
      document_id: params.documentId,
      source_type: params.sourceType,
      metadata: params.metadata,
    };

    const collectionName = params.collection === 'libraries'
      ? LIBRARIES_COLLECTION
      : PROJECTS_COLLECTION;

    // Use state manager to enqueue
    const result = this.stateManager.enqueueUnified(
      'content',
      'ingest',
      params.tenantId,
      collectionName,
      payload,
      5, // Normal priority for store operations
      params.branch,
      { source: 'mcp_store_tool' }
    );

    if (result.status !== 'ok' || !result.data) {
      throw new Error(result.message ?? 'Failed to enqueue store operation');
    }

    return { queueId: result.data.queueId };
  }
}
