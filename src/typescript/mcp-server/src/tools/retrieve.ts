/**
 * Retrieve tool implementation for direct document access
 *
 * Provides direct document retrieval from collections with:
 * - Retrieval by document_id using Qdrant retrieve
 * - Retrieval by metadata filter using Qdrant scroll
 * - Pagination with limit and offset parameters
 *
 * Uses Qdrant client directly (read-only operation)
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import type { ProjectDetector } from '../utils/project-detector.js';

// Canonical collection names from native bridge (single source of truth)
import { COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_MEMORY } from '../common/native-bridge.js';
const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
const MEMORY_COLLECTION = COLLECTION_MEMORY;

export type RetrieveCollectionType = 'projects' | 'libraries' | 'memory';

export interface RetrieveOptions {
  documentId?: string;
  collection?: RetrieveCollectionType;
  filter?: Record<string, string>;
  limit?: number;
  offset?: number;
  projectId?: string;
  libraryName?: string;
}

export interface RetrievedDocument {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  score?: number;
}

export interface RetrieveResponse {
  success: boolean;
  documents: RetrievedDocument[];
  total?: number;
  hasMore?: boolean;
  message?: string;
}

export interface RetrieveToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
}

/**
 * Retrieve tool for direct document access
 */
export class RetrieveTool {
  private readonly qdrantClient: QdrantClient;
  private readonly projectDetector: ProjectDetector;

  constructor(config: RetrieveToolConfig, projectDetector: ProjectDetector) {
    const clientConfig: { url: string; apiKey?: string; timeout?: number } = {
      url: config.qdrantUrl,
      timeout: config.qdrantTimeout ?? 5000,
    };
    if (config.qdrantApiKey) {
      clientConfig.apiKey = config.qdrantApiKey;
    }
    this.qdrantClient = new QdrantClient(clientConfig);
    this.projectDetector = projectDetector;
  }

  /**
   * Retrieve documents from a collection
   */
  async retrieve(options: RetrieveOptions): Promise<RetrieveResponse> {
    const {
      documentId,
      collection = 'projects',
      filter,
      limit = 10,
      offset = 0,
      projectId,
      libraryName,
    } = options;

    const collectionName = this.getCollectionName(collection);

    // If documentId provided, retrieve by ID
    if (documentId) {
      return this.retrieveById(collectionName, documentId);
    }

    // Otherwise, retrieve by filter using scroll
    // Build params object conditionally to satisfy exactOptionalPropertyTypes
    const filterParams: {
      collectionName: string;
      collection: RetrieveCollectionType;
      filter?: Record<string, string>;
      limit: number;
      offset: number;
      projectId?: string;
      libraryName?: string;
    } = {
      collectionName,
      collection,
      limit,
      offset,
    };
    if (filter) filterParams.filter = filter;
    if (projectId) filterParams.projectId = projectId;
    if (libraryName) filterParams.libraryName = libraryName;

    return this.retrieveByFilter(filterParams);
  }

  /**
   * Retrieve a single document by ID
   */
  private async retrieveById(
    collectionName: string,
    documentId: string
  ): Promise<RetrieveResponse> {
    try {
      const result = await this.qdrantClient.retrieve(collectionName, {
        ids: [documentId],
        with_payload: true,
        with_vector: false,
      });

      if (result.length === 0) {
        return {
          success: false,
          documents: [],
          message: `Document not found: ${documentId}`,
        };
      }

      const point = result[0];
      // point is guaranteed to exist after the length check above
      if (!point) {
        return {
          success: false,
          documents: [],
          message: `Document not found: ${documentId}`,
        };
      }
      const document: RetrievedDocument = {
        id: String(point.id),
        content: (point.payload?.['content'] as string) ?? '',
        metadata: this.extractMetadata(point.payload),
      };

      return {
        success: true,
        documents: [document],
        total: 1,
        hasMore: false,
      };
    } catch (error) {
      return {
        success: false,
        documents: [],
        message: `Failed to retrieve document: ${error instanceof Error ? error.message : 'unknown error'}`,
      };
    }
  }

  /**
   * Retrieve documents by metadata filter using scroll
   */
  private async retrieveByFilter(params: {
    collectionName: string;
    collection: RetrieveCollectionType;
    filter?: Record<string, string>;
    limit: number;
    offset: number;
    projectId?: string;
    libraryName?: string;
  }): Promise<RetrieveResponse> {
    const { collectionName, collection, filter, limit, offset, projectId, libraryName } = params;

    try {
      // Build filter based on collection type and provided filters
      const qdrantFilter = await this.buildFilter(collection, filter, projectId, libraryName);

      // Build scroll request
      const scrollRequest: {
        limit: number;
        offset?: number;
        with_payload: boolean;
        with_vector: boolean;
        filter?: Record<string, unknown>;
      } = {
        limit: limit + 1, // Request one extra to determine hasMore
        with_payload: true,
        with_vector: false,
      };

      if (offset > 0) {
        scrollRequest.offset = offset;
      }

      if (qdrantFilter) {
        scrollRequest.filter = qdrantFilter;
      }

      const result = await this.qdrantClient.scroll(collectionName, scrollRequest);

      // Determine if there are more results
      const hasMore = result.points.length > limit;
      const points = hasMore ? result.points.slice(0, limit) : result.points;

      const documents: RetrievedDocument[] = points.map((point) => ({
        id: String(point.id),
        content: (point.payload?.['content'] as string) ?? '',
        metadata: this.extractMetadata(point.payload),
      }));

      return {
        success: true,
        documents,
        total: documents.length,
        hasMore,
      };
    } catch (error) {
      // Handle collection not found or other errors gracefully
      const errorMessage = error instanceof Error ? error.message : 'unknown error';
      if (errorMessage.includes('not found') || errorMessage.includes('doesn\'t exist')) {
        return {
          success: true,
          documents: [],
          total: 0,
          hasMore: false,
          message: 'Collection not found or empty',
        };
      }

      return {
        success: false,
        documents: [],
        message: `Failed to retrieve documents: ${errorMessage}`,
      };
    }
  }

  /**
   * Build Qdrant filter from options
   */
  private async buildFilter(
    collection: RetrieveCollectionType,
    filter?: Record<string, string>,
    projectId?: string,
    libraryName?: string
  ): Promise<Record<string, unknown> | null> {
    const mustConditions: Record<string, unknown>[] = [];

    // Add tenant filter based on collection type
    if (collection === 'projects') {
      const tenantId = projectId ?? (await this.resolveProjectId());
      if (tenantId) {
        mustConditions.push({
          key: 'tenant_id',
          match: { value: tenantId },
        });
      }
    } else if (collection === 'libraries' && libraryName) {
      mustConditions.push({
        key: 'tenant_id',
        match: { value: libraryName },
      });
    }

    // Add custom filter conditions
    if (filter) {
      for (const [key, value] of Object.entries(filter)) {
        mustConditions.push({
          key,
          match: { value },
        });
      }
    }

    if (mustConditions.length === 0) {
      return null;
    }

    return { must: mustConditions };
  }

  /**
   * Resolve project ID from current working directory
   */
  private async resolveProjectId(): Promise<string | undefined> {
    const cwd = process.cwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
    return projectInfo?.projectId;
  }

  /**
   * Get canonical collection name
   */
  private getCollectionName(collection: RetrieveCollectionType): string {
    switch (collection) {
      case 'projects':
        return PROJECTS_COLLECTION;
      case 'libraries':
        return LIBRARIES_COLLECTION;
      case 'memory':
        return MEMORY_COLLECTION;
      default:
        return PROJECTS_COLLECTION;
    }
  }

  /**
   * Extract metadata from payload (excluding content and vectors)
   */
  private extractMetadata(
    payload: Record<string, unknown> | null | undefined
  ): Record<string, unknown> {
    if (!payload) {
      return {};
    }

    const metadata: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(payload)) {
      // Skip content and vector fields
      if (key === 'content' || key === 'dense_vector' || key === 'sparse_vector') {
        continue;
      }
      metadata[key] = value;
    }
    return metadata;
  }
}
