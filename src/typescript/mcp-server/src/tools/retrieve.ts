/**
 * Retrieve tool — direct document access from Qdrant collections.
 *
 * - retrieve-types.ts: Types, constants, helpers
 * - retrieve.ts (this): RetrieveTool class with byId and byFilter operations
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import type { ProjectDetector } from '../utils/project-detector.js';

// Re-export all types so existing imports from './retrieve.js' continue to work
export type {
  RetrieveCollectionType,
  RetrieveOptions,
  RetrievedDocument,
  RetrieveResponse,
  RetrieveToolConfig,
} from './retrieve-types.js';

import type {
  RetrieveCollectionType,
  RetrieveOptions,
  RetrievedDocument,
  RetrieveResponse,
  RetrieveToolConfig,
} from './retrieve-types.js';
import { getCollectionName, extractMetadata } from './retrieve-types.js';

export class RetrieveTool {
  private readonly qdrantClient: QdrantClient;
  private readonly projectDetector: ProjectDetector;

  constructor(config: RetrieveToolConfig, projectDetector: ProjectDetector) {
    const clientConfig: { url: string; apiKey?: string; timeout?: number } = {
      url: config.qdrantUrl,
      timeout: config.qdrantTimeout ?? 5000,
    };
    if (config.qdrantApiKey) clientConfig.apiKey = config.qdrantApiKey;
    this.qdrantClient = new QdrantClient(clientConfig);
    this.projectDetector = projectDetector;
  }

  async retrieve(options: RetrieveOptions): Promise<RetrieveResponse> {
    const {
      documentId, collection = 'projects', filter,
      limit = 10, offset = 0, projectId, libraryName,
    } = options;

    const collectionName = getCollectionName(collection);

    if (documentId) return this.retrieveById(collectionName, documentId);

    const filterParams: {
      collectionName: string; collection: RetrieveCollectionType;
      filter?: Record<string, string>; limit: number; offset: number;
      projectId?: string; libraryName?: string;
    } = { collectionName, collection, limit, offset };
    if (filter) filterParams.filter = filter;
    if (projectId) filterParams.projectId = projectId;
    if (libraryName) filterParams.libraryName = libraryName;

    return this.retrieveByFilter(filterParams);
  }

  private async retrieveById(collectionName: string, documentId: string): Promise<RetrieveResponse> {
    try {
      const result = await this.qdrantClient.retrieve(collectionName, {
        ids: [documentId], with_payload: true, with_vector: false,
      });

      const point = result[0];
      if (!point) {
        return { success: false, documents: [], message: `Document not found: ${documentId}` };
      }

      const document: RetrievedDocument = {
        id: String(point.id),
        content: (point.payload?.['content'] as string) ?? '',
        metadata: extractMetadata(point.payload),
      };

      return { success: true, documents: [document], total: 1, hasMore: false };
    } catch (error) {
      return {
        success: false, documents: [],
        message: `Failed to retrieve document: ${error instanceof Error ? error.message : 'unknown error'}`,
      };
    }
  }

  private async retrieveByFilter(params: {
    collectionName: string; collection: RetrieveCollectionType;
    filter?: Record<string, string>; limit: number; offset: number;
    projectId?: string; libraryName?: string;
  }): Promise<RetrieveResponse> {
    const { collectionName, collection, filter, limit, offset, projectId, libraryName } = params;

    try {
      const qdrantFilter = await this.buildFilter(collection, filter, projectId, libraryName);
      const scrollRequest: {
        limit: number; offset?: number;
        with_payload: boolean; with_vector: boolean;
        filter?: Record<string, unknown>;
      } = { limit: limit + 1, with_payload: true, with_vector: false };
      if (offset > 0) scrollRequest.offset = offset;
      if (qdrantFilter) scrollRequest.filter = qdrantFilter;

      const result = await this.qdrantClient.scroll(collectionName, scrollRequest);

      const hasMore = result.points.length > limit;
      const points = hasMore ? result.points.slice(0, limit) : result.points;

      const documents: RetrievedDocument[] = points.map((point) => ({
        id: String(point.id),
        content: (point.payload?.['content'] as string) ?? '',
        metadata: extractMetadata(point.payload),
      }));

      return { success: true, documents, total: documents.length, hasMore };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'unknown error';
      if (errorMessage.includes('not found') || errorMessage.includes('doesn\'t exist')) {
        return { success: true, documents: [], total: 0, hasMore: false, message: 'Collection not found or empty' };
      }
      return { success: false, documents: [], message: `Failed to retrieve documents: ${errorMessage}` };
    }
  }

  private async buildFilter(
    collection: RetrieveCollectionType,
    filter?: Record<string, string>,
    projectId?: string,
    libraryName?: string,
  ): Promise<Record<string, unknown> | null> {
    const mustConditions: Record<string, unknown>[] = [];

    if (collection === 'projects') {
      const tenantId = projectId ?? (await this.resolveProjectId());
      if (tenantId) mustConditions.push({ key: 'tenant_id', match: { value: tenantId } });
    } else if (collection === 'libraries' && libraryName) {
      mustConditions.push({ key: 'tenant_id', match: { value: libraryName } });
    }

    if (filter) {
      for (const [key, value] of Object.entries(filter)) {
        mustConditions.push({ key, match: { value } });
      }
    }

    return mustConditions.length > 0 ? { must: mustConditions } : null;
  }

  private async resolveProjectId(): Promise<string | undefined> {
    const cwd = process.cwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
    return projectInfo?.projectId;
  }
}
