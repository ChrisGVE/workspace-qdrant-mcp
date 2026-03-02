/**
 * Search tool facade — delegates to domain-specific modules.
 *
 * - search-types.ts: Types, interfaces, constants
 * - search-filters.ts: Filter construction, collection determination
 * - search-qdrant.ts: Qdrant search, RRF fusion, parent context, fallback
 * - search-exact.ts: FTS5 exact/substring search via daemon
 * - search-expansion.ts: Tag-based BM25 query expansion
 */

import { randomUUID } from 'node:crypto';
import { QdrantClient } from '@qdrant/js-client-rest';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';

// Re-export all types so existing imports from './search.js' continue to work
export type {
  SearchMode,
  SearchScope,
  SearchOptions,
  ParentContext,
  SearchResult,
  SearchResponse,
  SearchToolConfig,
  FilterParams,
  SearchCollectionParams,
  GraphContext,
  GraphContextNode,
} from './search-types.js';

import type {
  SearchOptions,
  SearchResponse,
  SearchResult,
  SearchToolConfig,
  FilterParams,
  SearchCollectionParams,
  ParentContext,
} from './search-types.js';
import {
  PROJECTS_COLLECTION,
  DEFAULT_LIMIT,
  DEFAULT_SCORE_THRESHOLD,
  DEFAULT_EXPANSION_WEIGHT,
  DEFAULT_MAX_EXPANDED_KEYWORDS,
} from './search-types.js';

import { determineCollections, buildFilter } from './search-filters.js';
import {
  searchCollection,
  applyRRFFusion,
  expandParentContext,
  retrieveParent,
  fallbackSearch,
  collectionExists,
} from './search-qdrant.js';
import { searchExact } from './search-exact.js';
import { expandSparseWithTags } from './search-expansion.js';
import { expandGraphContext } from './search-graph-context.js';

/**
 * Search tool for hybrid semantic + keyword search
 */
export class SearchTool {
  private readonly qdrantClient: QdrantClient;
  private readonly daemonClient: DaemonClient;
  private readonly _stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;
  private readonly enableTagExpansion: boolean;
  private readonly expansionWeight: number;
  private readonly maxExpandedKeywords: number;

  constructor(
    config: SearchToolConfig,
    daemonClient: DaemonClient,
    stateManager: SqliteStateManager,
    projectDetector: ProjectDetector
  ) {
    const clientConfig: { url: string; apiKey?: string; timeout?: number } = {
      url: config.qdrantUrl,
      timeout: config.qdrantTimeout ?? 5000,
    };
    if (config.qdrantApiKey) {
      clientConfig.apiKey = config.qdrantApiKey;
    }
    this.qdrantClient = new QdrantClient(clientConfig);
    this.daemonClient = daemonClient;
    this._stateManager = stateManager;
    this.projectDetector = projectDetector;
    this.enableTagExpansion = config.enableTagExpansion ?? true;
    this.expansionWeight = config.expansionWeight ?? DEFAULT_EXPANSION_WEIGHT;
    this.maxExpandedKeywords = config.maxExpandedKeywords ?? DEFAULT_MAX_EXPANDED_KEYWORDS;
  }

  get stateManager(): SqliteStateManager {
    return this._stateManager;
  }

  async search(options: SearchOptions): Promise<SearchResponse> {
    if (options.exact) {
      return searchExact(this.daemonClient, this._stateManager, this.projectDetector, options);
    }

    const {
      query,
      collection,
      mode = 'hybrid',
      limit = DEFAULT_LIMIT,
      scoreThreshold = DEFAULT_SCORE_THRESHOLD,
      scope = 'project',
      branch,
      fileType,
      projectId,
      libraryName,
      includeLibraries = false,
      tag,
      tags,
    } = options;

    // Search event instrumentation
    const eventId = randomUUID();
    const searchStartMs = Date.now();

    const collectionsToSearch = determineCollections(collection, scope, includeLibraries);

    // Resolve current project context
    let currentProjectId = projectId;
    if (!currentProjectId && scope === 'project') {
      const cwd = process.cwd();
      const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
      currentProjectId = projectInfo?.projectId;
    }

    // Task 15: Resolve base_points for instance-aware filtering
    let basePoints: string[] | undefined;
    if (currentProjectId && scope === 'project') {
      const watchFolderId = this._stateManager.getWatchFolderIdByTenantId(currentProjectId);
      if (watchFolderId) {
        const points = this._stateManager.getActiveBasePoints(watchFolderId, false);
        if (points.length > 0 && points.length <= 500) {
          basePoints = points;
        }
      }
    }

    // Log search event (pre-execution)
    const filters: Record<string, unknown> = {};
    if (collection) filters.collection = collection;
    if (scope !== 'project') filters.scope = scope;
    if (branch) filters.branch = branch;
    if (fileType) filters.file_type = fileType;
    if (libraryName) filters.library_name = libraryName;
    if (tag) filters.tag = tag;

    this._stateManager.logSearchEvent({
      id: eventId,
      projectId: currentProjectId,
      actor: 'claude',
      tool: 'mcp_qdrant',
      op: 'search',
      queryText: query,
      filters: Object.keys(filters).length > 0 ? JSON.stringify(filters) : undefined,
      topK: limit,
    });

    // Generate embeddings
    let denseEmbedding: number[] | undefined;
    let sparseVector: Record<number, number> | undefined;

    try {
      if (mode === 'hybrid' || mode === 'semantic') {
        const embedResponse = await this.daemonClient.embedText({ text: query });
        if (embedResponse.success) denseEmbedding = embedResponse.embedding;
      }
      if (mode === 'hybrid' || mode === 'keyword') {
        const sparseResponse = await this.daemonClient.generateSparseVector({ text: query });
        if (sparseResponse.success) sparseVector = sparseResponse.indices_values;
      }
    } catch {
      return fallbackSearch(this.qdrantClient, options, collectionsToSearch);
    }

    // Tag-based query expansion for BM25/sparse search
    if (this.enableTagExpansion && sparseVector && (mode === 'hybrid' || mode === 'keyword')) {
      sparseVector = await expandSparseWithTags(
        this.daemonClient, this._stateManager, query, sparseVector,
        collectionsToSearch, this.expansionWeight, this.maxExpandedKeywords,
        currentProjectId,
      );
    }

    // Search each collection
    const allResults: SearchResult[] = [];
    let status: 'ok' | 'uncertain' = 'ok';
    let statusReason: string | undefined;

    for (const coll of collectionsToSearch) {
      try {
        const filterParams: FilterParams = {
          collection: coll, scope, projectId: currentProjectId,
          branch, fileType, libraryName, tag, tags,
          pathGlob: options.pathGlob,
          component: options.component,
          basePoints: coll === PROJECTS_COLLECTION ? basePoints : undefined,
        };
        const filter = buildFilter(filterParams);

        const searchParams: SearchCollectionParams = {
          collection: coll, mode, denseEmbedding, sparseVector,
          filter, limit: limit * 2, scoreThreshold,
        };
        allResults.push(...await searchCollection(this.qdrantClient, searchParams));
      } catch (error) {
        status = 'uncertain';
        statusReason = `Some collections unavailable: ${error instanceof Error ? error.message : 'unknown'}`;
      }
    }

    // Apply RRF fusion, sort, limit
    const fusedResults = applyRRFFusion(allResults, mode);
    fusedResults.sort((a, b) => b.score - a.score);
    const finalResults = fusedResults.slice(0, limit);

    if (options.expandContext) {
      await expandParentContext(this.qdrantClient, finalResults);
    }

    if (options.includeGraphContext) {
      await expandGraphContext(this.daemonClient, finalResults);
    }

    // Update search event with results
    const latencyMs = Date.now() - searchStartMs;
    const topRefs = finalResults.slice(0, 5).map((r) => ({
      id: r.id,
      score: Math.round(r.score * 1000) / 1000,
      collection: r.collection,
    }));
    this._stateManager.updateSearchEvent(eventId, {
      resultCount: finalResults.length, latencyMs,
      topResultRefs: JSON.stringify(topRefs),
    });

    const response: SearchResponse = {
      results: finalResults, total: finalResults.length,
      query, mode, scope, collections_searched: collectionsToSearch, status,
    };
    if (statusReason) response.status_reason = statusReason;
    return response;
  }

  async retrieveParent(parentUnitId: string, collection: string): Promise<ParentContext | null> {
    return retrieveParent(this.qdrantClient, parentUnitId, collection);
  }

  async collectionExists(collectionName: string): Promise<boolean> {
    return collectionExists(this.qdrantClient, collectionName);
  }
}
