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
  SearchMode,
  SearchScope,
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

    const eventId = randomUUID();
    const searchStartMs = Date.now();
    const collectionsToSearch = determineCollections(collection, scope, includeLibraries);

    const { currentProjectId, basePoints } = await this.resolveProjectContext(projectId, scope);

    this.logSearchEventPre(eventId, currentProjectId, query, limit, {
      collection,
      scope,
      branch,
      fileType,
      libraryName,
      tag,
    });

    const embeddings = await this.generateEmbeddings(query, mode, options, collectionsToSearch);
    if ('fallback' in embeddings) return embeddings.fallback;
    let { denseEmbedding, sparseVector } = embeddings;

    if (this.enableTagExpansion && sparseVector && (mode === 'hybrid' || mode === 'keyword')) {
      sparseVector = await expandSparseWithTags(
        this.daemonClient,
        this._stateManager,
        query,
        sparseVector,
        collectionsToSearch,
        this.expansionWeight,
        this.maxExpandedKeywords,
        currentProjectId
      );
    }

    const { allResults, status, statusReason } = await this.searchAllCollections(
      collectionsToSearch,
      scope,
      currentProjectId,
      basePoints,
      branch,
      fileType,
      libraryName,
      tag,
      tags,
      options,
      mode,
      denseEmbedding,
      sparseVector,
      limit,
      scoreThreshold
    );

    return this.finalizeResults(
      allResults,
      mode,
      limit,
      options,
      eventId,
      searchStartMs,
      query,
      scope,
      collectionsToSearch,
      status,
      statusReason
    );
  }

  /** Resolve current project ID and base_points for instance-aware filtering. */
  private async resolveProjectContext(
    projectId: string | undefined,
    scope: SearchScope
  ): Promise<{ currentProjectId: string | undefined; basePoints: string[] | undefined }> {
    let currentProjectId = projectId;
    if (!currentProjectId && scope === 'project') {
      const projectInfo = await this.projectDetector.getProjectInfo(process.cwd(), false);
      currentProjectId = projectInfo?.projectId;
    }

    let basePoints: string[] | undefined;
    if (currentProjectId && scope === 'project') {
      const watchFolderId = this._stateManager.getWatchFolderIdByTenantId(currentProjectId);
      if (watchFolderId) {
        const points = this._stateManager.getActiveBasePoints(watchFolderId, false);
        if (points.length > 0 && points.length <= 500) basePoints = points;
      }
    }
    return { currentProjectId, basePoints };
  }

  /** Log the pre-execution search event. */
  private logSearchEventPre(
    eventId: string,
    projectId: string | undefined,
    query: string,
    limit: number,
    opts: {
      collection?: string | undefined;
      scope: SearchScope;
      branch?: string | undefined;
      fileType?: string | undefined;
      libraryName?: string | undefined;
      tag?: string | undefined;
    }
  ): void {
    const filters: Record<string, unknown> = {};
    if (opts.collection) filters.collection = opts.collection;
    if (opts.scope !== 'project') filters.scope = opts.scope;
    if (opts.branch) filters.branch = opts.branch;
    if (opts.fileType) filters.file_type = opts.fileType;
    if (opts.libraryName) filters.library_name = opts.libraryName;
    if (opts.tag) filters.tag = opts.tag;
    this._stateManager.logSearchEvent({
      id: eventId,
      projectId,
      actor: 'claude',
      tool: 'mcp_qdrant',
      op: 'search',
      queryText: query,
      filters: Object.keys(filters).length > 0 ? JSON.stringify(filters) : undefined,
      topK: limit,
    });
  }

  /** Generate dense and sparse embeddings. Returns `{ fallback }` on error. */
  private async generateEmbeddings(
    query: string,
    mode: SearchMode,
    options: SearchOptions,
    collectionsToSearch: string[]
  ): Promise<
    | { denseEmbedding: number[] | undefined; sparseVector: Record<number, number> | undefined }
    | { fallback: SearchResponse }
  > {
    let denseEmbedding: number[] | undefined;
    let sparseVector: Record<number, number> | undefined;
    try {
      if (mode === 'hybrid' || mode === 'semantic') {
        const r = await this.daemonClient.embedText({ text: query });
        if (r.success) denseEmbedding = r.embedding;
      }
      if (mode === 'hybrid' || mode === 'keyword') {
        const r = await this.daemonClient.generateSparseVector({ text: query });
        if (r.success) sparseVector = r.indices_values;
      }
    } catch {
      return { fallback: await fallbackSearch(this.qdrantClient, options, collectionsToSearch) };
    }
    return { denseEmbedding, sparseVector };
  }

  /** Search all target collections and collect results, tolerating partial failures. */
  private async searchAllCollections(
    collectionsToSearch: string[],
    scope: SearchScope,
    currentProjectId: string | undefined,
    basePoints: string[] | undefined,
    branch: string | undefined,
    fileType: string | undefined,
    libraryName: string | undefined,
    tag: string | undefined,
    tags: string[] | undefined,
    options: SearchOptions,
    mode: SearchMode,
    denseEmbedding: number[] | undefined,
    sparseVector: Record<number, number> | undefined,
    limit: number,
    scoreThreshold: number
  ): Promise<{
    allResults: SearchResult[];
    status: 'ok' | 'uncertain';
    statusReason: string | undefined;
  }> {
    const allResults: SearchResult[] = [];
    let status: 'ok' | 'uncertain' = 'ok';
    let statusReason: string | undefined;

    for (const coll of collectionsToSearch) {
      try {
        const filterParams: FilterParams = {
          collection: coll,
          scope,
          projectId: currentProjectId,
          branch,
          fileType,
          libraryName,
          tag,
          tags,
          pathGlob: options.pathGlob,
          component: options.component,
          basePoints: coll === PROJECTS_COLLECTION ? basePoints : undefined,
        };
        const searchParams: SearchCollectionParams = {
          collection: coll,
          mode,
          denseEmbedding,
          sparseVector,
          filter: buildFilter(filterParams),
          limit: limit * 2,
          scoreThreshold,
        };
        allResults.push(...(await searchCollection(this.qdrantClient, searchParams)));
      } catch (error) {
        status = 'uncertain';
        statusReason = `Some collections unavailable: ${error instanceof Error ? error.message : 'unknown'}`;
      }
    }
    return { allResults, status, statusReason };
  }

  /** Fuse, sort, expand context, update event, and assemble the final response. */
  private async finalizeResults(
    allResults: SearchResult[],
    mode: SearchMode,
    limit: number,
    options: SearchOptions,
    eventId: string,
    searchStartMs: number,
    query: string,
    scope: SearchScope,
    collectionsToSearch: string[],
    status: 'ok' | 'uncertain',
    statusReason: string | undefined
  ): Promise<SearchResponse> {
    const fusedResults = applyRRFFusion(allResults, mode);
    fusedResults.sort((a, b) => b.score - a.score);
    const finalResults = fusedResults.slice(0, limit);

    if (options.expandContext) await expandParentContext(this.qdrantClient, finalResults);
    if (options.includeGraphContext) await expandGraphContext(this.daemonClient, finalResults);

    const latencyMs = Date.now() - searchStartMs;
    const topRefs = finalResults.slice(0, 5).map((r) => ({
      id: r.id,
      score: Math.round(r.score * 1000) / 1000,
      collection: r.collection,
    }));
    this._stateManager.updateSearchEvent(eventId, {
      resultCount: finalResults.length,
      latencyMs,
      topResultRefs: JSON.stringify(topRefs),
    });

    const response: SearchResponse = {
      results: finalResults,
      total: finalResults.length,
      query,
      mode,
      scope,
      collections_searched: collectionsToSearch,
      status,
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
