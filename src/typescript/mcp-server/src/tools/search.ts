/**
 * Search tool facade — delegates to domain-specific modules.
 *
 * - search-types.ts: Types, interfaces, constants
 * - search-filters.ts: Filter construction, collection determination
 * - search-qdrant.ts: Qdrant search, RRF fusion, parent context, fallback
 * - search-exact.ts: FTS5 exact/substring search via daemon
 * - search-expansion.ts: Tag-based BM25 query expansion
 * - search-helpers.ts: Phase helpers (project context, embeddings, fan-out, finalize)
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
  SearchToolConfig,
  ParentContext,
} from './search-types.js';
import {
  DEFAULT_LIMIT,
  DEFAULT_SCORE_THRESHOLD,
  DEFAULT_EXPANSION_WEIGHT,
  DEFAULT_MAX_EXPANDED_KEYWORDS,
} from './search-types.js';

import { determineCollections } from './search-filters.js';
import { retrieveParent, collectionExists } from './search-qdrant.js';
import { searchExact } from './search-exact.js';
import { expandSparseWithTags } from './search-expansion.js';
import {
  resolveProjectContext,
  logSearchEventPre,
  generateEmbeddings,
  searchAllCollections,
  finalizeResults,
} from './search-helpers.js';

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

  private async prepareEmbeddings(
    options: SearchOptions,
    query: string,
    mode: import('./search-types.js').SearchMode,
    collectionsToSearch: string[],
    currentProjectId: string | undefined,
    basePoints: string[] | undefined
  ): Promise<
    | { fallback: SearchResponse }
    | { denseEmbedding: number[] | undefined; sparseVector: Record<number, number> | undefined }
  > {
    const embeddings = await generateEmbeddings(
      this.daemonClient,
      this.qdrantClient,
      query,
      mode,
      options,
      collectionsToSearch,
      { currentProjectId, basePoints }
    );
    if ('fallback' in embeddings) return embeddings;
    let { denseEmbedding, sparseVector } = embeddings;
    if (sparseVector && this.enableTagExpansion && (mode === 'hybrid' || mode === 'keyword')) {
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
    return { denseEmbedding, sparseVector };
  }

  private async resolveContextAndLog(
    options: SearchOptions,
    query: string,
    limit: number,
    scope: import('./search-types.js').SearchScope,
    projectId: string | undefined
  ): Promise<{
    eventId: string;
    searchStartMs: number;
    currentProjectId: string | undefined;
    basePoints: string[] | undefined;
  }> {
    const eventId = randomUUID();
    const searchStartMs = Date.now();
    const { currentProjectId, basePoints } = await resolveProjectContext(
      projectId,
      scope,
      this.projectDetector,
      this._stateManager
    );
    logSearchEventPre(this._stateManager, eventId, currentProjectId, query, limit, {
      collection: options.collection,
      scope,
      branch: options.branch,
      fileType: options.fileType,
      libraryName: options.libraryName,
      tag: options.tag,
    });
    return { eventId, searchStartMs, currentProjectId, basePoints };
  }

  async search(options: SearchOptions): Promise<SearchResponse> {
    if (options.exact) {
      return searchExact(this.daemonClient, this._stateManager, this.projectDetector, options);
    }
    const mode = options.mode ?? 'hybrid';
    const limit = options.limit ?? DEFAULT_LIMIT;
    const scope = options.scope ?? 'project';
    const collectionsToSearch = determineCollections(
      options.collection,
      scope,
      options.includeLibraries ?? false
    );
    const { eventId, searchStartMs, currentProjectId, basePoints } =
      await this.resolveContextAndLog(options, options.query, limit, scope, options.projectId);
    const embeddings = await this.prepareEmbeddings(
      options,
      options.query,
      mode,
      collectionsToSearch,
      currentProjectId,
      basePoints
    );
    if ('fallback' in embeddings) return embeddings.fallback;
    return this.runSearchAndFinalize(
      options,
      mode,
      limit,
      scope,
      collectionsToSearch,
      eventId,
      searchStartMs,
      currentProjectId,
      basePoints,
      embeddings.denseEmbedding,
      embeddings.sparseVector
    );
  }

  private async runSearchCollections(
    options: SearchOptions,
    mode: import('./search-types.js').SearchMode,
    limit: number,
    scope: import('./search-types.js').SearchScope,
    collectionsToSearch: string[],
    currentProjectId: string | undefined,
    basePoints: string[] | undefined,
    denseEmbedding: number[] | undefined,
    sparseVector: Record<number, number> | undefined
  ) {
    const scoreThreshold = options.scoreThreshold ?? DEFAULT_SCORE_THRESHOLD;
    return searchAllCollections(this.qdrantClient, {
      collectionsToSearch,
      scope,
      currentProjectId,
      basePoints,
      branch: options.branch,
      fileType: options.fileType,
      libraryName: options.libraryName,
      tag: options.tag,
      tags: options.tags,
      options,
      mode,
      denseEmbedding,
      sparseVector,
      limit,
      scoreThreshold,
    });
  }

  private async runSearchAndFinalize(
    options: SearchOptions,
    mode: import('./search-types.js').SearchMode,
    limit: number,
    scope: import('./search-types.js').SearchScope,
    collectionsToSearch: string[],
    eventId: string,
    searchStartMs: number,
    currentProjectId: string | undefined,
    basePoints: string[] | undefined,
    denseEmbedding: number[] | undefined,
    sparseVector: Record<number, number> | undefined
  ): Promise<SearchResponse> {
    const { allResults, status, statusReason } = await this.runSearchCollections(
      options,
      mode,
      limit,
      scope,
      collectionsToSearch,
      currentProjectId,
      basePoints,
      denseEmbedding,
      sparseVector
    );
    return finalizeResults(this.qdrantClient, this.daemonClient, this._stateManager, {
      allResults,
      mode,
      limit,
      options,
      eventId,
      searchStartMs,
      query: options.query,
      scope,
      collectionsToSearch,
      status,
      statusReason,
    });
  }

  async retrieveParent(parentUnitId: string, collection: string): Promise<ParentContext | null> {
    return retrieveParent(this.qdrantClient, parentUnitId, collection);
  }

  async collectionExists(collectionName: string): Promise<boolean> {
    return collectionExists(this.qdrantClient, collectionName);
  }
}
