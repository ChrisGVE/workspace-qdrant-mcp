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

    const { currentProjectId, basePoints } = await resolveProjectContext(
      projectId,
      scope,
      this.projectDetector,
      this._stateManager
    );

    logSearchEventPre(this._stateManager, eventId, currentProjectId, query, limit, {
      collection,
      scope,
      branch,
      fileType,
      libraryName,
      tag,
    });

    const embeddings = await generateEmbeddings(
      this.daemonClient,
      this.qdrantClient,
      query,
      mode,
      options,
      collectionsToSearch
    );
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

    return this.runSearchAndFinalize({
      query,
      mode,
      limit,
      scoreThreshold,
      scope,
      branch,
      fileType,
      libraryName,
      tag,
      tags,
      options,
      eventId,
      searchStartMs,
      collectionsToSearch,
      currentProjectId,
      basePoints,
      denseEmbedding,
      sparseVector,
    });
  }

  private async runSearchAndFinalize(p: {
    query: string;
    mode: import('./search-types.js').SearchMode;
    limit: number;
    scoreThreshold: number;
    scope: import('./search-types.js').SearchScope;
    branch: string | undefined;
    fileType: string | undefined;
    libraryName: string | undefined;
    tag: string | undefined;
    tags: string[] | undefined;
    options: SearchOptions;
    eventId: string;
    searchStartMs: number;
    collectionsToSearch: string[];
    currentProjectId: string | undefined;
    basePoints: string[] | undefined;
    denseEmbedding: number[] | undefined;
    sparseVector: Record<number, number> | undefined;
  }): Promise<SearchResponse> {
    const { allResults, status, statusReason } = await searchAllCollections(this.qdrantClient, {
      collectionsToSearch: p.collectionsToSearch,
      scope: p.scope,
      currentProjectId: p.currentProjectId,
      basePoints: p.basePoints,
      branch: p.branch,
      fileType: p.fileType,
      libraryName: p.libraryName,
      tag: p.tag,
      tags: p.tags,
      options: p.options,
      mode: p.mode,
      denseEmbedding: p.denseEmbedding,
      sparseVector: p.sparseVector,
      limit: p.limit,
      scoreThreshold: p.scoreThreshold,
    });

    return finalizeResults(this.qdrantClient, this.daemonClient, this._stateManager, {
      allResults,
      mode: p.mode,
      limit: p.limit,
      options: p.options,
      eventId: p.eventId,
      searchStartMs: p.searchStartMs,
      query: p.query,
      scope: p.scope,
      collectionsToSearch: p.collectionsToSearch,
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
