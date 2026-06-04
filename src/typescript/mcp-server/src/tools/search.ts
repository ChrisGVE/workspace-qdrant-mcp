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
import type { QdrantClient } from '@qdrant/js-client-rest';
import { getQdrantClient } from '../clients/qdrant-client-factory.js';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { SERVER_VERSION as MCP_SERVER_VERSION } from '../server-types.js';

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
  SearchResult,
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
import { shapeHitPayloads } from './search-shaping.js';
import {
  resolveProjectContext,
  logSearchEventPre,
  generateEmbeddings,
  searchAllCollections,
  searchScratchpadLane,
  finalizeResults,
} from './search-helpers.js';

/** Format an explicit status_reason for the F-014 base-point degradation
 * case. Used in both the embedding-fallback path and the normal
 * fan-out path so callers always see why instance isolation was
 * bypassed. */
function formatBasePointsDegradedReason(activeCount: number | undefined): string {
  const count = activeCount ?? 'too many';
  return (
    `Worktree/instance isolation degraded: project has ${count} active base points, ` +
    `exceeding the 500-filter cap; tenant filter still applies but base-point ` +
    `narrowing was bypassed. Narrow further with pathGlob, branch, or component to ` +
    `restore worktree-level isolation.`
  );
}

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
    this.qdrantClient = getQdrantClient({
      url: config.qdrantUrl,
      apiKey: config.qdrantApiKey,
      timeout: config.qdrantTimeout ?? 5000,
    });
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
    projectId: string | undefined,
    eventId: string
  ): Promise<{
    searchStartMs: number;
    currentProjectId: string | undefined;
    basePoints: string[] | undefined;
    basePointsDegraded: boolean;
    basePointsActiveCount: number | undefined;
  }> {
    const searchStartMs = Date.now();
    const resolution = await resolveProjectContext(
      projectId,
      scope,
      this.projectDetector,
      this._stateManager
    );
    logSearchEventPre(this._stateManager, eventId, resolution.currentProjectId, query, limit, {
      collection: options.collection,
      scope,
      branch: options.branch,
      fileType: options.fileType,
      libraryName: options.libraryName,
      tag: options.tag,
    });
    return {
      searchStartMs,
      currentProjectId: resolution.currentProjectId,
      basePoints: resolution.basePoints,
      basePointsDegraded: resolution.basePointsDegraded ?? false,
      basePointsActiveCount: resolution.basePointsActiveCount,
    };
  }

  async search(options: SearchOptions): Promise<SearchResponse> {
    // Single shaping pass at the outer boundary keeps the budget-cap
    // logic in one place and applies to every exit (exact, fallback,
    // and normal pipeline) without sprinkling it through helpers.
    // The eventId is generated here so the post-shape token-economy
    // update can attribute metrics to the same search_events row that
    // the inner pipeline already wrote/updated. Spec:
    // docs/specs/20-token-economy-instrumentation.md
    const eventId = randomUUID();
    const response = await this.executeSearch(options, eventId);
    const { response: shaped, metrics } = shapeHitPayloads(response, options);
    this._stateManager.updateSearchEventEconomy(eventId, {
      bytesIn: metrics.bytesInShaped,
      bytesOut: metrics.bytesOutShaped,
      hitsTruncated: metrics.hitsTruncated,
      shapeMode: metrics.mode,
      toolVersion: MCP_SERVER_VERSION,
    });
    return shaped;
  }

  private async executeSearch(
    options: SearchOptions,
    eventId: string
  ): Promise<SearchResponse> {
    if (options.exact) {
      return searchExact(
        this.daemonClient,
        this._stateManager,
        this.projectDetector,
        options,
        eventId
      );
    }
    const mode = options.mode ?? 'hybrid';
    const limit = options.limit ?? DEFAULT_LIMIT;
    const scope = options.scope ?? 'project';
    const collectionsToSearch = determineCollections(
      options.collection,
      scope,
      options.includeLibraries ?? false
    );
    const {
      searchStartMs,
      currentProjectId,
      basePoints,
      basePointsDegraded,
      basePointsActiveCount,
    } = await this.resolveContextAndLog(
      options,
      options.query,
      limit,
      scope,
      options.projectId,
      eventId
    );
    const embeddings = await this.prepareEmbeddings(
      options,
      options.query,
      mode,
      collectionsToSearch,
      currentProjectId,
      basePoints
    );
    if ('fallback' in embeddings) {
      // F-014: if base-point isolation degraded, surface it on the
      // fallback response too — tenant filter still applies via
      // fallbackSearch, but instance/worktree narrowing was bypassed.
      if (basePointsDegraded) {
        embeddings.fallback.status = 'uncertain';
        const reason = formatBasePointsDegradedReason(basePointsActiveCount);
        embeddings.fallback.status_reason = embeddings.fallback.status_reason
          ? `${embeddings.fallback.status_reason}; ${reason}`
          : reason;
      }
      return embeddings.fallback;
    }
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
      basePointsDegraded,
      basePointsActiveCount,
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
    basePointsDegraded: boolean,
    basePointsActiveCount: number | undefined,
    denseEmbedding: number[] | undefined,
    sparseVector: Record<number, number> | undefined
  ): Promise<SearchResponse> {
    // Project-memory recall lane: enabled only for the default project scope,
    // when not targeting an explicit collection, with the lane on and a tenant
    // resolved. Resolve its tenant up front so the lane runs CONCURRENTLY with
    // the main collection fan-out — it reuses the same embeddings, so there is
    // no reason to serialize it behind the code search. `undefined` skips the
    // lane (resolves to []); failures inside the lane also degrade to [].
    const laneProjectId =
      scope === 'project' && !options.collection && options.includeScratchpad !== false
        ? currentProjectId
        : undefined;
    const [collectionsResult, scratchpadHits] = await Promise.all([
      this.runSearchCollections(
        options,
        mode,
        limit,
        scope,
        collectionsToSearch,
        currentProjectId,
        basePoints,
        denseEmbedding,
        sparseVector
      ),
      laneProjectId
        ? searchScratchpadLane(this.qdrantClient, {
            projectId: laneProjectId,
            mode,
            denseEmbedding,
            sparseVector,
            scoreThreshold: options.scoreThreshold ?? DEFAULT_SCORE_THRESHOLD,
          })
        : Promise.resolve<SearchResult[]>([]),
    ]);
    let { status, statusReason } = collectionsResult;
    if (basePointsDegraded) {
      // F-014: merge the explicit degradation message into the final
      // response so callers know instance isolation was bypassed.
      status = 'uncertain';
      const reason = formatBasePointsDegradedReason(basePointsActiveCount);
      statusReason = statusReason ? `${statusReason}; ${reason}` : reason;
    }
    return finalizeResults(this.qdrantClient, this.daemonClient, this._stateManager, {
      allResults: collectionsResult.allResults,
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
      currentProjectId,
      scratchpadHits,
    });
  }

  async retrieveParent(parentUnitId: string, collection: string): Promise<ParentContext | null> {
    return retrieveParent(this.qdrantClient, parentUnitId, collection);
  }

  async collectionExists(collectionName: string): Promise<boolean> {
    return collectionExists(this.qdrantClient, collectionName);
  }
}
