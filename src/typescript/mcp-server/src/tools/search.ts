/**
 * Search tool implementation with hybrid search support
 *
 * Provides:
 * - Hybrid search combining dense (semantic) and sparse (keyword) vectors
 * - Multiple search modes: hybrid, semantic, keyword
 * - Reciprocal Rank Fusion (RRF) for result combination
 * - Filter building based on collection type and scope
 * - Direct Qdrant queries with daemon embedding generation
 */

import { randomUUID } from 'node:crypto';
import { QdrantClient } from '@qdrant/js-client-rest';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';

// Canonical collection names from native bridge (single source of truth)
import { COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_SCRATCHPAD } from '../common/native-bridge.js';
const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
const SCRATCHPAD_COLLECTION = COLLECTION_SCRATCHPAD;

// Vector names for hybrid search
const DENSE_VECTOR_NAME = 'dense';
const SPARSE_VECTOR_NAME = 'sparse';

// RRF constant (k=60 is standard)
const RRF_K = 60;

// Default search parameters
const DEFAULT_LIMIT = 10;
const DEFAULT_SCORE_THRESHOLD = 0.3;

export type SearchMode = 'hybrid' | 'semantic' | 'keyword';
export type SearchScope = 'project' | 'global' | 'all';

export interface SearchOptions {
  query: string;
  collection?: string;
  mode?: SearchMode;
  limit?: number;
  scoreThreshold?: number;
  scope?: SearchScope;
  branch?: string;
  fileType?: string;
  projectId?: string;
  libraryName?: string;
  includeLibraries?: boolean;
  includeDeleted?: boolean;
  tag?: string;
  /** When true, fetch parent unit context for each chunk result */
  expandContext?: boolean;
}

export interface ParentContext {
  parent_unit_id: string;
  unit_type: string;
  unit_text: string;
  locator?: Record<string, unknown>;
}

export interface SearchResult {
  id: string;
  score: number;
  collection: string;
  content: string;
  title?: string;
  metadata: Record<string, unknown>;
  parent_context?: ParentContext;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  mode: SearchMode;
  scope: SearchScope;
  collections_searched: string[];
  status?: 'ok' | 'uncertain';
  status_reason?: string;
}

export interface SearchToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
}

interface FilterParams {
  collection: string;
  scope: SearchScope;
  projectId: string | undefined;
  branch: string | undefined;
  fileType: string | undefined;
  libraryName: string | undefined;
  includeDeleted: boolean;
  tag: string | undefined;
}

interface SearchCollectionParams {
  collection: string;
  mode: SearchMode;
  denseEmbedding: number[] | undefined;
  sparseVector: Record<number, number> | undefined;
  filter: Record<string, unknown> | null;
  limit: number;
  scoreThreshold: number;
}

/**
 * Search tool for hybrid semantic + keyword search
 */
export class SearchTool {
  private readonly qdrantClient: QdrantClient;
  private readonly daemonClient: DaemonClient;
  private readonly _stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;

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
  }

  /**
   * Get the state manager (for future use)
   */
  get stateManager(): SqliteStateManager {
    return this._stateManager;
  }

  /**
   * Execute search with specified options
   */
  async search(options: SearchOptions): Promise<SearchResponse> {
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
      includeDeleted = false,
      tag,
    } = options;

    // ── Search event instrumentation (Task 12) ──
    const eventId = randomUUID();
    const searchStartMs = Date.now();

    // Determine which collections to search
    const collectionsToSearch = this.determineCollections(
      collection,
      scope,
      includeLibraries
    );

    // Get current project context if needed
    let currentProjectId = projectId;
    if (!currentProjectId && scope === 'project') {
      const cwd = process.cwd();
      const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
      currentProjectId = projectInfo?.projectId;
    }

    // Log search event (pre-execution)
    const filters: Record<string, unknown> = {};
    if (collection) filters.collection = collection;
    if (scope !== 'project') filters.scope = scope;
    if (branch) filters.branch = branch;
    if (fileType) filters.file_type = fileType;
    if (libraryName) filters.library_name = libraryName;
    if (tag) filters.tag = tag;
    if (includeDeleted) filters.include_deleted = true;

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

    // Generate embeddings based on mode
    let denseEmbedding: number[] | undefined;
    let sparseVector: Record<number, number> | undefined;

    try {
      if (mode === 'hybrid' || mode === 'semantic') {
        const embedResponse = await this.daemonClient.embedText({ text: query });
        if (embedResponse.success) {
          denseEmbedding = embedResponse.embedding;
        }
      }

      if (mode === 'hybrid' || mode === 'keyword') {
        const sparseResponse = await this.daemonClient.generateSparseVector({ text: query });
        if (sparseResponse.success) {
          sparseVector = sparseResponse.indices_values;
        }
      }
    } catch {
      // Daemon unavailable - fall back to Qdrant scroll if possible
      return this.fallbackSearch(options, collectionsToSearch);
    }

    // Search each collection
    const allResults: SearchResult[] = [];
    let status: 'ok' | 'uncertain' = 'ok';
    let statusReason: string | undefined;

    for (const coll of collectionsToSearch) {
      try {
        // Build filter for this collection
        const filterParams: FilterParams = {
          collection: coll,
          scope,
          projectId: currentProjectId,
          branch,
          fileType,
          libraryName,
          includeDeleted,
          tag,
        };
        const filter = this.buildFilter(filterParams);

        // Execute search based on mode
        const searchParams: SearchCollectionParams = {
          collection: coll,
          mode,
          denseEmbedding,
          sparseVector,
          filter,
          limit: limit * 2, // Get more for fusion
          scoreThreshold,
        };
        const results = await this.searchCollection(searchParams);

        allResults.push(...results);
      } catch (error) {
        // Collection may not exist - continue with degraded status
        status = 'uncertain';
        statusReason = `Some collections unavailable: ${error instanceof Error ? error.message : 'unknown'}`;
      }
    }

    // Apply RRF fusion if we have results from multiple sources
    const fusedResults = this.applyRRFFusion(allResults, mode);

    // Sort by score and limit
    fusedResults.sort((a, b) => b.score - a.score);
    const finalResults = fusedResults.slice(0, limit);

    // Expand parent context if requested
    if (options.expandContext) {
      await this.expandParentContext(finalResults);
    }

    // ── Update search event with results (Task 12) ──
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

    if (statusReason) {
      response.status_reason = statusReason;
    }

    return response;
  }

  /**
   * Determine which collections to search based on scope
   */
  private determineCollections(
    collection: string | undefined,
    scope: SearchScope,
    includeLibraries: boolean
  ): string[] {
    // Explicit collection overrides scope
    if (collection) {
      return [collection];
    }

    switch (scope) {
      case 'project':
        return includeLibraries
          ? [PROJECTS_COLLECTION, LIBRARIES_COLLECTION]
          : [PROJECTS_COLLECTION];
      case 'global':
        return [PROJECTS_COLLECTION];
      case 'all':
        return [PROJECTS_COLLECTION, LIBRARIES_COLLECTION, SCRATCHPAD_COLLECTION];
      default:
        return [PROJECTS_COLLECTION];
    }
  }

  /**
   * Build Qdrant filter based on search parameters
   */
  private buildFilter(params: FilterParams): Record<string, unknown> | null {
    const mustConditions: Record<string, unknown>[] = [];
    const mustNotConditions: Record<string, unknown>[] = [];

    // Project filter for project scope (applies to both projects and libraries collections)
    if (params.scope === 'project' && params.projectId) {
      mustConditions.push({
        key: 'tenant_id',
        match: { value: params.projectId },
      });
    }

    // Branch filter
    if (params.branch && params.branch !== '*') {
      mustConditions.push({
        key: 'branch',
        match: { value: params.branch },
      });
    }

    // File type filter
    if (params.fileType) {
      mustConditions.push({
        key: 'file_type',
        match: { value: params.fileType },
      });
    }

    // Library name filter
    if (params.collection === LIBRARIES_COLLECTION && params.libraryName) {
      mustConditions.push({
        key: 'library_name',
        match: { value: params.libraryName },
      });
    }

    // Tag filter (dot-separated hierarchy)
    if (params.tag) {
      mustConditions.push({
        key: 'tag',
        match: { value: params.tag },
      });
    }

    // Exclude deleted libraries unless explicitly included
    if (params.collection === LIBRARIES_COLLECTION && !params.includeDeleted) {
      mustNotConditions.push({
        key: 'deleted',
        match: { value: true },
      });
    }

    // Build final filter
    if (mustConditions.length === 0 && mustNotConditions.length === 0) {
      return null;
    }

    const filter: Record<string, unknown> = {};
    if (mustConditions.length > 0) {
      filter.must = mustConditions;
    }
    if (mustNotConditions.length > 0) {
      filter.must_not = mustNotConditions;
    }

    return filter;
  }

  /**
   * Search a single collection
   */
  private async searchCollection(params: SearchCollectionParams): Promise<SearchResult[]> {
    const results: SearchResult[] = [];

    // Semantic/dense search
    if ((params.mode === 'hybrid' || params.mode === 'semantic') && params.denseEmbedding) {
      try {
        const searchRequest: {
          vector: { name: string; vector: number[] };
          limit: number;
          score_threshold: number;
          with_payload: boolean;
          filter?: Record<string, unknown>;
        } = {
          vector: {
            name: DENSE_VECTOR_NAME,
            vector: params.denseEmbedding,
          },
          limit: params.limit,
          score_threshold: params.scoreThreshold,
          with_payload: true,
        };

        if (params.filter) {
          searchRequest.filter = params.filter;
        }

        const searchResults = await this.qdrantClient.search(params.collection, searchRequest);

        for (const hit of searchResults) {
          const result: SearchResult = {
            id: String(hit.id),
            score: hit.score,
            collection: params.collection,
            content: (hit.payload?.['content'] as string) ?? '',
            metadata: { ...hit.payload, _search_type: 'semantic' },
          };
          const title = hit.payload?.['title'] as string | undefined;
          if (title) {
            result.title = title;
          }
          results.push(result);
        }
      } catch {
        // Collection may not support dense vectors
      }
    }

    // Keyword/sparse search
    if ((params.mode === 'hybrid' || params.mode === 'keyword') && params.sparseVector) {
      try {
        // Convert sparse vector to Qdrant format
        const indices = Object.keys(params.sparseVector).map(Number);
        const values = Object.values(params.sparseVector);

        if (indices.length > 0) {
          const searchRequest: {
            vector: { name: string; vector: { indices: number[]; values: number[] } };
            limit: number;
            score_threshold: number;
            with_payload: boolean;
            filter?: Record<string, unknown>;
          } = {
            vector: {
              name: SPARSE_VECTOR_NAME,
              vector: {
                indices,
                values,
              },
            },
            limit: params.limit,
            score_threshold: params.scoreThreshold * 0.5, // Lower threshold for sparse
            with_payload: true,
          };

          if (params.filter) {
            searchRequest.filter = params.filter;
          }

          const searchResults = await this.qdrantClient.search(params.collection, searchRequest);

          for (const hit of searchResults) {
            const result: SearchResult = {
              id: String(hit.id),
              score: hit.score,
              collection: params.collection,
              content: (hit.payload?.['content'] as string) ?? '',
              metadata: { ...hit.payload, _search_type: 'keyword' },
            };
            const title = hit.payload?.['title'] as string | undefined;
            if (title) {
              result.title = title;
            }
            results.push(result);
          }
        }
      } catch {
        // Collection may not support sparse vectors
      }
    }

    return results;
  }

  /**
   * Apply Reciprocal Rank Fusion to combine results
   * RRF score = sum(1 / (k + rank_i)) for each result across rankings
   */
  private applyRRFFusion(results: SearchResult[], mode: SearchMode): SearchResult[] {
    if (mode !== 'hybrid' || results.length === 0) {
      return results;
    }

    // Group results by search type
    const semanticResults = results.filter(
      (r) => r.metadata['_search_type'] === 'semantic'
    );
    const keywordResults = results.filter(
      (r) => r.metadata['_search_type'] === 'keyword'
    );

    // If only one type, return as-is
    if (semanticResults.length === 0 || keywordResults.length === 0) {
      return results;
    }

    // Build RRF scores
    const rrfScores = new Map<string, { score: number; result: SearchResult }>();

    // Add semantic results
    semanticResults.forEach((result, rank) => {
      const key = `${result.collection}:${result.id}`;
      const rrfScore = 1 / (RRF_K + rank + 1);
      const existing = rrfScores.get(key);
      if (existing) {
        existing.score += rrfScore;
      } else {
        rrfScores.set(key, { score: rrfScore, result: { ...result } });
      }
    });

    // Add keyword results
    keywordResults.forEach((result, rank) => {
      const key = `${result.collection}:${result.id}`;
      const rrfScore = 1 / (RRF_K + rank + 1);
      const existing = rrfScores.get(key);
      if (existing) {
        existing.score += rrfScore;
      } else {
        rrfScores.set(key, { score: rrfScore, result: { ...result } });
      }
    });

    // Convert back to results with fused scores
    return Array.from(rrfScores.values()).map(({ score, result }) => ({
      ...result,
      score,
      metadata: { ...result.metadata, _search_type: 'hybrid' },
    }));
  }

  /**
   * Expand parent context for search results.
   * Fetches parent unit records from Qdrant by their point IDs.
   */
  private async expandParentContext(results: SearchResult[]): Promise<void> {
    // Group results by collection and collect parent_unit_ids
    const parentsByCollection = new Map<string, Map<string, SearchResult[]>>();

    for (const result of results) {
      const parentId = result.metadata['parent_unit_id'] as string | undefined;
      if (!parentId) continue;

      let collMap = parentsByCollection.get(result.collection);
      if (!collMap) {
        collMap = new Map();
        parentsByCollection.set(result.collection, collMap);
      }

      let resultsForParent = collMap.get(parentId);
      if (!resultsForParent) {
        resultsForParent = [];
        collMap.set(parentId, resultsForParent);
      }
      resultsForParent.push(result);
    }

    // Fetch parent records from each collection
    for (const [collection, parentMap] of parentsByCollection) {
      const parentIds = Array.from(parentMap.keys());
      if (parentIds.length === 0) continue;

      try {
        const points = await this.qdrantClient.retrieve(collection, {
          ids: parentIds,
          with_payload: true,
        });

        for (const point of points) {
          const pointId = String(point.id);
          const linkedResults = parentMap.get(pointId);
          if (!linkedResults) continue;

          const parentContext: ParentContext = {
            parent_unit_id: pointId,
            unit_type: (point.payload?.['unit_type'] as string) ?? 'unknown',
            unit_text: (point.payload?.['unit_text'] as string) ?? '',
          };

          // Include locator if present
          const locator = point.payload?.['locator'] as Record<string, unknown> | undefined;
          if (locator) {
            parentContext.locator = locator;
          }

          for (const r of linkedResults) {
            r.parent_context = parentContext;
          }
        }
      } catch {
        // Parent records may not exist yet (pre-migration data)
      }
    }
  }

  /**
   * Retrieve a single parent unit by ID for on-demand expansion.
   */
  async retrieveParent(
    parentUnitId: string,
    collection: string
  ): Promise<ParentContext | null> {
    try {
      const points = await this.qdrantClient.retrieve(collection, {
        ids: [parentUnitId],
        with_payload: true,
      });

      const point = points[0];
      if (!point) return null;

      const context: ParentContext = {
        parent_unit_id: String(point.id),
        unit_type: (point.payload?.['unit_type'] as string) ?? 'unknown',
        unit_text: (point.payload?.['unit_text'] as string) ?? '',
      };

      const locator = point.payload?.['locator'] as Record<string, unknown> | undefined;
      if (locator) {
        context.locator = locator;
      }

      return context;
    } catch {
      return null;
    }
  }

  /**
   * Fallback search when daemon is unavailable
   * Uses Qdrant scroll for text matching
   */
  private async fallbackSearch(
    options: SearchOptions,
    collections: string[]
  ): Promise<SearchResponse> {
    const results: SearchResult[] = [];
    const queryLower = options.query.toLowerCase();

    for (const collection of collections) {
      try {
        // Use scroll to get documents and filter by content
        const scrollResult = await this.qdrantClient.scroll(collection, {
          limit: (options.limit ?? DEFAULT_LIMIT) * 3,
          with_payload: true,
        });

        for (const point of scrollResult.points) {
          const content = (point.payload?.['content'] as string) ?? '';
          const titlePayload = (point.payload?.['title'] as string) ?? '';

          // Simple text matching
          if (
            content.toLowerCase().includes(queryLower) ||
            titlePayload.toLowerCase().includes(queryLower)
          ) {
            const result: SearchResult = {
              id: String(point.id),
              score: 0.5, // Arbitrary score for text match
              collection,
              content,
              metadata: { ...point.payload, _search_type: 'fallback' },
            };
            if (titlePayload) {
              result.title = titlePayload;
            }
            results.push(result);
          }
        }
      } catch {
        // Collection may not exist
      }
    }

    // Limit results
    const limitedResults = results.slice(0, options.limit ?? DEFAULT_LIMIT);

    return {
      results: limitedResults,
      total: limitedResults.length,
      query: options.query,
      mode: options.mode ?? 'hybrid',
      scope: options.scope ?? 'project',
      collections_searched: collections,
      status: 'uncertain',
      status_reason: 'Daemon unavailable - using fallback text search',
    };
  }

  /**
   * Check if a collection exists
   */
  async collectionExists(collectionName: string): Promise<boolean> {
    try {
      await this.qdrantClient.getCollection(collectionName);
      return true;
    } catch {
      return false;
    }
  }
}
