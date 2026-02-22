/**
 * Qdrant search operations: collection search, RRF fusion, parent context,
 * fallback search, and collection existence check.
 */
import type { QdrantClient } from '@qdrant/js-client-rest';
import {
  DENSE_VECTOR_NAME,
  SPARSE_VECTOR_NAME,
  RRF_K,
  DEFAULT_LIMIT,
  type SearchMode,
  type SearchResult,
  type SearchCollectionParams,
  type ParentContext,
  type SearchOptions,
  type SearchResponse,
} from './search-types.js';
import { FIELD_CONTENT, FIELD_TITLE, FIELD_PARENT_UNIT_ID } from '../common/native-bridge.js';

/**
 * Search a single collection with dense and/or sparse vectors.
 */
export async function searchCollection(
  qdrantClient: QdrantClient,
  params: SearchCollectionParams,
): Promise<SearchResult[]> {
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
        vector: { name: DENSE_VECTOR_NAME, vector: params.denseEmbedding },
        limit: params.limit,
        score_threshold: params.scoreThreshold,
        with_payload: true,
      };
      if (params.filter) searchRequest.filter = params.filter;

      const searchResults = await qdrantClient.search(params.collection, searchRequest);
      for (const hit of searchResults) {
        const result: SearchResult = {
          id: String(hit.id),
          score: hit.score,
          collection: params.collection,
          content: (hit.payload?.[FIELD_CONTENT] as string) ?? '',
          metadata: { ...hit.payload, _search_type: 'semantic' },
        };
        const title = hit.payload?.[FIELD_TITLE] as string | undefined;
        if (title) result.title = title;
        results.push(result);
      }
    } catch {
      // Collection may not support dense vectors
    }
  }

  // Keyword/sparse search
  if ((params.mode === 'hybrid' || params.mode === 'keyword') && params.sparseVector) {
    try {
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
          vector: { name: SPARSE_VECTOR_NAME, vector: { indices, values } },
          limit: params.limit,
          score_threshold: params.scoreThreshold * 0.5,
          with_payload: true,
        };
        if (params.filter) searchRequest.filter = params.filter;

        const searchResults = await qdrantClient.search(params.collection, searchRequest);
        for (const hit of searchResults) {
          const result: SearchResult = {
            id: String(hit.id),
            score: hit.score,
            collection: params.collection,
            content: (hit.payload?.[FIELD_CONTENT] as string) ?? '',
            metadata: { ...hit.payload, _search_type: 'keyword' },
          };
          const title = hit.payload?.[FIELD_TITLE] as string | undefined;
          if (title) result.title = title;
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
 * Apply Reciprocal Rank Fusion to combine results.
 * RRF score = sum(1 / (k + rank_i)) for each result across rankings.
 */
export function applyRRFFusion(results: SearchResult[], mode: SearchMode): SearchResult[] {
  if (mode !== 'hybrid' || results.length === 0) return results;

  const semanticResults = results.filter((r) => r.metadata['_search_type'] === 'semantic');
  const keywordResults = results.filter((r) => r.metadata['_search_type'] === 'keyword');

  if (semanticResults.length === 0 || keywordResults.length === 0) return results;

  const rrfScores = new Map<string, { score: number; result: SearchResult }>();

  semanticResults.forEach((result, rank) => {
    const key = `${result.collection}:${result.id}`;
    const rrfScore = 1 / (RRF_K + rank + 1);
    const existing = rrfScores.get(key);
    if (existing) existing.score += rrfScore;
    else rrfScores.set(key, { score: rrfScore, result: { ...result } });
  });

  keywordResults.forEach((result, rank) => {
    const key = `${result.collection}:${result.id}`;
    const rrfScore = 1 / (RRF_K + rank + 1);
    const existing = rrfScores.get(key);
    if (existing) existing.score += rrfScore;
    else rrfScores.set(key, { score: rrfScore, result: { ...result } });
  });

  return Array.from(rrfScores.values()).map(({ score, result }) => ({
    ...result,
    score,
    metadata: { ...result.metadata, _search_type: 'hybrid' },
  }));
}

/** Expand parent context for search results (fetches parent unit records). */
export async function expandParentContext(
  qdrantClient: QdrantClient,
  results: SearchResult[],
): Promise<void> {
  const parentsByCollection = new Map<string, Map<string, SearchResult[]>>();

  for (const result of results) {
    const parentId = result.metadata[FIELD_PARENT_UNIT_ID] as string | undefined;
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

  for (const [collection, parentMap] of parentsByCollection) {
    const parentIds = Array.from(parentMap.keys());
    if (parentIds.length === 0) continue;

    try {
      const points = await qdrantClient.retrieve(collection, {
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

        const locator = point.payload?.['locator'] as Record<string, unknown> | undefined;
        if (locator) parentContext.locator = locator;

        for (const r of linkedResults) r.parent_context = parentContext;
      }
    } catch {
      // Parent records may not exist yet (pre-migration data)
    }
  }
}

/** Retrieve a single parent unit by ID for on-demand expansion. */
export async function retrieveParent(
  qdrantClient: QdrantClient,
  parentUnitId: string,
  collection: string,
): Promise<ParentContext | null> {
  try {
    const points = await qdrantClient.retrieve(collection, {
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
    if (locator) context.locator = locator;

    return context;
  } catch {
    return null;
  }
}

/** Fallback search when daemon is unavailable (Qdrant scroll text matching). */
export async function fallbackSearch(
  qdrantClient: QdrantClient,
  options: SearchOptions,
  collections: string[],
): Promise<SearchResponse> {
  const results: SearchResult[] = [];
  const queryLower = options.query.toLowerCase();

  for (const collection of collections) {
    try {
      const scrollResult = await qdrantClient.scroll(collection, {
        limit: (options.limit ?? DEFAULT_LIMIT) * 3,
        with_payload: true,
      });

      for (const point of scrollResult.points) {
        const content = (point.payload?.[FIELD_CONTENT] as string) ?? '';
        const titlePayload = (point.payload?.[FIELD_TITLE] as string) ?? '';

        if (
          content.toLowerCase().includes(queryLower) ||
          titlePayload.toLowerCase().includes(queryLower)
        ) {
          const result: SearchResult = {
            id: String(point.id),
            score: 0.5,
            collection,
            content,
            metadata: { ...point.payload, _search_type: 'fallback' },
          };
          if (titlePayload) result.title = titlePayload;
          results.push(result);
        }
      }
    } catch {
      // Collection may not exist
    }
  }

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

/** Check if a collection exists. */
export async function collectionExists(
  qdrantClient: QdrantClient,
  collectionName: string,
): Promise<boolean> {
  try {
    await qdrantClient.getCollection(collectionName);
    return true;
  } catch {
    return false;
  }
}
