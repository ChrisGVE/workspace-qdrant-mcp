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
  PROJECTS_COLLECTION,
  type SearchMode,
  type SearchResult,
  type SearchCollectionParams,
  type ParentContext,
  type SearchOptions,
  type SearchResponse,
  type FilterParams,
} from './search-types.js';
import { buildFilter } from './search-filters.js';
import { FIELD_CONTENT, FIELD_TITLE, FIELD_PARENT_UNIT_ID } from '../common/native-bridge.js';

/** Map a Qdrant search hit to a SearchResult. */
function hitToResult(
  hit: { id: string | number; score: number; payload?: Record<string, unknown> | null },
  collection: string,
  searchType: 'semantic' | 'keyword'
): SearchResult {
  const result: SearchResult = {
    id: String(hit.id),
    score: hit.score,
    collection,
    content: (hit.payload?.[FIELD_CONTENT] as string) ?? '',
    metadata: { ...hit.payload, _search_type: searchType },
  };
  const title = hit.payload?.[FIELD_TITLE] as string | undefined;
  if (title) result.title = title;
  return result;
}

/** Run the dense (semantic) search leg. */
async function searchDense(
  qdrantClient: QdrantClient,
  params: SearchCollectionParams
): Promise<SearchResult[]> {
  if (!(params.mode === 'hybrid' || params.mode === 'semantic') || !params.denseEmbedding)
    return [];
  try {
    const req: {
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
    if (params.filter) req.filter = params.filter;
    const hits = await qdrantClient.search(params.collection, req);
    return hits.map((h) => hitToResult(h, params.collection, 'semantic'));
  } catch {
    return [];
  }
}

/** Run the sparse (keyword) search leg. */
async function searchSparse(
  qdrantClient: QdrantClient,
  params: SearchCollectionParams
): Promise<SearchResult[]> {
  if (!(params.mode === 'hybrid' || params.mode === 'keyword') || !params.sparseVector) return [];
  try {
    const indices = Object.keys(params.sparseVector).map(Number);
    const values = Object.values(params.sparseVector);
    if (indices.length === 0) return [];
    const req: {
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
    if (params.filter) req.filter = params.filter;
    const hits = await qdrantClient.search(params.collection, req);
    return hits.map((h) => hitToResult(h, params.collection, 'keyword'));
  } catch {
    return [];
  }
}

/**
 * Search a single collection with dense and/or sparse vectors.
 */
export async function searchCollection(
  qdrantClient: QdrantClient,
  params: SearchCollectionParams
): Promise<SearchResult[]> {
  const [dense, sparse] = await Promise.all([
    searchDense(qdrantClient, params),
    searchSparse(qdrantClient, params),
  ]);
  return [...dense, ...sparse];
}

/**
 * Stable per-file identity for a search hit. Every chunk of a file shares its
 * `document_id` (stable across branches); path fallbacks cover collections
 * that don't carry it, and the point id is the last resort (e.g. scratchpad
 * notes), which degrades gracefully to per-chunk identity. Shared by RRF
 * fusion and the same-file dedup in search-helpers so ranking and dedup agree
 * on granularity.
 */
export function resultFileKey(result: SearchResult): string {
  return (
    (result.metadata['document_id'] as string | undefined) ??
    (result.metadata['relative_path'] as string | undefined) ??
    (result.metadata['file_path'] as string | undefined) ??
    result.id
  );
}

/** Down-weight applied to sparse-leg chunks that the dense leg did NOT also
 * retrieve ("unconfirmed" keyword hits).
 *
 * Why: for natural-language queries the BM25 leg is noise-heavy — files
 * stuffed with common query terms (README, reference docs, changelogs) fill
 * its ranking, and plain RRF hands each of them a vote interleaved with the
 * dense leg's, displacing semantically-found code from the final top-k. A
 * chunk the dense leg retrieved as well needs no demotion: the same-chunk
 * double vote is a high-precision agreement signal and is what hybrid's
 * top-1/top-3 edge over pure semantic comes from (measured on the 12-query
 * known-item benchmark). So fusion keeps both-leg sums at full strength and
 * halves only the sparse-ONLY entries.
 *
 * 0.5 puts the best unconfirmed sparse vote (0.5/(k+1) ≈ 0.0082) just below
 * the dense leg's floor at the default pool depth (1/(k+50) ≈ 0.0091,
 * k = 60, pool = limit×5 = 50), so unconfirmed keyword hits act as backfill
 * below dense results rather than displacing them. The path-relevance boost
 * (up to ×1.8) applied after fusion can still lift a sparse-only hit whose
 * file path/symbol matches the query — the exact-identifier rescue case —
 * past the dense tail. */
const SPARSE_ONLY_WEIGHT = 0.5;

/**
 * Apply Reciprocal Rank Fusion to combine results.
 * RRF score = sum(1 / (k + rank_i)) for each chunk across the two leg
 * rankings, with sparse-only chunks down-weighted by
 * {@link SPARSE_ONLY_WEIGHT} (dense-primary fusion: the dense leg orders the
 * list, same-chunk agreement boosts, unconfirmed keyword hits backfill).
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
    // Confirmed by the dense leg: full-strength vote on top of the dense one.
    // Dense leg never saw this chunk: demoted, unconfirmed keyword evidence.
    if (existing) existing.score += rrfScore;
    else rrfScores.set(key, { score: rrfScore * SPARSE_ONLY_WEIGHT, result: { ...result } });
  });

  return Array.from(rrfScores.values()).map(({ score, result }) => ({
    ...result,
    score,
    metadata: { ...result.metadata, _search_type: 'hybrid' },
  }));
}

/** Index results by collection and parent ID for batch retrieval. */
function indexByParent(results: SearchResult[]): Map<string, Map<string, SearchResult[]>> {
  const parentsByCollection = new Map<string, Map<string, SearchResult[]>>();
  for (const result of results) {
    const parentId = result.metadata[FIELD_PARENT_UNIT_ID] as string | undefined;
    if (!parentId) continue;
    let collMap = parentsByCollection.get(result.collection);
    if (!collMap) {
      collMap = new Map();
      parentsByCollection.set(result.collection, collMap);
    }
    let bucket = collMap.get(parentId);
    if (!bucket) {
      bucket = [];
      collMap.set(parentId, bucket);
    }
    bucket.push(result);
  }
  return parentsByCollection;
}

/** Fetch parent points for one collection and attach context to linked results. */
async function attachParentsForCollection(
  qdrantClient: QdrantClient,
  collection: string,
  parentMap: Map<string, SearchResult[]>
): Promise<void> {
  const parentIds = Array.from(parentMap.keys());
  if (parentIds.length === 0) return;
  try {
    const points = await qdrantClient.retrieve(collection, { ids: parentIds, with_payload: true });
    for (const point of points) {
      const linked = parentMap.get(String(point.id));
      if (!linked) continue;
      const ctx: ParentContext = {
        parent_unit_id: String(point.id),
        unit_type: (point.payload?.['unit_type'] as string) ?? 'unknown',
        unit_text: (point.payload?.['unit_text'] as string) ?? '',
      };
      const locator = point.payload?.['locator'] as Record<string, unknown> | undefined;
      if (locator) ctx.locator = locator;
      for (const r of linked) r.parent_context = ctx;
    }
  } catch {
    // Parent records may not exist yet (pre-migration data)
  }
}

/** Expand parent context for search results (fetches parent unit records). */
export async function expandParentContext(
  qdrantClient: QdrantClient,
  results: SearchResult[]
): Promise<void> {
  const parentsByCollection = indexByParent(results);
  for (const [collection, parentMap] of parentsByCollection) {
    await attachParentsForCollection(qdrantClient, collection, parentMap);
  }
}

/** Retrieve a single parent unit by ID for on-demand expansion. */
export async function retrieveParent(
  qdrantClient: QdrantClient,
  parentUnitId: string,
  collection: string
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

/** Text-match points from a scroll result against a query string. */
function matchScrollPoints(
  points: Array<{ id: string | number; payload?: Record<string, unknown> | null }>,
  collection: string,
  queryLower: string
): SearchResult[] {
  const matched: SearchResult[] = [];
  for (const point of points) {
    const content = (point.payload?.[FIELD_CONTENT] as string) ?? '';
    const titlePayload = (point.payload?.[FIELD_TITLE] as string) ?? '';
    if (
      !content.toLowerCase().includes(queryLower) &&
      !titlePayload.toLowerCase().includes(queryLower)
    )
      continue;
    const result: SearchResult = {
      id: String(point.id),
      score: 0.5,
      collection,
      content,
      metadata: { ...point.payload, _search_type: 'fallback' },
    };
    if (titlePayload) result.title = titlePayload;
    matched.push(result);
  }
  return matched;
}

/** Resolved tenant context for the fallback path. */
export interface FallbackTenantContext {
  /** Resolved project ID — required when `scope === 'project'`. */
  currentProjectId: string | undefined;
  /** Optional base_points list for instance-aware filtering. */
  basePoints: string[] | undefined;
}

/**
 * Build the per-collection scroll filter for the fallback path. Returns
 * `null` when scope = `'project'` but the tenant could not be resolved —
 * the caller MUST refuse to scroll in that case rather than broaden to
 * every tenant in the collection (F-001).
 */
function buildFallbackFilter(
  collection: string,
  options: SearchOptions,
  context: FallbackTenantContext
): Record<string, unknown> | null {
  const scope = options.scope ?? 'project';
  // F-001: project-scope fallback REQUIRES a resolved tenant. Without
  // one, refuse to scroll — broad scroll + local substring match would
  // leak documents from foreign tenants.
  if (scope === 'project' && !context.currentProjectId) return null;
  const filterParams: FilterParams = {
    collection,
    scope,
    projectId: context.currentProjectId,
    branch: options.branch,
    fileType: options.fileType,
    libraryName: options.libraryName,
    tag: options.tag,
    tags: options.tags,
    pathGlob: options.pathGlob,
    component: options.component,
    basePoints: collection === PROJECTS_COLLECTION ? context.basePoints : undefined,
  };
  return buildFilter(filterParams);
}

/**
 * Fallback search when daemon is unavailable.
 *
 * Closes F-001: scrolls Qdrant with a tenant/project/library filter built
 * from the same `buildFilter` helper as the primary search path. If the
 * filter cannot be assembled (project-scope with unresolved project), no
 * scroll is performed and the caller receives a degraded, empty result
 * with a `status_reason` explaining why. Local substring matching on raw
 * scroll output has been removed — when filtering is in place the scroll
 * itself is the matcher; when it isn't, we refuse to read.
 */
export async function fallbackSearch(
  qdrantClient: QdrantClient,
  options: SearchOptions,
  collections: string[],
  context: FallbackTenantContext
): Promise<SearchResponse> {
  const results: SearchResult[] = [];
  const queryLower = options.query.toLowerCase();
  const refusedCollections: string[] = [];
  let attemptedCollections = 0;

  for (const collection of collections) {
    const filter = buildFallbackFilter(collection, options, context);
    if (!filter) {
      // F-001: project-scope fallback with unresolved tenant — never scroll.
      refusedCollections.push(collection);
      continue;
    }
    attemptedCollections += 1;
    try {
      const scrollResult = await qdrantClient.scroll(collection, {
        limit: (options.limit ?? DEFAULT_LIMIT) * 3,
        with_payload: true,
        filter,
      });
      // Substring-match within the tenant-scoped page only (the filter
      // already enforces isolation; this is a coarse keyword filter).
      results.push(...matchScrollPoints(scrollResult.points, collection, queryLower));
    } catch {
      // Collection may not exist
    }
  }

  const limitedResults = results.slice(0, options.limit ?? DEFAULT_LIMIT);
  const scope = options.scope ?? 'project';
  const isDegraded = attemptedCollections === 0 && refusedCollections.length > 0;
  const statusReason = isDegraded
    ? `Daemon unavailable and project scope unresolved - cannot run cross-tenant fallback. Refused collections: ${refusedCollections.join(', ')}`
    : 'Daemon unavailable - using fallback text search';
  // No indexing block here: fallback runs precisely when the daemon
  // is unreachable. The cached probe in search-helpers will return any
  // recent value transparently on the normal path; we don't fire a fresh
  // gRPC here only to have it fail.
  return {
    results: limitedResults,
    total: limitedResults.length,
    query: options.query,
    mode: options.mode ?? 'hybrid',
    scope,
    collections_searched: collections,
    status: 'uncertain',
    status_reason: statusReason,
  };
}

/** Check if a collection exists. */
export async function collectionExists(
  qdrantClient: QdrantClient,
  collectionName: string
): Promise<boolean> {
  try {
    await qdrantClient.getCollection(collectionName);
    return true;
  } catch {
    return false;
  }
}
