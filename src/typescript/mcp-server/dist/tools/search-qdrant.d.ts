/**
 * Qdrant search operations: collection search, RRF fusion, parent context,
 * fallback search, and collection existence check.
 */
import type { QdrantClient } from '@qdrant/js-client-rest';
import { type SearchMode, type SearchResult, type SearchCollectionParams, type ParentContext, type SearchOptions, type SearchResponse } from './search-types.js';
/**
 * Search a single collection with dense and/or sparse vectors.
 */
export declare function searchCollection(qdrantClient: QdrantClient, params: SearchCollectionParams): Promise<SearchResult[]>;
/**
 * Apply Reciprocal Rank Fusion to combine results.
 * RRF score = sum(1 / (k + rank_i)) for each result across rankings.
 */
export declare function applyRRFFusion(results: SearchResult[], mode: SearchMode): SearchResult[];
/** Expand parent context for search results (fetches parent unit records). */
export declare function expandParentContext(qdrantClient: QdrantClient, results: SearchResult[]): Promise<void>;
/** Retrieve a single parent unit by ID for on-demand expansion. */
export declare function retrieveParent(qdrantClient: QdrantClient, parentUnitId: string, collection: string): Promise<ParentContext | null>;
/** Resolved tenant context for the fallback path. */
export interface FallbackTenantContext {
    /** Resolved project ID — required when `scope === 'project'`. */
    currentProjectId: string | undefined;
    /** Optional base_points list for instance-aware filtering. */
    basePoints: string[] | undefined;
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
export declare function fallbackSearch(qdrantClient: QdrantClient, options: SearchOptions, collections: string[], context: FallbackTenantContext): Promise<SearchResponse>;
/** Check if a collection exists. */
export declare function collectionExists(qdrantClient: QdrantClient, collectionName: string): Promise<boolean>;
//# sourceMappingURL=search-qdrant.d.ts.map