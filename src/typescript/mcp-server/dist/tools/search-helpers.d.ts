/**
 * Search tool helper functions — phase-level operations extracted from SearchTool.
 *
 * Each function corresponds to one phase of the hybrid search pipeline:
 *   resolveProjectContext  → project/instance disambiguation
 *   logSearchEventPre      → pre-execution telemetry
 *   generateEmbeddings     → dense + sparse vector generation
 *   searchAllCollections   → per-collection fan-out
 *   finalizeResults        → RRF fusion, context expansion, telemetry update
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { SearchMode, SearchScope, SearchOptions, SearchResponse, SearchResult } from './search-types.js';
/** Resolution outcome for project-scoped search context.
 *
 * `basePointsDegraded` (F-014) surfaces the case where the active
 * base-point set exceeds {@link BASE_POINTS_FILTER_CAP}. Pre-fix the
 * filter was silently dropped (worktree/instance isolation broadened
 * to the whole tenant). The flag lets the caller report `status:
 * 'uncertain'` with an explicit `status_reason` instead. */
export interface ProjectContextResolution {
    currentProjectId: string | undefined;
    basePoints: string[] | undefined;
    /** True when active base-point count exceeds the filter cap. */
    basePointsDegraded?: boolean;
    /** Active base-point count when degraded — useful for the caller's
     *  `status_reason` message. */
    basePointsActiveCount?: number;
}
/** Resolve current project ID and base_points for instance-aware filtering.
 *
 * Project detection runs for all scopes because the current project ID is
 * needed both for tenant filtering (project/group) and relevance decay (group/all). */
export declare function resolveProjectContext(projectId: string | undefined, scope: SearchScope, projectDetector: ProjectDetector, stateManager: SqliteStateManager): Promise<ProjectContextResolution>;
/** Log the pre-execution search event. */
export declare function logSearchEventPre(stateManager: SqliteStateManager, eventId: string, projectId: string | undefined, query: string, limit: number, opts: {
    collection?: string | undefined;
    scope: SearchScope;
    branch?: string | undefined;
    fileType?: string | undefined;
    libraryName?: string | undefined;
    tag?: string | undefined;
}): void;
/** Generate dense and sparse embeddings. Returns `{ fallback }` on daemon error. */
export declare function generateEmbeddings(daemonClient: DaemonClient, qdrantClient: QdrantClient, query: string, mode: SearchMode, options: SearchOptions, collectionsToSearch: string[], fallbackContext: {
    currentProjectId: string | undefined;
    basePoints: string[] | undefined;
}): Promise<{
    denseEmbedding: number[] | undefined;
    sparseVector: Record<number, number> | undefined;
} | {
    fallback: SearchResponse;
}>;
export interface SearchAllCollectionsParams {
    collectionsToSearch: string[];
    scope: SearchScope;
    currentProjectId: string | undefined;
    groupTenantIds: string[] | undefined;
    basePoints: string[] | undefined;
    branch: string | undefined;
    fileType: string | undefined;
    libraryName: string | undefined;
    libraryPath: string | undefined;
    tag: string | undefined;
    tags: string[] | undefined;
    options: SearchOptions;
    mode: SearchMode;
    denseEmbedding: number[] | undefined;
    sparseVector: Record<number, number> | undefined;
    limit: number;
    scoreThreshold: number;
}
/** Search all target collections and collect results, tolerating partial failures. */
export declare function searchAllCollections(qdrantClient: QdrantClient, params: SearchAllCollectionsParams): Promise<{
    allResults: SearchResult[];
    status: 'ok' | 'uncertain';
    statusReason: string | undefined;
}>;
export interface FinalizeResultsParams {
    allResults: SearchResult[];
    mode: SearchMode;
    limit: number;
    options: SearchOptions;
    eventId: string;
    searchStartMs: number;
    query: string;
    scope: SearchScope;
    collectionsToSearch: string[];
    status: 'ok' | 'uncertain';
    statusReason: string | undefined;
}
/** Fuse, sort, apply diversity re-ranking, expand context, update event, and assemble the final response. */
export declare function finalizeResults(qdrantClient: QdrantClient, daemonClient: DaemonClient, stateManager: SqliteStateManager, params: FinalizeResultsParams): Promise<SearchResponse>;
//# sourceMappingURL=search-helpers.d.ts.map