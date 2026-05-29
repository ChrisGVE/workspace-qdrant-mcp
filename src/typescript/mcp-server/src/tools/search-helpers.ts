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
import { getEffectiveCwd } from '../utils/request-context.js';
import type {
  IndexingProgress,
  SearchMode,
  SearchScope,
  SearchOptions,
  SearchResponse,
  SearchResult,
  FilterParams,
  SearchCollectionParams,
} from './search-types.js';
import { PROJECTS_COLLECTION } from './search-types.js';
import { buildFilter } from './search-filters.js';
import {
  searchCollection,
  applyRRFFusion,
  expandParentContext,
  fallbackSearch,
} from './search-qdrant.js';
import { expandGraphContext } from './search-graph-context.js';
import { logDebug } from '../utils/logger.js';

/** Maximum active base_points we still attach as a Qdrant filter. Above
 * this the filter clause would blow past server-side limits; we instead
 * surface a degradation flag so the caller reports it explicitly. */
const BASE_POINTS_FILTER_CAP = 500;

/** Cache TTL for the per-search indexing-progress probe. Short enough that
 *  long-running indexing operations show drain progress within a few hits;
 *  long enough that a burst of searches doesn't fan out to gRPC each time. */
const INDEXING_PROGRESS_CACHE_TTL_MS = 1500;

interface IndexingProgressCacheEntry {
  value: IndexingProgress | null;
  fetchedAt: number;
}

/** Module-level cache keyed by tenant_id. The MCP server is long-lived;
 *  cache survives across tool invocations. Never grows unbounded — one
 *  entry per active project, and projects are O(few). */
const indexingProgressCache = new Map<string, IndexingProgressCacheEntry>();

/**
 * Fetch indexing progress for `projectId`, with a short in-memory cache.
 *
 * Returns `null` when the daemon call fails or the project is unknown —
 * callers should treat that as "no signal" and omit the `indexing` block
 * rather than guess. When the queue is fully drained, the daemon reports
 * `pending = in_progress = failed = 0` and we still return the block so
 * callers can render "100% indexed" without an ambiguous absence.
 */
export async function fetchIndexingProgress(
  daemonClient: DaemonClient,
  projectId: string
): Promise<IndexingProgress | null> {
  const now = Date.now();
  const cached = indexingProgressCache.get(projectId);
  if (cached && now - cached.fetchedAt < INDEXING_PROGRESS_CACHE_TTL_MS) {
    return cached.value;
  }
  try {
    const status = await daemonClient.getProjectStatus({ project_id: projectId });
    if (!status.found) {
      indexingProgressCache.set(projectId, { value: null, fetchedAt: now });
      return null;
    }
    const value: IndexingProgress = {
      pending: status.pending_count ?? 0,
      in_progress: status.in_progress_count ?? 0,
      failed: status.failed_count ?? 0,
      done: status.done_count ?? 0,
      total: status.total_count ?? 0,
      percent: status.percent_complete ?? 100,
    };
    // ETA is optional — only attach when the daemon actually provided
    // one. Skipping it preserves the "warming up" semantic for callers.
    if (typeof status.eta_seconds === 'number') {
      value.eta_seconds = status.eta_seconds;
    }
    indexingProgressCache.set(projectId, { value, fetchedAt: now });
    return value;
  } catch (err) {
    logDebug('Indexing progress probe failed; omitting block', {
      project_id: projectId,
      error: err instanceof Error ? err.message : String(err),
    });
    indexingProgressCache.set(projectId, { value: null, fetchedAt: now });
    return null;
  }
}

/** Attach the indexing block to a SearchResponse when scope=project and we
 *  have a tenant. No-op otherwise. Safe to call on every exit path. */
export async function attachIndexingProgress(
  response: SearchResponse,
  daemonClient: DaemonClient,
  scope: SearchScope,
  projectId: string | undefined
): Promise<void> {
  if (scope !== 'project' || !projectId) return;
  const progress = await fetchIndexingProgress(daemonClient, projectId);
  if (progress) response.indexing = progress;
}

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

/** Resolve current project ID and base_points for instance-aware filtering. */
export async function resolveProjectContext(
  projectId: string | undefined,
  scope: SearchScope,
  projectDetector: ProjectDetector,
  stateManager: SqliteStateManager
): Promise<ProjectContextResolution> {
  let currentProjectId = projectId;
  if (!currentProjectId && scope === 'project') {
    const projectInfo = await projectDetector.getProjectInfo(getEffectiveCwd(), false, {
      fallbackToSoleProject: true,
    });
    currentProjectId = projectInfo?.projectId;
  }

  let basePoints: string[] | undefined;
  let basePointsDegraded = false;
  let basePointsActiveCount: number | undefined;
  if (currentProjectId && scope === 'project') {
    const watchFolderId = stateManager.getWatchFolderIdByTenantId(currentProjectId);
    if (watchFolderId) {
      // base_point narrowing only disambiguates *instances* — i.e. multiple
      // clones/worktrees of the same project sharing one tenant_id. With a
      // single watch folder the tenant filter alone isolates results, so
      // enumerating one base_point per tracked file (which never scales past
      // the cap on a real repo) buys nothing and would falsely report
      // `status: uncertain` on every project search. Only pay the cost — and
      // only flag degradation — when there genuinely are 2+ clones.
      const cloneCount = stateManager.countWatchFoldersByTenantId(currentProjectId);
      if (cloneCount > 1) {
        const points = stateManager.getActiveBasePoints(watchFolderId, false);
        if (points.length > 0 && points.length <= BASE_POINTS_FILTER_CAP) {
          basePoints = points;
        } else if (points.length > BASE_POINTS_FILTER_CAP) {
          // Genuine multi-clone ambiguity that exceeds the filter cap: we
          // cannot enumerate every base_point, so instance isolation is
          // unavailable. Surface it so the caller reports it explicitly.
          basePointsDegraded = true;
          basePointsActiveCount = points.length;
        }
      }
    }
  }
  const result: ProjectContextResolution = { currentProjectId, basePoints };
  if (basePointsDegraded) {
    result.basePointsDegraded = true;
    if (basePointsActiveCount !== undefined) {
      result.basePointsActiveCount = basePointsActiveCount;
    }
  }
  return result;
}

/** Log the pre-execution search event. */
export function logSearchEventPre(
  stateManager: SqliteStateManager,
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
  stateManager.logSearchEvent({
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

/** Generate dense and sparse embeddings. Returns `{ fallback }` on daemon error. */
export async function generateEmbeddings(
  daemonClient: DaemonClient,
  qdrantClient: QdrantClient,
  query: string,
  mode: SearchMode,
  options: SearchOptions,
  collectionsToSearch: string[],
  fallbackContext: { currentProjectId: string | undefined; basePoints: string[] | undefined }
): Promise<
  | { denseEmbedding: number[] | undefined; sparseVector: Record<number, number> | undefined }
  | { fallback: SearchResponse }
> {
  let denseEmbedding: number[] | undefined;
  let sparseVector: Record<number, number> | undefined;
  try {
    if (mode === 'hybrid' || mode === 'semantic') {
      const r = await daemonClient.embedText({ text: query });
      if (r.success) denseEmbedding = r.embedding;
    }
    if (mode === 'hybrid' || mode === 'keyword') {
      const r = await daemonClient.generateSparseVector({ text: query });
      if (r.success) sparseVector = r.indices_values;
    }
  } catch {
    return {
      fallback: await fallbackSearch(qdrantClient, options, collectionsToSearch, fallbackContext),
    };
  }
  return { denseEmbedding, sparseVector };
}

export interface SearchAllCollectionsParams {
  collectionsToSearch: string[];
  scope: SearchScope;
  currentProjectId: string | undefined;
  basePoints: string[] | undefined;
  branch: string | undefined;
  fileType: string | undefined;
  libraryName: string | undefined;
  tag: string | undefined;
  tags: string[] | undefined;
  options: SearchOptions;
  mode: SearchMode;
  denseEmbedding: number[] | undefined;
  sparseVector: Record<number, number> | undefined;
  limit: number;
  scoreThreshold: number;
}

/** Build SearchCollectionParams for one collection in a fan-out search. */
function buildCollectionSearchParams(
  coll: string,
  params: SearchAllCollectionsParams
): SearchCollectionParams {
  const filterParams: FilterParams = {
    collection: coll,
    scope: params.scope,
    projectId: params.currentProjectId,
    branch: params.branch,
    fileType: params.fileType,
    libraryName: params.libraryName,
    tag: params.tag,
    tags: params.tags,
    pathGlob: params.options.pathGlob,
    component: params.options.component,
    basePoints: coll === PROJECTS_COLLECTION ? params.basePoints : undefined,
  };
  return {
    collection: coll,
    mode: params.mode,
    denseEmbedding: params.denseEmbedding,
    sparseVector: params.sparseVector,
    filter: buildFilter(filterParams),
    limit: params.limit * 2,
    scoreThreshold: params.scoreThreshold,
  };
}

/** Search all target collections and collect results, tolerating partial failures. */
export async function searchAllCollections(
  qdrantClient: QdrantClient,
  params: SearchAllCollectionsParams
): Promise<{
  allResults: SearchResult[];
  status: 'ok' | 'uncertain';
  statusReason: string | undefined;
}> {
  const allResults: SearchResult[] = [];
  let status: 'ok' | 'uncertain' = 'ok';
  let statusReason: string | undefined;

  for (const coll of params.collectionsToSearch) {
    try {
      allResults.push(
        ...(await searchCollection(qdrantClient, buildCollectionSearchParams(coll, params)))
      );
    } catch (error) {
      status = 'uncertain';
      statusReason = `Some collections unavailable: ${error instanceof Error ? error.message : 'unknown'}`;
    }
  }
  return { allResults, status, statusReason };
}

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
  /** Resolved tenant_id for the current search. Drives the optional
   *  per-response `indexing` block when scope === 'project'. */
  currentProjectId: string | undefined;
}

/** Record search telemetry and build the final SearchResponse. */
function recordAndBuildResponse(
  stateManager: SqliteStateManager,
  finalResults: SearchResult[],
  params: FinalizeResultsParams,
  searchStartMs: number
): SearchResponse {
  const latencyMs = Date.now() - searchStartMs;
  const topRefs = finalResults
    .slice(0, 5)
    .map((r) => ({ id: r.id, score: Math.round(r.score * 1000) / 1000, collection: r.collection }));
  stateManager.updateSearchEvent(params.eventId, {
    resultCount: finalResults.length,
    latencyMs,
    topResultRefs: JSON.stringify(topRefs),
  });
  const response: SearchResponse = {
    results: finalResults,
    total: finalResults.length,
    query: params.query,
    mode: params.mode,
    scope: params.scope,
    collections_searched: params.collectionsToSearch,
    status: params.status,
  };
  if (params.statusReason) response.status_reason = params.statusReason;
  return response;
}

/** Fuse, sort, expand context, update event, and assemble the final response. */
export async function finalizeResults(
  qdrantClient: QdrantClient,
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  params: FinalizeResultsParams
): Promise<SearchResponse> {
  const fusedResults = applyRRFFusion(params.allResults, params.mode);
  fusedResults.sort((a, b) => b.score - a.score);
  const finalResults = fusedResults.slice(0, params.limit);

  if (params.options.expandContext) await expandParentContext(qdrantClient, finalResults);
  if (params.options.includeGraphContext) await expandGraphContext(daemonClient, finalResults);

  const response = recordAndBuildResponse(stateManager, finalResults, params, params.searchStartMs);
  await attachIndexingProgress(
    response,
    daemonClient,
    params.scope,
    params.currentProjectId ?? params.options.projectId
  );
  return response;
}
