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
import type {
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

/** Resolve current project ID and base_points for instance-aware filtering. */
export async function resolveProjectContext(
  projectId: string | undefined,
  scope: SearchScope,
  projectDetector: ProjectDetector,
  stateManager: SqliteStateManager
): Promise<{ currentProjectId: string | undefined; basePoints: string[] | undefined }> {
  let currentProjectId = projectId;
  if (!currentProjectId && scope === 'project') {
    const projectInfo = await projectDetector.getProjectInfo(process.cwd(), false);
    currentProjectId = projectInfo?.projectId;
  }

  let basePoints: string[] | undefined;
  if (currentProjectId && scope === 'project') {
    const watchFolderId = stateManager.getWatchFolderIdByTenantId(currentProjectId);
    if (watchFolderId) {
      const points = stateManager.getActiveBasePoints(watchFolderId, false);
      if (points.length > 0 && points.length <= 500) basePoints = points;
    }
  }
  return { currentProjectId, basePoints };
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
  collectionsToSearch: string[]
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
    return { fallback: await fallbackSearch(qdrantClient, options, collectionsToSearch) };
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

/** Search all target collections and collect results, tolerating partial failures. */
export async function searchAllCollections(
  qdrantClient: QdrantClient,
  params: SearchAllCollectionsParams
): Promise<{
  allResults: SearchResult[];
  status: 'ok' | 'uncertain';
  statusReason: string | undefined;
}> {
  const {
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
    scoreThreshold,
  } = params;

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
      allResults.push(...(await searchCollection(qdrantClient, searchParams)));
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
}

/** Fuse, sort, expand context, update event, and assemble the final response. */
export async function finalizeResults(
  qdrantClient: QdrantClient,
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  params: FinalizeResultsParams
): Promise<SearchResponse> {
  const {
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
    statusReason,
  } = params;

  const fusedResults = applyRRFFusion(allResults, mode);
  fusedResults.sort((a, b) => b.score - a.score);
  const finalResults = fusedResults.slice(0, limit);

  if (options.expandContext) await expandParentContext(qdrantClient, finalResults);
  if (options.includeGraphContext) await expandGraphContext(daemonClient, finalResults);

  const latencyMs = Date.now() - searchStartMs;
  const topRefs = finalResults.slice(0, 5).map((r) => ({
    id: r.id,
    score: Math.round(r.score * 1000) / 1000,
    collection: r.collection,
  }));
  stateManager.updateSearchEvent(eventId, {
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
