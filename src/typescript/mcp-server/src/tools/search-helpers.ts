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
  IndexingProgress,
  SearchMode,
  SearchScope,
  SearchOptions,
  SearchResponse,
  SearchResult,
  FilterParams,
  SearchCollectionParams,
} from './search-types.js';
import { PROJECTS_COLLECTION, SCRATCHPAD_COLLECTION, tuningFromEnv } from './search-types.js';
import { buildFilter } from './search-filters.js';
import {
  searchCollection,
  applyRRFFusion,
  resultFileKey,
  expandParentContext,
  fallbackSearch,
} from './search-qdrant.js';
import { expandGraphContext } from './search-graph-context.js';
import { logDebug } from '../utils/logger.js';
import {
  resolveEffectiveBranch,
  resolveProjectIdentity,
} from './branch-scope.js';

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
  currentBranch: string | undefined;
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
  let projectPath: string | undefined;
  if (scope === 'project') {
    const identity = await resolveProjectIdentity(projectDetector, currentProjectId);
    currentProjectId = identity.projectId;
    projectPath =
      identity.projectPath ??
      (currentProjectId ? stateManager.getProjectById(currentProjectId).data?.project_path : undefined);
  }

  const currentBranch = resolveEffectiveBranch({
    explicitBranch: undefined,
    scope,
    projectId: currentProjectId,
    projectPath,
  });

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
  const result: ProjectContextResolution = { currentProjectId, currentBranch, basePoints };
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
    actor?: string | undefined;
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
    actor: opts.actor ?? 'claude',
    tool: 'mcp_qdrant',
    op: 'search',
    queryText: query,
    filters: Object.keys(filters).length > 0 ? JSON.stringify(filters) : undefined,
    topK: limit,
  });
}

/**
 * Pick the collection whose persisted lexicon resolves query-side sparse
 * terms. 'projects' wins when present since that is where hybrid code
 * search matters. Shared by the main-query embedding AND tag expansion
 * (`expandSparseWithTags`) so both vectors resolve term ids against the
 * SAME vocabulary — merging vectors from different lexicons would mix
 * unrelated term ids silently.
 */
export function resolveSparseCollection(collectionsToSearch: string[]): string {
  return collectionsToSearch.includes('projects')
    ? 'projects'
    : (collectionsToSearch[0] ?? 'projects');
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
      // The daemon resolves sparse terms against the per-collection lexicon
      // vocabulary — the same term ids stored in the Qdrant sparse vectors.
      const sparseCollection = resolveSparseCollection(collectionsToSearch);
      const r = await daemonClient.generateSparseVector({
        text: query,
        collection: sparseCollection,
      });
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
  stateManager: SqliteStateManager;
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

const SUPPLEMENTAL_SYMBOL_LIMIT = 8;
const SUPPLEMENTAL_SYMBOL_SCORE = 1.2;

function unique(values: Iterable<string>): string[] {
  return Array.from(new Set(Array.from(values).filter(Boolean)));
}

function normalizeNeedle(value: string): string {
  return value.trim().replace(/[_\s]+/g, '-').replace(/-+/g, '-').toLowerCase();
}

function isDistinctiveIdentifier(token: string): boolean {
  if (token.includes('_')) return true;
  return token.length >= 8 && /[a-z][A-Z]/.test(token);
}

export function extractSupplementalNeedles(query: string): string[] {
  const rawIdentifiers = query.match(/[A-Za-z_$][A-Za-z0-9_$]*(?:[._-][A-Za-z0-9_$]+)*/g) ?? [];
  const needles = new Set<string>();
  for (const token of rawIdentifiers) {
    if (token.length < 4) continue;
    if (isDistinctiveIdentifier(token)) {
      needles.add(token);
      needles.add(normalizeNeedle(token));
    }
  }

  return unique(needles).slice(0, SUPPLEMENTAL_SYMBOL_LIMIT);
}

function qdrantPointId(pointId: string): string {
  if (/^[0-9a-f]{32}$/i.test(pointId)) {
    return `${pointId.slice(0, 8)}-${pointId.slice(8, 12)}-${pointId.slice(12, 16)}-${pointId.slice(16, 20)}-${pointId.slice(20)}`;
  }
  return pointId;
}

async function searchSupplementalSymbolCandidates(
  qdrantClient: QdrantClient,
  coll: string,
  params: SearchAllCollectionsParams
): Promise<SearchResult[]> {
  if (process.env['WQM_SUPPLEMENTAL_SYMBOLS'] === '0') return [];
  if (coll !== PROJECTS_COLLECTION || !params.currentProjectId) return [];
  const needles = extractSupplementalNeedles(params.options.query);
  if (needles.length === 0) return [];
  if (!params.stateManager.isConnected()) return [];
  const watchFolderId = params.stateManager.getWatchFolderIdByTenantId(params.currentProjectId);
  if (!watchFolderId) return [];

  const candidates = params.stateManager.listChunkCandidates({
    watchFolderId,
    needles,
    ...(params.fileType ? { fileType: params.fileType } : {}),
    limit: SUPPLEMENTAL_SYMBOL_LIMIT,
  });
  if (candidates.data.length === 0) return [];

  try {
    const points = await qdrantClient.retrieve(coll, {
      ids: candidates.data.map((candidate) => qdrantPointId(candidate.pointId)),
      with_payload: true,
    });
    return points.map((point, index) => ({
      id: String(point.id),
      score: SUPPLEMENTAL_SYMBOL_SCORE - index * 0.001,
      collection: coll,
      content: (point.payload?.['content'] as string) ?? '',
      metadata: {
        ...point.payload,
        _search_type: 'keyword',
        _supplemental_symbol_match: true,
      },
    }));
  } catch {
    return [];
  }
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
    // Fetch a wider candidate pool than we return: the path-relevance
    // re-rank and same-file dedup both reach past the first `limit` cosine
    // hits, so a precisely-named file ranked ~30th by raw similarity can
    // still be promoted into the top-k.
    limit: params.limit * 5,
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
      allResults.push(...(await searchSupplementalSymbolCandidates(qdrantClient, coll, params)));
    } catch (error) {
      status = 'uncertain';
      statusReason = `Some collections unavailable: ${error instanceof Error ? error.message : 'unknown'}`;
    }
  }
  return { allResults, status, statusReason };
}

/** Max scratchpad notes appended by the project-memory recall lane. Small by
 *  design: the lane is a recall nudge, not a primary result set — it must never
 *  crowd code hits or inflate the response. */
export const SCRATCHPAD_LANE_LIMIT = 3;

/**
 * Project-memory recall lane: a small, tenant-filtered scratchpad query whose
 * hits are appended AFTER the code results so notes never displace code in the
 * ranked top-k. Reuses the embeddings already computed for the main search.
 *
 * Best-effort and self-contained: any failure (including an absent/empty
 * scratchpad collection) degrades to `[]` so it never marks the main search
 * `uncertain` or throws. Tenant scoping comes from `buildFilter` with
 * scope='project' + projectId; results self-label via `SearchResult.collection`.
 */
export async function searchScratchpadLane(
  qdrantClient: QdrantClient,
  params: {
    projectId: string;
    mode: SearchMode;
    denseEmbedding: number[] | undefined;
    sparseVector: Record<number, number> | undefined;
    scoreThreshold: number;
  }
): Promise<SearchResult[]> {
  const filterParams: FilterParams = {
    collection: SCRATCHPAD_COLLECTION,
    scope: 'project',
    projectId: params.projectId,
    branch: undefined,
    fileType: undefined,
    libraryName: undefined,
    tag: undefined,
    tags: undefined,
    pathGlob: undefined,
    component: undefined,
    basePoints: undefined,
  };
  try {
    const hits = await searchCollection(qdrantClient, {
      collection: SCRATCHPAD_COLLECTION,
      mode: params.mode,
      denseEmbedding: params.denseEmbedding,
      sparseVector: params.sparseVector,
      filter: buildFilter(filterParams),
      limit: SCRATCHPAD_LANE_LIMIT,
      scoreThreshold: params.scoreThreshold,
    });
    // searchCollection concatenates the dense + sparse legs WITHOUT fusion, so
    // in hybrid mode the same note arrives twice. Collapse by id (keep the
    // better score) and cap — the lane is a small recall nudge, not a ranked
    // result set, so a coarse score max across legs is sufficient here.
    const byId = new Map<string, SearchResult>();
    for (const h of hits) {
      const existing = byId.get(h.id);
      if (!existing || h.score > existing.score) byId.set(h.id, h);
    }
    return Array.from(byId.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, SCRATCHPAD_LANE_LIMIT);
  } catch (err) {
    logDebug('Scratchpad recall lane failed; omitting', {
      project_id: params.projectId,
      error: err instanceof Error ? err.message : String(err),
    });
    return [];
  }
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
  /** Project-memory recall lane hits (scope="project" only). Appended AFTER
   *  the code top-k, never fused/deduped with it. Empty when the lane is
   *  disabled, no tenant resolved, or it returned nothing. */
  scratchpadHits?: SearchResult[];
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

/** Query words too generic to signal file relevance — dropped before
 * path matching so "how/where/the/search" don't inflate every result. */
const PATH_BOOST_STOPWORDS = new Set([
  'the',
  'a',
  'an',
  'is',
  'are',
  'was',
  'were',
  'how',
  'does',
  'do',
  'did',
  'where',
  'what',
  'which',
  'when',
  'who',
  'why',
  'to',
  'of',
  'in',
  'on',
  'for',
  'and',
  'or',
  'with',
  'from',
  'by',
  'that',
  'this',
  'it',
  'its',
  'as',
  'at',
  'be',
  'can',
  'we',
  'you',
  'i',
  'use',
  'used',
  'using',
  'into',
  'across',
  'ao',
  'aos',
  'com',
  'como',
  'da',
  'das',
  'de',
  'dos',
  'e',
  'em',
  'na',
  'nas',
  'no',
  'nos',
  'o',
  'onde',
  'os',
  'para',
  'por',
  'qual',
  'quais',
  'que',
  'sao',
]);

/** Multiplicative weight for the path-relevance re-rank. A result whose file
 * path/symbol contains ALL query content-words gets +50%; partial overlap
 * scales linearly. Multiplicative so it composes with any score scale
 * (raw cosine ~0.4–0.6 for semantic, RRF ~0.01–0.03 for hybrid). */
// Tunable via WQM_PATH_BOOST_ALPHA. 0.8 was tuned for MiniLM-L6 (384d, raw
// chunk text); with e5-large + path/symbol-enriched dense text the path signal
// is already inside the embedding, so the optimal boost may differ — re-tune
// against the 44-query benchmark after retrieval-side changes.
const PATH_BOOST_ALPHA = tuningFromEnv('WQM_PATH_BOOST_ALPHA', 0.8);

/** Split a string into lowercase alphanumeric tokens (path separators,
 * underscores, dashes, dots, and camelCase humps all break tokens). */
function toTokens(text: string): string[] {
  return text
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

const QUERY_TOKEN_SYNONYMS: Record<string, readonly string[]> = {
  arquivo: ['file'],
  arquivos: ['file'],
  arvore: ['tree'],
  busca: ['search'],
  chave: ['key'],
  chunks: ['chunk'],
  codigo: ['code'],
  colecoes: ['collections'],
  colecao: ['collection'],
  combina: ['combine'],
  combinados: ['combine'],
  densos: ['dense'],
  deteccao: ['detect'],
  detecta: ['detect'],
  dividido: ['split'],
  escopo: ['scope'],
  esparsos: ['sparse'],
  falharam: ['failed'],
  fila: ['queue'],
  filtradas: ['filter'],
  filtrados: ['filter'],
  hibrida: ['hybrid'],
  hibrido: ['hybrid'],
  idempotencia: ['idempotency'],
  ignorar: ['ignore'],
  ignorados: ['ignore'],
  indexacao: ['indexing'],
  itens: ['items'],
  mudanca: ['change'],
  pastas: ['folders'],
  pontos: ['points'],
  projeto: ['project'],
  regras: ['rules'],
  resultados: ['results'],
  semanticos: ['semantic'],
};

/** Content words from the query, used as the path-relevance signal. */
function queryContentTokens(query: string): string[] {
  const tokens = toTokens(query).filter((t) => t.length >= 3 && !PATH_BOOST_STOPWORDS.has(t));
  const expanded: string[] = [];
  for (const token of tokens) {
    expanded.push(token, ...(QUERY_TOKEN_SYNONYMS[token] ?? []));
  }
  return [...new Set(expanded)];
}

/** Fraction of query content-words whose stem appears in the result's file
 * path or symbol name. The path is a high-precision relevance signal a human
 * reader uses first (e.g. "queue throughput metrics" → .../metrics.rs) and is
 * orthogonal to the dense/sparse legs, which over-reward content-term magnets
 * like a 2k-line config file or an adjacent subsystem. */
function pathRelevance(queryTokens: string[], result: SearchResult): number {
  if (queryTokens.length === 0) return 0;
  const path =
    (result.metadata['relative_path'] as string | undefined) ??
    (result.metadata['file_path'] as string | undefined) ??
    '';
  const symbol = (result.metadata['chunk_symbol_name'] as string | undefined) ?? '';
  const pathToks = new Set(toTokens(`${path} ${symbol}`));
  let hits = 0;
  for (const qt of queryTokens) {
    if (pathToks.has(qt)) {
      hits += 1;
      continue;
    }
    // Stem-ish containment so "metrics"~"metric", "resolution"~"resolve".
    for (const pt of pathToks) {
      if (pt.length >= 4 && qt.length >= 4 && (pt.includes(qt) || qt.includes(pt))) {
        hits += 1;
        break;
      }
    }
  }
  return hits / queryTokens.length;
}

// Tunable via WQM_SYMBOL_MATCH_BOOST (multiplier; 1.0 = OFF, the default).
// Hypothesis: when the query NAMES the chunk's own symbol, the definition
// site should outrank files that merely mention or test it ("applyRRFFusion
// implementation" ranked the test file above the defining chunk).
// MEASURED 2026-06-10 (44-query benchmark, e5-1024d index) and REFUTED:
//   1.0 (off): semantic top1 34.1 / top3 56.8 / recall 62.5 / MRR 0.46
//   1.6:                    25.0 / 54.5 / 61.4 / 0.41
//   2.0:                    25.0 / 52.3 / 61.4 / 0.40
// — even with the strict containment rule the boost regressed top-1 by 9pp
// and did not move the sym category it targets (4/6 in every run): when the
// defining chunk misses the candidate pool entirely, no post-hoc multiplier
// can rescue it, and the residual matches it does hit are mostly noise. The
// real lever for definition-site queries is POOL inclusion (e.g. a
// complementary payload-filtered lookup on chunk_symbol_name when the query
// contains an identifier), not rescoring. Knob kept for experiments.
const SYMBOL_MATCH_BOOST = tuningFromEnv('WQM_SYMBOL_MATCH_BOOST', 1.0);

// Small implementation-intent nudge: when the user explicitly asks where code
// lives, prefer source/proto/schema files and demote docs/tests that merely
// mention the same terms. Tunable because path priors are corpus-dependent.
const IMPLEMENTATION_INTENT_CODE_BOOST = tuningFromEnv(
  'WQM_IMPLEMENTATION_INTENT_CODE_BOOST',
  1.12
);
const IMPLEMENTATION_INTENT_NON_CODE_PENALTY = tuningFromEnv(
  'WQM_IMPLEMENTATION_INTENT_NON_CODE_PENALTY',
  0.86
);

const IMPLEMENTATION_INTENT_TERMS = new Set([
  'code',
  'implemented',
  'implementation',
  'implementacao',
  'codigo',
  'defined',
  'definition',
  'function',
  'method',
  'class',
  'command',
  'provider',
  'schema',
  'service',
]);

const CODE_PATH_PREFIX_RE = /^(src|app|lib|packages|crates|cmd|internal|pkg|components)\//;
const CODE_PATH_EXT_RE =
  /\.(rs|ts|tsx|js|jsx|mjs|cjs|py|go|java|kt|swift|c|cc|cpp|h|hpp|cs|rb|php|scala|sh|sql|proto)$/;
const NON_IMPL_PATH_RE =
  /(^|\/)(docs?|test|tests|__tests__|fixtures?|examples?|playwright-report|coverage)\/|\.(md|mdx|rst|txt)$|\.(test|spec)\.[^.]+$/;

/** True when the query names this result's chunk symbol: every token of the
 * symbol appears among the query's content tokens (an identifier in the query
 * tokenizes into exactly its camelCase/snake_case parts, so
 * "applyRRFFusion implementation" includes symbol applyRRFFusion).
 *
 * Precision guards (a first sweep with a lax >=4-char single-token rule
 * REGRESSED semantic top-1 36.4->20.5: every module/function literally named
 * queue, search, config got boosted on ordinary queries):
 * - multi-token symbols: full containment required, high precision since the
 *   query must spell out the compound identifier;
 * - single-token symbols: only distinctive names (>=8 chars, e.g. debouncer)
 *   qualify;
 * - pseudo-symbols (_text, _preamble) never match.
 * Exported for unit tests. */
export function symbolNamedInQuery(queryTokens: ReadonlySet<string>, symbolName: string): boolean {
  if (!symbolName || symbolName.startsWith('_')) return false;
  const symbolTokens = toTokens(symbolName);
  if (symbolTokens.length === 0) return false;
  if (symbolTokens.length === 1) {
    const only = symbolTokens[0] ?? '';
    return only.length >= 8 && queryTokens.has(only);
  }
  return symbolTokens.every((t) => queryTokens.has(t));
}

export function queryHasImplementationIntent(query: string): boolean {
  const lowered = query.toLowerCase();
  if (
    /\bimplementa(?:do|da|cao|ção|r|tion|ed)?\b|\bc[oó]digo\b|\bdefini(?:do|da|ção|cao)\b/.test(
      lowered
    )
  ) {
    return true;
  }
  const tokens = queryContentTokens(query);
  if (!tokens.some((token) => IMPLEMENTATION_INTENT_TERMS.has(token))) return false;
  return /\b(where|which|what|how|onde|qual|como)\b/.test(lowered);
}

function resultPath(result: SearchResult): string {
  return (
    (result.metadata['relative_path'] as string | undefined) ??
    (result.metadata['file_path'] as string | undefined) ??
    ''
  );
}

export function implementationIntentMultiplier(result: SearchResult): number {
  const path = resultPath(result).toLowerCase();
  const fileType = ((result.metadata['file_type'] as string | undefined) ?? '').toLowerCase();
  if (NON_IMPL_PATH_RE.test(path) || fileType === 'docs' || fileType === 'text') {
    return IMPLEMENTATION_INTENT_NON_CODE_PENALTY;
  }
  if (fileType === 'code' || CODE_PATH_PREFIX_RE.test(path) || CODE_PATH_EXT_RE.test(path)) {
    return IMPLEMENTATION_INTENT_CODE_BOOST;
  }
  return 1;
}

/** Re-score results in place by their path relevance to the query. */
function applyPathRelevanceBoost(results: SearchResult[], query: string): void {
  const queryTokens = queryContentTokens(query);
  if (queryTokens.length === 0) return;
  const querySet = new Set(queryTokens);
  const hasImplementationIntent = queryHasImplementationIntent(query);
  for (const result of results) {
    const rel = pathRelevance(queryTokens, result);
    if (rel > 0) result.score *= 1 + PATH_BOOST_ALPHA * rel;
    if (hasImplementationIntent) result.score *= implementationIntentMultiplier(result);
    if (SYMBOL_MATCH_BOOST !== 1) {
      const symbol = (result.metadata['chunk_symbol_name'] as string | undefined) ?? '';
      if (symbolNamedInQuery(querySet, symbol)) result.score *= SYMBOL_MATCH_BOOST;
    }
  }
}

/** Collapse multiple chunks of the same file into a single hit, keeping the
 * highest-scored chunk. Input MUST already be sorted by score descending so
 * the first occurrence per file is the best one.
 *
 * Without this, a large or repetitive file contributes several chunk-level
 * hits that each consume a top-k slot — e.g. one query returned the same
 * `watchdog.rs` four times in the top 6, crowding out other relevant files
 * and tanking recall. Dense and sparse hits from the same file also collide.
 * Identity comes from {@link resultFileKey}: the per-file `document_id`
 * (shared by every chunk of a file, stable across branches) with path/id
 * fallbacks for collections that don't carry it. Exact/FTS5 search has its
 * own line-level path and never reaches this function. */
function dedupeByFile(results: SearchResult[]): SearchResult[] {
  const seen = new Set<string>();
  const deduped: SearchResult[] = [];
  for (const result of results) {
    const key = resultFileKey(result);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(result);
  }
  return deduped;
}

/** How many top deduped candidates to send to the cross-encoder reranker.
 * Cross-encoder scoring is O(pool) CPU work, so we cap the pool — the relevant
 * file is almost always within the top ~30 of the bi-encoder + path-boost order
 * even when it isn't in the top-k. */
const RERANK_POOL = 30;

/** Default blend weight (0–1) for the cross-encoder score when reranking.
 * 1 sorts the pool purely by the reranker (legacy replace behavior); values
 * in between mix the bi-encoder + path-boost order with the reranker signal.
 * 0.05 is the balanced measured default for the BGE-M3 profile after the
 * implementation-intent nudge (2026-06-13, 46-query benchmark): semantic
 * top1 39.1→41.3 and MRR 0.50→0.52 with unchanged recall@10/top10.
 * The e5-large profile previously peaked at 0.25, and 0.10 maximizes BGE-M3
 * top1/MRR at some recall cost, so deployments can still override via
 * WQM_SEARCH_RERANK_WEIGHT. Values
 * >= 0.5 consistently degrade code search; w=1 is the legacy pure-reranker
 * order and should be used only for experiments. Per-call rerankWeight wins
 * over the env default. */
const RERANK_WEIGHT = tuningFromEnv('WQM_SEARCH_RERANK_WEIGHT', 0.05);
const HIGH_RERANK_WEIGHT_WARNING_THRESHOLD = 0.5;
// Cross-encoder latency scales mostly with per-document text length. Measured
// on the 46-query benchmark: 500 chars preserved quality and cut semantic
// rerank P95 from ~102ms to ~63ms; keep env override for larger-context corpora.
const RERANK_DOCUMENT_CHARS = tuningFromEnv('WQM_RERANK_DOCUMENT_CHARS', 500);

/** Blend min-max-normalized base (RRF + path boost) and cross-encoder scores
 * for the rerank pool: `(1-w)·norm(base) + w·norm(rerank)`. `baseScores` is
 * positional (pool order); `rerankScores` maps pool index → cross-encoder
 * score; only indices present there are blended/returned. A degenerate signal
 * (all values equal) normalizes to a constant so it cancels out of the
 * ordering instead of dividing by zero. Exported for unit tests. */
export function blendPoolScores(
  baseScores: readonly number[],
  rerankScores: ReadonlyMap<number, number>,
  weight: number
): Map<number, number> {
  const w = Math.max(0, Math.min(1, weight));
  const indices = [...rerankScores.keys()].filter((i) => i >= 0 && i < baseScores.length);
  const normalizer = (values: readonly number[]): ((v: number) => number) => {
    const min = Math.min(...values);
    const span = Math.max(...values) - min;
    return span > 0 ? (v: number): number => (v - min) / span : (): number => 1;
  };
  const normBase = normalizer(indices.map((i) => baseScores[i] ?? 0));
  const normRerank = normalizer(indices.map((i) => rerankScores.get(i) ?? 0));
  const blended = new Map<number, number>();
  for (const i of indices) {
    blended.set(
      i,
      (1 - w) * normBase(baseScores[i] ?? 0) + w * normRerank(rerankScores.get(i) ?? 0)
    );
  }
  return blended;
}

/** Re-order the top candidates with the daemon's cross-encoder reranker.
 *
 * The bi-encoder (dense) + path-boost order gets a relevant file into the pool
 * but often not the top-k; a cross-encoder scores each (query, chunk) pair
 * jointly — a much stronger relevance signal. Rather than fully replacing the
 * pool order, the final score blends both signals via {@link blendPoolScores}
 * with `weight` ∈ (0,1] (1 = pure cross-encoder order). Best-effort: on any
 * daemon/model error the pre-rerank order stands (the local model also
 * lazy-loads on first call, so the first search pays a one-time warm-up).
 * Input MUST be sorted/deduped; output is the reranked pool followed by any
 * untouched tail. */
async function rerankResults(
  daemonClient: DaemonClient,
  query: string,
  results: SearchResult[],
  limit: number,
  weight: number
): Promise<SearchResult[]> {
  if (results.length <= 1) return results;
  const poolSize = Math.min(results.length, Math.max(limit, RERANK_POOL));
  const pool = results.slice(0, poolSize);
  const documents = pool.map((r) => (r.content ?? '').slice(0, RERANK_DOCUMENT_CHARS));
  try {
    const resp = await daemonClient.rerank({ query, documents });
    if (!resp.success || resp.results.length === 0) return results;
    const rerankScores = new Map<number, number>();
    for (const rr of resp.results) {
      if (rr.index < 0 || rr.index >= pool.length || rerankScores.has(rr.index)) continue;
      rerankScores.set(rr.index, rr.score);
    }
    if (rerankScores.size === 0) return results;
    const blended = blendPoolScores(
      pool.map((r) => r.score),
      rerankScores,
      weight
    );
    const scored: SearchResult[] = [];
    const leftover: SearchResult[] = [];
    pool.forEach((item, index) => {
      const score = blended.get(index);
      if (score === undefined) {
        // Defensive: keep pool items the reranker didn't score so we never
        // drop a candidate.
        leftover.push(item);
        return;
      }
      // Surface the blended score so the final ordering and the reported
      // score agree.
      item.score = score;
      scored.push(item);
    });
    scored.sort((a, b) => b.score - a.score);
    return [...scored, ...leftover, ...results.slice(poolSize)];
  } catch (err) {
    logDebug('Rerank failed; using pre-rerank order', {
      error: err instanceof Error ? err.message : String(err),
    });
    return results;
  }
}

/** Fuse, sort, expand context, update event, and assemble the final response. */
export async function finalizeResults(
  qdrantClient: QdrantClient,
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  params: FinalizeResultsParams
): Promise<SearchResponse> {
  const fusedResults = applyRRFFusion(params.allResults, params.mode);
  // Promote results whose file path/symbol matches the query before ranking
  // — surfaces the precisely-named file over content-term magnets, and shapes
  // which candidates enter the rerank pool below.
  applyPathRelevanceBoost(fusedResults, params.query);
  fusedResults.sort((a, b) => b.score - a.score);
  // Collapse same-file chunks BEFORE reranking/slicing so the pool holds
  // distinct files, not repeated chunks of one file.
  const deduped = dedupeByFile(fusedResults);
  // Cross-encoder rerank is OPT-IN (default OFF). Measured on the 12-query
  // known-item benchmark (settled ext4 index, 2026-06-02): the cross-encoder
  // HURT code search — it scored implementation `.rs`/`.ts` files below
  // prose/docs for "where is X" queries, dropping recall@10 to 38% (vs 46%
  // without) and top10 to 67% (vs 83%) at ~40x the latency (932ms vs 23ms).
  // RE-MEASURED 2026-06-10 after the multilingual-e5-large (1024d) retrieval
  // upgrade, 44-query benchmark: still strictly worse — semantic top-3
  // 59.1%→31.8%, MRR 0.48→0.28 at ~48x latency. Better retrieval did NOT
  // rehabilitate this reranker (jina-turbo is English prose-trained); the
  // only gain was PT top-10 (3/8→5/8), so a MULTILINGUAL cross-encoder
  // (e.g. bge-reranker-v2-m3 on the GPU Infinity backend) is the next thing
  // worth testing — not more tuning of this one.
  // The bi-encoder + path-relevance-boosted `deduped` order is strictly better
  // here. Enable per-call with `rerank: true` for experimentation;
  // WQM_SEARCH_RERANK=1 flips the default ON deployment-wide (per-call
  // `rerank` still wins either way).
  // Both follow-ups from those measurements now exist: the daemon serves a
  // MULTILINGUAL cross-encoder when WQM_RERANK_BASE_URL points at the Infinity
  // GPU sidecar (bge-reranker-v2-m3 — see core embedding::rerank), and the
  // pool order BLENDS the two signals instead of fully reordering:
  // `(1-w)·norm(rrf_boosted) + w·norm(rerank)` with w from `rerankWeight` /
  // WQM_SEARCH_RERANK_WEIGHT (1 = legacy pure-reranker order, 0 = rerank off).
  // MEASURED 2026-06-10 (bge-reranker-v2-m3 on GPU, 44 queries): the full
  // reorder is still strictly worse even multilingual (w=1: semantic top1
  // 31.8→6.8, MRR 0.45→0.21) — but as a weak nudge it beats the baseline on
  // every semantic aggregate: w=0.25 → top3 56.8→59.1, top10 68.2→72.7,
  // recall@10 60.2→62.5, MRR 0.45→0.47 at +24ms avg latency; hybrid top3
  // 50→56.8. RE-TUNED 2026-06-13 for the BGE-M3 dense profile (46 queries):
  // after the implementation-intent nudge, w=0.05 preserved top10/recall and
  // improved top1/MRR; w=0.10 maximized top1/MRR but lost recall. The code
  // default is the balanced 0.05; e5 deployments can still set
  // WQM_SEARCH_RERANK_WEIGHT=0.25 explicitly.
  const rerankDefault = process.env['WQM_SEARCH_RERANK'] === '1';
  const rerankEnabled = params.options.rerank ?? rerankDefault;
  const rerankWeight = Math.min(params.options.rerankWeight ?? RERANK_WEIGHT, 1);
  if (rerankEnabled && rerankWeight >= HIGH_RERANK_WEIGHT_WARNING_THRESHOLD) {
    logDebug('High rerankWeight may degrade code search; use only for experiments', {
      rerankWeight,
      threshold: HIGH_RERANK_WEIGHT_WARNING_THRESHOLD,
      mode: params.mode,
    });
  }
  const ranked =
    rerankEnabled && rerankWeight > 0
      ? await rerankResults(daemonClient, params.query, deduped, params.limit, rerankWeight)
      : deduped;
  const finalResults = ranked.slice(0, params.limit);

  // Context expansion applies to the code results only — scratchpad notes carry
  // no parent unit or graph node, so they are appended afterwards untouched.
  if (params.options.expandContext) await expandParentContext(qdrantClient, finalResults);
  if (params.options.includeGraphContext) await expandGraphContext(daemonClient, finalResults);

  // Append the project-memory recall lane AFTER the code top-k so notes never
  // displace code. They are tenant-filtered + capped upstream and self-label via
  // `collection: "scratchpad"`.
  const scratchpadHits = params.scratchpadHits ?? [];
  const combinedResults =
    scratchpadHits.length > 0 ? [...finalResults, ...scratchpadHits] : finalResults;
  const collectionsSearched =
    scratchpadHits.length > 0 && !params.collectionsToSearch.includes(SCRATCHPAD_COLLECTION)
      ? [...params.collectionsToSearch, SCRATCHPAD_COLLECTION]
      : params.collectionsToSearch;

  const response = recordAndBuildResponse(
    stateManager,
    combinedResults,
    { ...params, collectionsToSearch: collectionsSearched },
    params.searchStartMs
  );
  await attachIndexingProgress(
    response,
    daemonClient,
    params.scope,
    params.currentProjectId ?? params.options.projectId
  );
  return response;
}
