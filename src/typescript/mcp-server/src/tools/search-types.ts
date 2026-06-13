/**
 * Search tool types, interfaces, and constants.
 */

// Canonical collection names from native bridge (single source of truth)
import {
  COLLECTION_PROJECTS,
  COLLECTION_LIBRARIES,
  COLLECTION_SCRATCHPAD,
} from '../common/native-bridge.js';
export const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
export const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
export const SCRATCHPAD_COLLECTION = COLLECTION_SCRATCHPAD;

// Vector names for hybrid search
export const DENSE_VECTOR_NAME = 'dense';
export const SPARSE_VECTOR_NAME = 'sparse';

// RRF constant (k=60 is standard)
export const RRF_K = 60;

/** Parse a non-negative finite tuning value from an env var, else the default.
 *  Empty/unset/garbage values fall back — compose `${VAR:-}` passthroughs set
 *  empty strings, which must not zero out a tuning knob. */
export function tuningFromEnv(envVar: string, defaultValue: number): number {
  const raw = process.env[envVar];
  if (raw === undefined || raw.trim() === '') return defaultValue;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : defaultValue;
}

// Default search parameters
export const DEFAULT_LIMIT = 10;
export const DEFAULT_SCORE_THRESHOLD = 0.3;
/** Per-hit text cap (in chars). Default 1500 keeps a 10-hit response well
 *  under typical MCP client per-tool-result token budgets (~25k chars). */
export const DEFAULT_MAX_BYTES_PER_HIT = 1500;

// Tag expansion defaults
export const DEFAULT_EXPANSION_WEIGHT = 0.5;
export const DEFAULT_MAX_EXPANDED_KEYWORDS = 10;

export type SearchMode = 'hybrid' | 'semantic' | 'keyword';
export type SearchScope = 'project' | 'global' | 'all';

export interface SearchOptions {
  query: string;
  /** Telemetry-only override for the `search_events.actor` column (default
   *  "claude"). The benchmark harness sets "benchmark" so eval traffic can be
   *  excluded when mining REAL agent queries from the search history. */
  telemetryActor?: string;
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
  /** Append a small, tenant-filtered scratchpad recall lane to project-scoped
   *  searches so project notes/snippets surface automatically alongside code
   *  (labeled `collection: "scratchpad"`, capped, never displacing code hits).
   *  Default: true for scope="project". Set false to skip. No effect for
   *  global/all scopes or when an explicit `collection` is targeted. */
  includeScratchpad?: boolean;
  tag?: string;
  /** Filter results by multiple concept tags (OR logic) */
  tags?: string[];
  /** When true, fetch parent unit context for each chunk result */
  expandContext?: boolean;
  /** File path glob filter (e.g., "**\/*.rs") — applies in both exact and semantic modes */
  pathGlob?: string;
  /** Filter by project component (e.g., "daemon", "daemon.core"). Supports prefix matching. */
  component?: string;
  /** When true, use FTS5 exact/substring search instead of semantic search */
  exact?: boolean;
  /** Lines of context before/after matches (only for exact mode, default: 0) */
  contextLines?: number;
  /** When true, fetch 1-hop graph context for code symbol results */
  includeGraphContext?: boolean;
  /** Cross-encoder rerank of the top candidates (default: deployment setting,
   *  WQM_SEARCH_RERANK; code default is off). Set false to skip the reranker
   *  (lower latency), or true to enable it for a call. */
  rerank?: boolean;
  /** Blend weight (0–1) for the cross-encoder score when reranking. The final
   *  pool order is `(1-w)·norm(rrf_boosted) + w·norm(rerank)` over min-max
   *  normalized scores. 1 = pure cross-encoder order (legacy replace
   *  behavior); 0 = reranking disabled. Default: WQM_SEARCH_RERANK_WEIGHT
   *  env, else 0.05 (balanced BGE-M3 default after implementation-intent tuning). */
  rerankWeight?: number;
  /** Per-hit text cap (in chars). Content longer than this is truncated
   *  with a marker pointing to retrieve() for the full chunk. Defaults
   *  to {@link DEFAULT_MAX_BYTES_PER_HIT}. Set to 0 to disable truncation. */
  maxBytesPerHit?: number;
  /** When true, drop chunk text bodies entirely and return only
   *  metadata (id, score, collection, title, path/symbol). Intended for
   *  pure discovery before a follow-up retrieve(). Default: false. */
  summary?: boolean;
}

export interface ParentContext {
  parent_unit_id: string;
  unit_type: string;
  unit_text: string;
  locator?: Record<string, unknown>;
}

export interface GraphContextNode {
  symbol: string;
  file_path: string;
  line?: number;
}

export interface GraphContext {
  symbol: string;
  file_path: string;
  callers: GraphContextNode[];
  callees: GraphContextNode[];
}

export interface SearchResult {
  id: string;
  score: number;
  collection: string;
  content: string;
  title?: string;
  metadata: Record<string, unknown>;
  parent_context?: ParentContext;
  graph_context?: GraphContext;
}

/**
 * Per-project indexing-progress block attached to project-scoped search
 * responses while the daemon's queue is still draining.
 *
 * `pending` + `in_progress` + `failed` come from `unified_queue`; `done`
 * is the durable count from `tracked_files`. `percent` is `done / total *
 * 100`, capped at 100.0. We only attach this when `(pending + in_progress)
 * > 0` so a fully indexed project doesn't pay the noise cost.
 */
export interface IndexingProgress {
  pending: number;
  in_progress: number;
  failed: number;
  done: number;
  total: number;
  percent: number;
  /** Estimated seconds until the queue drains for this tenant.
   *  Absent when the daemon doesn't have enough recent activity to
   *  estimate honestly (cold-start) or when the rate is zero with
   *  pending > 0. UIs should render "warming up" in those cases. */
  eta_seconds?: number;
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
  /** Attached only when `scope === 'project'` and the daemon queue
   *  still has work for the current tenant. Absent otherwise. */
  indexing?: IndexingProgress;
}

/**
 * Token-economy metrics emitted by `shapeHitPayloads`.
 *
 * Spec: docs/specs/20-token-economy-instrumentation.md §3.1
 *
 * `bytes_in_shaped` and `bytes_out_shaped` cover only the fields that the
 * shaping pass can see and rewrite — `result.content` and
 * `parent_context.unit_text`. The eventual full `bytes_in` recorded in
 * `search_events` is built on top of these by adding a per-hit file-size
 * probe (out of scope for this initial wiring).
 */
export interface ShapingMetrics {
  /** Sum of bytes in `result.content` + `parent_context.unit_text` BEFORE shaping. */
  bytesInShaped: number;
  /** Sum of bytes in `result.content` + `parent_context.unit_text` AFTER shaping. */
  bytesOutShaped: number;
  /** Number of hits whose body was truncated (0 in summary mode). */
  hitsTruncated: number;
  /** Which shaping mode produced the response. */
  mode: 'truncate' | 'summary' | 'none';
}

export interface SearchToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
  /** Enable tag-based query expansion for BM25 sparse search (default: true) */
  enableTagExpansion?: boolean;
  /** Weight multiplier for expanded keywords (default: 0.5) */
  expansionWeight?: number;
  /** Maximum number of expanded keywords to add (default: 10) */
  maxExpandedKeywords?: number;
}

export interface FilterParams {
  collection: string;
  scope: SearchScope;
  projectId: string | undefined;
  branch: string | undefined;
  fileType: string | undefined;
  libraryName: string | undefined;
  tag: string | undefined;
  tags: string[] | undefined;
  pathGlob: string | undefined;
  /** Filter by component_id in Qdrant payload (prefix matching) */
  component: string | undefined;
  /** Task 15: base_point values for instance-aware filtering */
  basePoints: string[] | undefined;
}

export interface SearchCollectionParams {
  collection: string;
  mode: SearchMode;
  denseEmbedding: number[] | undefined;
  sparseVector: Record<number, number> | undefined;
  filter: Record<string, unknown> | null;
  limit: number;
  scoreThreshold: number;
}
