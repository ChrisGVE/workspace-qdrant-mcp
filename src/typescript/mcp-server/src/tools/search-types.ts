/**
 * Search tool types, interfaces, and constants.
 */

// Canonical collection names from native bridge (single source of truth)
import { COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_SCRATCHPAD } from '../common/native-bridge.js';
export const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
export const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
export const SCRATCHPAD_COLLECTION = COLLECTION_SCRATCHPAD;

// Vector names for hybrid search
export const DENSE_VECTOR_NAME = 'dense';
export const SPARSE_VECTOR_NAME = 'sparse';

// RRF constant (k=60 is standard)
export const RRF_K = 60;

// Default search parameters
export const DEFAULT_LIMIT = 10;
export const DEFAULT_SCORE_THRESHOLD = 0.3;

// Tag expansion defaults
export const DEFAULT_EXPANSION_WEIGHT = 0.5;
export const DEFAULT_MAX_EXPANDED_KEYWORDS = 10;

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
  includeDeleted: boolean;
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
