/**
 * Search tool types, interfaces, and constants.
 */
export declare const PROJECTS_COLLECTION: string;
export declare const LIBRARIES_COLLECTION: string;
export declare const SCRATCHPAD_COLLECTION: string;
export declare const RULES_COLLECTION: string;
export declare const DENSE_VECTOR_NAME = "dense";
export declare const SPARSE_VECTOR_NAME = "sparse";
export declare const RRF_K = 60;
export declare const DEFAULT_LIMIT = 10;
export declare const DEFAULT_SCORE_THRESHOLD = 0.3;
export declare const DEFAULT_EXPANSION_WEIGHT = 0.5;
export declare const DEFAULT_MAX_EXPANDED_KEYWORDS = 10;
export type SearchMode = 'hybrid' | 'semantic' | 'keyword';
export type SearchScope = 'project' | 'group' | 'all';
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
    libraryPath?: string;
    includeLibraries?: boolean;
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
    /** Enable source diversity re-ranking to surface results from different sources (default: true) */
    diverse?: boolean;
}
export interface Provenance {
    /** Collection the result originates from. */
    source: 'projects' | 'libraries' | 'scratchpad' | 'rules';
    /** Library name for library results. */
    library_name?: string;
    /** Library path prefix for library results. */
    library_path?: string;
    /** Document title extracted from the result payload. */
    doc_title?: string;
    /** Project/tenant ID that owns this result. */
    source_project_id?: string;
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
    provenance?: Provenance;
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
    status?: 'ok' | 'uncertain' | 'error';
    status_reason?: string;
    /** Branch filter applied to this search, or undefined when cross-branch ("*") */
    branch?: string;
    /**
     * Source diversity score for the returned results [0, 1].
     * 1.0 = every result from a unique source.
     * 0.0 = all from one source.
     * Absent when diversity re-ranking is disabled or not applicable.
     */
    diversity_score?: number;
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
    /** For group scope: additional tenant IDs to include in search */
    groupTenantIds: string[] | undefined;
    branch: string | undefined;
    fileType: string | undefined;
    libraryName: string | undefined;
    libraryPath: string | undefined;
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
//# sourceMappingURL=search-types.d.ts.map