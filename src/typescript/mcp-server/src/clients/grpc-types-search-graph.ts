/** gRPC types for TextSearchService and GraphService. */

// ── TextSearchService ──

export interface TextSearchRequest {
  pattern: string;
  regex: boolean;
  case_sensitive: boolean;
  tenant_id?: string;
  branch?: string;
  path_glob?: string;
  path_prefix?: string;
  context_lines: number;
  max_results: number;
}

export interface TextSearchResponse {
  matches: TextSearchMatch[];
  total_matches: number;
  truncated: boolean;
  query_time_ms: number;
}

export interface TextSearchCountResponse {
  count: number;
  query_time_ms: number;
}

export interface TextSearchMatch {
  file_path: string;
  line_number: number;
  content: string;
  tenant_id: string;
  branch?: string;
  context_before: string[];
  context_after: string[];
  /**
   * File size in bytes, populated when known. Consumed by the MCP
   * server's grep token-economy metric to compute real `bytes_in`
   * (spec docs/specs/20-token-economy-instrumentation.md §3.2).
   * `undefined` when search.db is at v6 or below — the tool falls
   * back to a per-file proxy in that case.
   */
  file_size?: number;
}

// ── GraphService ──

export interface QueryRelatedRequest {
  tenant_id: string;
  node_id: string;
  max_hops: number;
  edge_types?: string[];
}

export interface QueryRelatedResponse {
  nodes: TraversalNodeProto[];
  total: number;
  query_time_ms: number;
}

export interface TraversalNodeProto {
  node_id: string;
  symbol_name: string;
  symbol_type: string;
  file_path: string;
  edge_type: string;
  depth: number;
  path: string;
}

export interface ImpactAnalysisRequest {
  tenant_id: string;
  symbol_name: string;
  file_path?: string;
}

export interface ImpactAnalysisResponse {
  impacted_nodes: ImpactNodeProto[];
  total_impacted: number;
  query_time_ms: number;
}

export interface ImpactNodeProto {
  node_id: string;
  symbol_name: string;
  file_path: string;
  impact_type: string;
  distance: number;
}

export interface PageRankRequest {
  tenant_id: string;
  damping?: number;
  max_iterations?: number;
  tolerance?: number;
  edge_types?: string[];
  top_k?: number;
}

export interface PageRankNodeProto {
  node_id: string;
  symbol_name: string;
  symbol_type: string;
  file_path: string;
  score: number;
}

export interface PageRankResponse {
  entries: PageRankNodeProto[];
  total: number;
  query_time_ms: number;
}

export interface GraphStatsRequest {
  tenant_id?: string;
}

export interface GraphStatsResponse {
  total_nodes: number;
  total_edges: number;
  nodes_by_type: Record<string, number>;
  edges_by_type: Record<string, number>;
}

export interface CommunityMemberProto {
  node_id: string;
  symbol_name: string;
  symbol_type: string;
  file_path: string;
}

export interface CommunityProto {
  community_id: number;
  members: CommunityMemberProto[];
}

export interface CommunityRequest {
  tenant_id: string;
  max_iterations?: number;
  min_community_size?: number;
  edge_types?: string[];
}

export interface CommunityResponse {
  communities: CommunityProto[];
  total_communities: number;
  query_time_ms: number;
}

export interface BetweennessRequest {
  tenant_id: string;
  edge_types?: string[];
  max_samples?: number;
  top_k?: number;
}

export interface BetweennessNodeProto {
  node_id: string;
  symbol_name: string;
  symbol_type: string;
  file_path: string;
  score: number;
}

export interface BetweennessResponse {
  entries: BetweennessNodeProto[];
  total: number;
  query_time_ms: number;
}
