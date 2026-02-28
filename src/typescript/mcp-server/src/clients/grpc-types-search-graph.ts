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
