/** gRPC message types for EmbeddingService, QueueWriteService, and TrackingWriteService. */

// ── EmbeddingService ──

export interface EmbedTextRequest {
  text: string;
  model?: string;
}

export interface EmbedTextResponse {
  embedding: number[];
  dimensions: number;
  model_name: string;
  success: boolean;
  error_message?: string;
}

export interface SparseVectorRequest {
  text: string;
  /** Lexicon collection whose vocabulary/IDF to use (daemon default: "projects"). */
  collection?: string;
}

export interface SparseVectorResponse {
  indices_values: Record<number, number>;
  vocab_size: number;
  success: boolean;
  error_message?: string;
}

export interface RerankRequest {
  query: string;
  documents: string[];
  top_k?: number;
}

export interface RerankResult {
  index: number;
  score: number;
}

export interface RerankResponse {
  results: RerankResult[];
  success: boolean;
  error_message?: string;
}

// ── QueueWriteService ──

export interface EnqueueItemRequest {
  item_type: string;
  op: string;
  tenant_id: string;
  collection: string;
  payload_json: string;
  branch: string;
  metadata_json?: string;
}

export interface EnqueueItemResponse {
  queue_id: string;
  idempotency_key: string;
  is_new: boolean;
}

export interface RetryAllResponse {
  reset_count: number;
}

export interface RetryItemRequest {
  queue_id: string;
}

export interface RetryItemResponse {
  found: boolean;
  resolved_id: string;
  previous_status: string;
  previous_retry_count: number;
  reset: boolean;
}

// ── TrackingWriteService ──

export interface LogSearchEventRequest {
  id: string;
  session_id?: string;
  project_id?: string;
  actor: string;
  tool: string;
  op: string;
  query_text?: string;
  filters?: string;
  top_k?: number;
  result_count?: number;
  latency_ms?: number;
  top_result_refs?: string;
  outcome?: string;
  parent_event_id?: string;
}

export interface UpdateSearchEventRequest {
  event_id: string;
  result_count: number;
  latency_ms: number;
  top_result_refs?: string;
  outcome?: string;
}

/**
 * Token-economy metrics from the post-execution shaping pass.
 * Spec: docs/specs/20-token-economy-instrumentation.md
 */
export interface UpdateSearchEventEconomyRequest {
  event_id: string;
  bytes_in: number;
  bytes_out: number;
  hits_truncated: number;
  shape_mode: 'truncate' | 'summary' | 'none';
  tool_version?: string;
}

export interface UpsertRuleMirrorRequest {
  rule_id: string;
  rule_text: string;
  scope?: string;
  tenant_id?: string;
  created_at: string;
  updated_at: string;
}

export interface DeleteRuleMirrorRequest {
  rule_id: string;
}

export interface UpsertScratchpadMirrorRequest {
  scratchpad_id: string;
  content: string;
  title?: string;
  tags?: string;
  tenant_id: string;
  created_at: string;
  updated_at: string;
}

export interface DeleteScratchpadMirrorRequest {
  scratchpad_id: string;
}
