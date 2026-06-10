/**
 * Per-hit payload shaping for the `search` tool.
 *
 * Search hits can carry large chunk bodies. Without a cap, a 10-hit
 * response easily exceeds an MCP client's per-tool-result token budget
 * and triggers disk offload at the client side, which breaks the agent's
 * reasoning flow. This module trims hit bodies before serialization so
 * callers don't have to think about budgets at all.
 *
 * Two modes:
 *  - default: truncate each hit's `content` (and `parent_context.unit_text`)
 *    at {@link DEFAULT_MAX_BYTES_PER_HIT} chars; append a marker that
 *    points the agent at retrieve() for the full chunk.
 *  - `summary: true`: drop text bodies entirely; keep only id/score/
 *    collection/title and structural metadata. Intended for "which doc
 *    do I want?" discovery before a follow-up retrieve() call.
 *
 * The function also emits a {@link ShapingMetrics} sidecar so callers can
 * record token-economy stats per spec
 * `docs/specs/20-token-economy-instrumentation.md`.
 */

import { FIELD_CONTENT } from '../common/native-bridge.js';
import type {
  ParentContext,
  SearchOptions,
  SearchResponse,
  SearchResult,
  ShapingMetrics,
} from './search-types.js';
import { DEFAULT_MAX_BYTES_PER_HIT } from './search-types.js';

/** Metadata payload fields known to carry chunk text. Stripped in
 *  summary mode AND deduplicated against `result.content` in truncate
 *  mode (the daemon's payload already duplicates content into both
 *  `result.content` and `result.metadata[FIELD_CONTENT]`). */
const TEXT_BODY_KEYS: readonly string[] = [
  'content',
  'text',
  'chunk_text',
  'unit_text',
  'snippet',
  'body',
];

function truncateText(text: string, cap: number, id: string, collection: string): string {
  if (text.length <= cap) return text;
  const marker = ` ... [truncated at ${cap} chars; full chunk via retrieve(documentId="${id}", collection="${collection}")]`;
  const keep = Math.max(0, cap - marker.length);
  return text.slice(0, keep) + marker;
}

function stripTextFromMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...metadata };
  for (const key of TEXT_BODY_KEYS) {
    if (key in out) delete out[key];
  }
  if (FIELD_CONTENT && FIELD_CONTENT in out) delete out[FIELD_CONTENT];
  return out;
}

/** Metadata fields worth keeping in `summary` mode — just enough to decide
 *  "which result do I want?" before a follow-up retrieve(). Everything else
 *  (ranking aids like `keyword_baskets`/`keywords`, sparse-vector debris, and
 *  other large payload fields) is dropped: summary exists to economize tokens,
 *  and the verbose metadata was ~1–2k tokens of noise per hit.
 *
 *  Key names follow the daemon's Qdrant payload schema
 *  (src/rust/common/src/schema/qdrant/projects.rs): tree-sitter chunk
 *  metadata is prefixed `chunk_` — `chunk_symbol_name`, `chunk_start_line`,
 *  `chunk_end_line`, `chunk_chunk_type`. Unprefixed spellings never existed
 *  in the payload and silently matched nothing. */
const SUMMARY_METADATA_KEYS: readonly string[] = [
  'file_path',
  'relative_path',
  'language',
  'branch',
  'document_id',
  'chunk_symbol_name',
  'chunk_start_line',
  'chunk_end_line',
  'chunk_chunk_type',
];

function pickSummaryMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of SUMMARY_METADATA_KEYS) {
    const v = metadata[key];
    if (v !== undefined && v !== null && v !== '') out[key] = v;
  }
  return out;
}

function shapeAsSummary(r: SearchResult): SearchResult {
  const out: SearchResult = {
    id: r.id,
    score: r.score,
    collection: r.collection,
    content: '',
    // Allowlist, not just text-stripping: summary keeps only discovery-relevant
    // structural fields and drops the rest (keyword_baskets/keywords/etc.).
    metadata: pickSummaryMetadata(r.metadata),
  };
  if (r.title) out.title = r.title;
  return out;
}

function shapeParentContext(
  parent: ParentContext,
  cap: number,
  id: string,
  collection: string
): ParentContext {
  return {
    ...parent,
    unit_text: truncateText(parent.unit_text ?? '', cap, id, collection),
  };
}

function shapeAsTruncated(r: SearchResult, cap: number): SearchResult {
  const out: SearchResult = {
    ...r,
    content: truncateText(r.content ?? '', cap, r.id, r.collection),
    // Drop content duplication in metadata to keep total payload close
    // to the cap — without this, we'd ship the full text twice for any
    // hit whose body is under the cap.
    metadata: stripTextFromMetadata(r.metadata),
  };
  if (r.parent_context) {
    out.parent_context = shapeParentContext(r.parent_context, cap, r.id, r.collection);
  }
  return out;
}

function hitShapedBytes(r: SearchResult): number {
  return (r.content?.length ?? 0) + (r.parent_context?.unit_text?.length ?? 0);
}

function emptyMetrics(mode: ShapingMetrics['mode']): ShapingMetrics {
  return { bytesInShaped: 0, bytesOutShaped: 0, hitsTruncated: 0, mode };
}

/**
 * Apply per-hit payload shaping to a search response based on the
 * caller's options. Returns a new SearchResponse (input is not mutated)
 * along with shaping metrics for instrumentation.
 */
export function shapeHitPayloads(
  response: SearchResponse,
  options: SearchOptions
): { response: SearchResponse; metrics: ShapingMetrics } {
  if (options.summary) {
    const metrics = emptyMetrics('summary');
    const results = response.results.map((r) => {
      metrics.bytesInShaped += hitShapedBytes(r);
      return shapeAsSummary(r);
    });
    // bytesOutShaped stays 0 — summary mode drops bodies entirely.
    return { response: { ...response, results }, metrics };
  }
  const cap = options.maxBytesPerHit ?? DEFAULT_MAX_BYTES_PER_HIT;
  if (cap <= 0) {
    const metrics = emptyMetrics('none');
    for (const r of response.results) {
      const bytes = hitShapedBytes(r);
      metrics.bytesInShaped += bytes;
      metrics.bytesOutShaped += bytes;
    }
    return { response, metrics };
  }
  const metrics = emptyMetrics('truncate');
  const results = response.results.map((r) => {
    const beforeContent = r.content?.length ?? 0;
    const beforeParent = r.parent_context?.unit_text?.length ?? 0;
    metrics.bytesInShaped += beforeContent + beforeParent;
    if (beforeContent > cap || beforeParent > cap) {
      metrics.hitsTruncated += 1;
    }
    const shaped = shapeAsTruncated(r, cap);
    metrics.bytesOutShaped += hitShapedBytes(shaped);
    return shaped;
  });
  return { response: { ...response, results }, metrics };
}
