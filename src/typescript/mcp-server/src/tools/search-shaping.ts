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
 */

import { FIELD_CONTENT } from '../common/native-bridge.js';
import type {
  ParentContext,
  SearchOptions,
  SearchResponse,
  SearchResult,
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

function shapeAsSummary(r: SearchResult): SearchResult {
  const out: SearchResult = {
    id: r.id,
    score: r.score,
    collection: r.collection,
    content: '',
    metadata: stripTextFromMetadata(r.metadata),
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

/**
 * Apply per-hit payload shaping to a search response based on the
 * caller's options. Returns a new SearchResponse — does not mutate
 * the input.
 */
export function shapeHitPayloads(
  response: SearchResponse,
  options: SearchOptions
): SearchResponse {
  if (options.summary) {
    return { ...response, results: response.results.map(shapeAsSummary) };
  }
  const cap = options.maxBytesPerHit ?? DEFAULT_MAX_BYTES_PER_HIT;
  if (cap <= 0) return response;
  return { ...response, results: response.results.map((r) => shapeAsTruncated(r, cap)) };
}
