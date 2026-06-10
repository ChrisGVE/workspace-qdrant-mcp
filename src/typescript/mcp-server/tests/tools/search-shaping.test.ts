/**
 * Tests for per-hit payload shaping applied at the outer boundary of the
 * `search` tool. Without this cap, broad queries return responses that
 * exceed MCP client per-tool-result token budgets and trigger disk
 * offload — see issue #N (search payload cap).
 *
 * Also covers token-economy metrics (`ShapingMetrics`) emitted alongside
 * the shaped response. Spec: docs/specs/20-token-economy-instrumentation.md
 */

import { describe, it, expect } from 'vitest';
import { shapeHitPayloads } from '../../src/tools/search-shaping.js';
import {
  DEFAULT_MAX_BYTES_PER_HIT,
  type SearchOptions,
  type SearchResponse,
  type SearchResult,
} from '../../src/tools/search-types.js';

function makeResult(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    id: 'doc-1',
    score: 0.9,
    collection: 'projects',
    content: 'short body',
    metadata: { file_path: 'src/foo.ts' },
    ...overrides,
  };
}

function makeResponse(results: SearchResult[]): SearchResponse {
  return {
    results,
    total: results.length,
    query: 'q',
    mode: 'hybrid',
    scope: 'project',
    collections_searched: ['projects'],
    status: 'ok',
  };
}

function baseOptions(extra: Partial<SearchOptions> = {}): SearchOptions {
  return { query: 'q', ...extra };
}

describe('shapeHitPayloads', () => {
  describe('default truncation (maxBytesPerHit unset → 1500)', () => {
    it('passes through hits whose content is at or below the cap untouched', () => {
      const r = makeResult({ content: 'a'.repeat(100) });
      const { response: shaped } = shapeHitPayloads(makeResponse([r]), baseOptions());
      expect(shaped.results[0].content).toBe('a'.repeat(100));
    });

    it('truncates content longer than the cap with a marker pointing to retrieve()', () => {
      const longText = 'x'.repeat(5000);
      const r = makeResult({ id: 'doc-42', collection: 'projects', content: longText });
      const { response: shaped } = shapeHitPayloads(makeResponse([r]), baseOptions());
      const shapedContent = shaped.results[0].content;
      expect(shapedContent.length).toBeLessThanOrEqual(DEFAULT_MAX_BYTES_PER_HIT);
      expect(shapedContent).toContain('[truncated');
      // The marker must include the docId and collection so the agent
      // can call retrieve() without re-searching.
      expect(shapedContent).toContain('doc-42');
      expect(shapedContent).toContain('projects');
      expect(shapedContent).toContain('retrieve');
    });

    it('keeps the 10-hit response well under 25k chars on broad results', () => {
      // Simulate a worst-case broad search: 10 hits of ~10kB chunk text.
      const hits = Array.from({ length: 10 }, (_, i) =>
        makeResult({ id: `doc-${i}`, content: 'a'.repeat(10_000) })
      );
      const { response: shaped } = shapeHitPayloads(makeResponse(hits), baseOptions());
      const totalChars = shaped.results.reduce((acc, r) => acc + r.content.length, 0);
      // 10 hits × 1500 chars = 15k upper bound — comfortably below the
      // 25k informal budget for an MCP tool result.
      expect(totalChars).toBeLessThanOrEqual(10 * DEFAULT_MAX_BYTES_PER_HIT);
    });

    it('also truncates parent_context.unit_text when present', () => {
      const r = makeResult({
        id: 'doc-99',
        content: 'short',
        parent_context: {
          parent_unit_id: 'parent-1',
          unit_type: 'function',
          unit_text: 'y'.repeat(5000),
        },
      });
      const { response: shaped } = shapeHitPayloads(makeResponse([r]), baseOptions());
      const parentText = shaped.results[0].parent_context?.unit_text ?? '';
      expect(parentText.length).toBeLessThanOrEqual(DEFAULT_MAX_BYTES_PER_HIT);
      expect(parentText).toContain('[truncated');
    });

    it('strips duplicated content from metadata so the body is not shipped twice', () => {
      // The Qdrant payload carries `content` both as result.content and
      // duplicated in metadata. Without dedup, every hit would ship the
      // text twice — defeating the cap.
      const longText = 'z'.repeat(3000);
      const r = makeResult({
        content: longText,
        metadata: { file_path: 'a.ts', content: longText, chunk_text: longText },
      });
      const { response: shaped } = shapeHitPayloads(makeResponse([r]), baseOptions());
      expect(shaped.results[0].metadata).not.toHaveProperty('content');
      expect(shaped.results[0].metadata).not.toHaveProperty('chunk_text');
      // Non-text metadata must be preserved.
      expect(shaped.results[0].metadata).toHaveProperty('file_path', 'a.ts');
    });
  });

  describe('custom maxBytesPerHit', () => {
    it('respects a custom cap', () => {
      const r = makeResult({ content: 'q'.repeat(1000) });
      const { response: shaped } = shapeHitPayloads(
        makeResponse([r]),
        baseOptions({ maxBytesPerHit: 200 })
      );
      expect(shaped.results[0].content.length).toBeLessThanOrEqual(200);
      expect(shaped.results[0].content).toContain('[truncated');
    });

    it('disables truncation when maxBytesPerHit is 0', () => {
      const longText = 'k'.repeat(50_000);
      const r = makeResult({ content: longText });
      const { response: shaped } = shapeHitPayloads(
        makeResponse([r]),
        baseOptions({ maxBytesPerHit: 0 })
      );
      expect(shaped.results[0].content).toBe(longText);
    });

    it('disables truncation when maxBytesPerHit is negative', () => {
      const r = makeResult({ content: 'p'.repeat(10_000) });
      const { response: shaped } = shapeHitPayloads(
        makeResponse([r]),
        baseOptions({ maxBytesPerHit: -1 })
      );
      expect(shaped.results[0].content.length).toBe(10_000);
    });
  });

  describe('summary mode', () => {
    it('drops chunk bodies but keeps id, score, collection, title, and structural metadata', () => {
      const r = makeResult({
        id: 'doc-7',
        score: 0.7,
        title: 'Foo function',
        content: 'a'.repeat(5000),
        metadata: {
          // Real daemon payload keys (schema/qdrant/projects.rs): tree-sitter
          // chunk metadata is `chunk_`-prefixed.
          file_path: 'src/foo.ts',
          chunk_start_line: 10,
          chunk_symbol_name: 'fooFn',
          content: 'a'.repeat(5000),
          chunk_text: 'a'.repeat(5000),
        },
      });
      const { response: shaped } = shapeHitPayloads(makeResponse([r]), baseOptions({ summary: true }));
      expect(shaped.results[0].content).toBe('');
      expect(shaped.results[0].id).toBe('doc-7');
      expect(shaped.results[0].score).toBe(0.7);
      expect(shaped.results[0].collection).toBe('projects');
      expect(shaped.results[0].title).toBe('Foo function');
      // Structural metadata must survive.
      expect(shaped.results[0].metadata).toMatchObject({
        file_path: 'src/foo.ts',
        chunk_start_line: 10,
        chunk_symbol_name: 'fooFn',
      });
      // Text fields must be gone.
      expect(shaped.results[0].metadata).not.toHaveProperty('content');
      expect(shaped.results[0].metadata).not.toHaveProperty('chunk_text');
    });

    it('keeps a 10-hit summary response well under 5k chars', () => {
      const hits = Array.from({ length: 10 }, (_, i) =>
        makeResult({
          id: `doc-${i}`,
          title: `Hit ${i}`,
          content: 'a'.repeat(20_000),
          metadata: { file_path: `src/file${i}.ts`, chunk_start_line: i, content: 'a'.repeat(20_000) },
        })
      );
      const { response: shaped } = shapeHitPayloads(makeResponse(hits), baseOptions({ summary: true }));
      const serialized = JSON.stringify(shaped);
      expect(serialized.length).toBeLessThan(5000);
    });

    it('summary takes precedence over maxBytesPerHit', () => {
      const r = makeResult({ content: 'a'.repeat(3000) });
      const { response: shaped } = shapeHitPayloads(
        makeResponse([r]),
        baseOptions({ summary: true, maxBytesPerHit: 5000 })
      );
      expect(shaped.results[0].content).toBe('');
    });
  });

  describe('immutability', () => {
    it('does not mutate the input response or its hits', () => {
      const originalContent = 'r'.repeat(5000);
      const r = makeResult({ content: originalContent });
      const response = makeResponse([r]);
      shapeHitPayloads(response, baseOptions());
      expect(response.results[0].content).toBe(originalContent);
    });
  });

  describe('shaping metrics (token economy)', () => {
    it('reports mode "none" with bytes_in == bytes_out when cap is disabled', () => {
      const r = makeResult({ content: 'a'.repeat(800) });
      const { metrics } = shapeHitPayloads(
        makeResponse([r]),
        baseOptions({ maxBytesPerHit: 0 })
      );
      expect(metrics.mode).toBe('none');
      expect(metrics.bytesInShaped).toBe(800);
      expect(metrics.bytesOutShaped).toBe(800);
      expect(metrics.hitsTruncated).toBe(0);
    });

    it('reports mode "truncate" with bytes_out < bytes_in when content exceeds the cap', () => {
      const r = makeResult({ content: 'x'.repeat(5000) });
      const { metrics } = shapeHitPayloads(makeResponse([r]), baseOptions());
      expect(metrics.mode).toBe('truncate');
      expect(metrics.bytesInShaped).toBe(5000);
      expect(metrics.bytesOutShaped).toBeLessThanOrEqual(DEFAULT_MAX_BYTES_PER_HIT);
      expect(metrics.bytesOutShaped).toBeLessThan(metrics.bytesInShaped);
      expect(metrics.hitsTruncated).toBe(1);
    });

    it('counts only hits that actually got truncated', () => {
      const small = makeResult({ id: 'small', content: 'a'.repeat(100) });
      const big = makeResult({ id: 'big', content: 'a'.repeat(5000) });
      const { metrics } = shapeHitPayloads(makeResponse([small, big]), baseOptions());
      expect(metrics.hitsTruncated).toBe(1);
      expect(metrics.bytesInShaped).toBe(5100);
    });

    it('counts parent_context.unit_text into bytes_in alongside content', () => {
      const r = makeResult({
        content: 'a'.repeat(500),
        parent_context: {
          parent_unit_id: 'p',
          unit_type: 'function',
          unit_text: 'b'.repeat(300),
        },
      });
      const { metrics } = shapeHitPayloads(makeResponse([r]), baseOptions());
      expect(metrics.bytesInShaped).toBe(800);
    });

    it('marks a hit truncated when parent_context exceeds the cap even if content is small', () => {
      const r = makeResult({
        content: 'a'.repeat(50),
        parent_context: {
          parent_unit_id: 'p',
          unit_type: 'function',
          unit_text: 'b'.repeat(5000),
        },
      });
      const { metrics } = shapeHitPayloads(makeResponse([r]), baseOptions());
      expect(metrics.hitsTruncated).toBe(1);
    });

    it('reports mode "summary" with bytes_out == 0 when summary is requested', () => {
      const r = makeResult({ content: 'a'.repeat(4000) });
      const { metrics } = shapeHitPayloads(
        makeResponse([r]),
        baseOptions({ summary: true })
      );
      expect(metrics.mode).toBe('summary');
      expect(metrics.bytesInShaped).toBe(4000);
      expect(metrics.bytesOutShaped).toBe(0);
      expect(metrics.hitsTruncated).toBe(0);
    });

    it('reports zeros for an empty response', () => {
      const { metrics } = shapeHitPayloads(makeResponse([]), baseOptions());
      expect(metrics.mode).toBe('truncate');
      expect(metrics.bytesInShaped).toBe(0);
      expect(metrics.bytesOutShaped).toBe(0);
      expect(metrics.hitsTruncated).toBe(0);
    });
  });
});
