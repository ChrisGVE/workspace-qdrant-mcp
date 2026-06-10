/**
 * Tests for buildSearchOptions — the raw-args → SearchOptions mapper.
 *
 * Regression guard: the builder whitelists fields explicitly, so a newly added
 * tool-schema field is silently dropped unless it is wired here too (which is
 * exactly how includeScratchpad initially failed to disable the recall lane).
 */

import { describe, it, expect } from 'vitest';
import { buildSearchOptions } from '../src/tool-builders/search.js';

describe('buildSearchOptions — includeScratchpad', () => {
  it('maps includeScratchpad: false through to options', () => {
    const opts = buildSearchOptions({ query: 'x', includeScratchpad: false });
    expect(opts.includeScratchpad).toBe(false);
  });

  it('maps includeScratchpad: true through to options', () => {
    const opts = buildSearchOptions({ query: 'x', includeScratchpad: true });
    expect(opts.includeScratchpad).toBe(true);
  });

  it('leaves includeScratchpad undefined when omitted (lane default-on)', () => {
    const opts = buildSearchOptions({ query: 'x' });
    expect(opts.includeScratchpad).toBeUndefined();
  });
});

describe('buildSearchOptions — previously-dropped output options', () => {
  it('maps summary through to options', () => {
    expect(buildSearchOptions({ query: 'x', summary: true }).summary).toBe(true);
    expect(buildSearchOptions({ query: 'x' }).summary).toBeUndefined();
  });

  it('maps maxBytesPerHit through (including 0, which disables truncation)', () => {
    expect(buildSearchOptions({ query: 'x', maxBytesPerHit: 0 }).maxBytesPerHit).toBe(0);
    expect(buildSearchOptions({ query: 'x', maxBytesPerHit: 500 }).maxBytesPerHit).toBe(500);
  });

  it('maps expandContext through to options', () => {
    expect(buildSearchOptions({ query: 'x', expandContext: true }).expandContext).toBe(true);
  });

  it('maps rerank through to options', () => {
    expect(buildSearchOptions({ query: 'x', rerank: true }).rerank).toBe(true);
  });

  it('maps rerankWeight through (including 0, which disables reranking)', () => {
    expect(buildSearchOptions({ query: 'x', rerankWeight: 0 }).rerankWeight).toBe(0);
    expect(buildSearchOptions({ query: 'x', rerankWeight: 0.5 }).rerankWeight).toBe(0.5);
    expect(buildSearchOptions({ query: 'x' }).rerankWeight).toBeUndefined();
  });
});
