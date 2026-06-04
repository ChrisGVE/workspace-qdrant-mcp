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
