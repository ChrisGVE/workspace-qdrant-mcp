/**
 * Tests for extractMetadata — the payload→metadata projection used by the
 * `retrieve` tool. It must drop the chunk body, raw vectors, and the daemon's
 * ranking-aid fields (keywords/baskets/tags) while preserving everything an
 * agent actually consumes.
 */

import { describe, it, expect } from 'vitest';

import { extractMetadata } from '../../src/tools/retrieve-types.js';

describe('extractMetadata', () => {
  it('returns an empty object for null/undefined payloads', () => {
    expect(extractMetadata(null)).toEqual({});
    expect(extractMetadata(undefined)).toEqual({});
  });

  it('drops content and raw vector fields', () => {
    const md = extractMetadata({
      content: 'the chunk body',
      dense_vector: [0.1, 0.2],
      sparse_vector: { indices: [1], values: [0.5] },
      file_path: 'src/foo.ts',
    });
    expect(md).not.toHaveProperty('content');
    expect(md).not.toHaveProperty('dense_vector');
    expect(md).not.toHaveProperty('sparse_vector');
    expect(md).toHaveProperty('file_path', 'src/foo.ts');
  });

  it('drops the daemon ranking-aid fields (keywords/baskets/tags)', () => {
    // keyword_extract.rs injects these on every code chunk; they are indexing
    // signal (~1.5–2k tokens/hit) the agent never reads. The retrieve path must
    // strip the same set the search truncate path does.
    const md = extractMetadata({
      file_path: 'src/foo.ts',
      chunk_symbol_name: 'fooFn',
      keywords: Array.from({ length: 50 }, (_, i) => `kw${i}`),
      keyword_baskets: { tagA: ['kw1'], tagB: ['kw2'] },
      concept_tags: ['c1', 'c2'],
      structural_tags: { fn: ['x'] },
    });
    expect(md).not.toHaveProperty('keywords');
    expect(md).not.toHaveProperty('keyword_baskets');
    expect(md).not.toHaveProperty('concept_tags');
    expect(md).not.toHaveProperty('structural_tags');
    // Discovery-relevant metadata survives.
    expect(md).toEqual({ file_path: 'src/foo.ts', chunk_symbol_name: 'fooFn' });
  });
});
