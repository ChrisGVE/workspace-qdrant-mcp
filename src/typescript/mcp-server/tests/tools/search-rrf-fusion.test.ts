/**
 * Tests for dense-primary RRF fusion (applyRRFFusion) and the shared file key.
 *
 * Hybrid fusion must keep the dense (semantic) leg as the primary ranking
 * signal: chunks retrieved by BOTH legs sum their votes at full strength
 * (same-chunk agreement is the high-precision signal behind hybrid's top-1/3
 * edge), while chunks only the sparse BM25 leg retrieved are down-weighted to
 * backfill — on natural-language queries the sparse leg is noise-heavy and
 * its unconfirmed votes would otherwise displace semantically-found code
 * from the final top-k.
 */

import { describe, it, expect } from 'vitest';
import { applyRRFFusion, resultFileKey } from '../../src/tools/search-qdrant.js';
import { RRF_K, type SearchResult } from '../../src/tools/search-types.js';

/** Build a chunk-level hit as searchCollection would emit it. Leg rank is
 * positional: callers concatenate legs in Qdrant score order. */
function chunk(id: string, searchType: 'semantic' | 'keyword', score = 0.5): SearchResult {
  return {
    id,
    score,
    collection: 'projects',
    content: `content of ${id}`,
    metadata: { _search_type: searchType },
  };
}

const vote = (rank: number): number => 1 / (RRF_K + rank);

describe('applyRRFFusion (dense-primary)', () => {
  it('sums full-strength votes for chunks confirmed by both legs', () => {
    const results = [
      chunk('A', 'semantic'),
      chunk('B', 'semantic'),
      chunk('B', 'keyword'),
      chunk('C', 'keyword'),
    ];

    const fused = applyRRFFusion(results, 'hybrid');

    const byId = new Map(fused.map((r) => [r.id, r]));
    // B confirmed by both legs: dense rank 2 + sparse rank 1, no demotion.
    expect(byId.get('B')?.score).toBeCloseTo(vote(2) + vote(1), 10);
    // The double vote outranks the dense leg's own #1.
    expect(byId.get('B')!.score).toBeGreaterThan(byId.get('A')!.score);
    expect(fused.every((r) => r.metadata['_search_type'] === 'hybrid')).toBe(true);
  });

  it('keeps dense-only votes at full strength', () => {
    const results = [chunk('A', 'semantic'), chunk('B', 'keyword'), chunk('C', 'keyword')];

    const fused = applyRRFFusion(results, 'hybrid');

    expect(fused.find((r) => r.id === 'A')?.score).toBeCloseTo(vote(1), 10);
  });

  it('down-weights sparse-only chunks below dense-only chunks at equal rank', () => {
    const results = [chunk('A', 'semantic'), chunk('B', 'keyword')];

    const fused = applyRRFFusion(results, 'hybrid');

    const a = fused.find((r) => r.id === 'A');
    const b = fused.find((r) => r.id === 'B');
    // Sparse-only entries carry the demoted vote so unconfirmed keyword
    // matches backfill instead of displacing dense results.
    expect(b?.score).toBeCloseTo(vote(1) * 0.5, 10);
    expect(a!.score).toBeGreaterThan(b!.score);
  });

  it('keeps the best sparse-only vote below the dense pool floor', () => {
    // Dense pool depth at the default limit (10 × 5 overfetch) is 50: the
    // weight must hold the sparse leg's #1 unconfirmed vote under the dense
    // leg's deepest vote, so sparse noise never interleaves into dense ranks.
    const results = [chunk('sparse-top', 'keyword'), chunk('dense-floor', 'semantic')];
    const fused = applyRRFFusion(results, 'hybrid');
    const sparseTop = fused.find((r) => r.id === 'sparse-top');
    expect(sparseTop!.score).toBeLessThan(vote(50));
  });

  it('returns input unchanged for non-hybrid modes and single-leg input', () => {
    const semanticOnly = [chunk('A', 'semantic'), chunk('B', 'semantic')];
    // Non-hybrid mode: fusion does not apply.
    expect(applyRRFFusion(semanticOnly, 'semantic')).toBe(semanticOnly);
    // Hybrid mode but one leg empty: scores stay raw, no fusion.
    expect(applyRRFFusion(semanticOnly, 'hybrid')).toBe(semanticOnly);
  });
});

describe('resultFileKey', () => {
  it('prefers document_id, then paths, then the point id', () => {
    const base = { score: 0.5, collection: 'projects', content: '' };
    expect(
      resultFileKey({
        ...base,
        id: 'p1',
        metadata: { document_id: 'doc-1', relative_path: 'a.rs', file_path: '/x/a.rs' },
      })
    ).toBe('doc-1');
    expect(
      resultFileKey({ ...base, id: 'p1', metadata: { relative_path: 'a.rs', file_path: '/x/a.rs' } })
    ).toBe('a.rs');
    expect(resultFileKey({ ...base, id: 'p1', metadata: { file_path: '/x/a.rs' } })).toBe('/x/a.rs');
    expect(resultFileKey({ ...base, id: 'p1', metadata: {} })).toBe('p1');
  });
});
