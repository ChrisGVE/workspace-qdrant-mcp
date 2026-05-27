/**
 * Tests for source diversity re-ranking (search-diversity.ts).
 */

import { describe, it, expect } from 'vitest';
import {
  diversifyResults,
  extractSource,
  computeDiversityScore,
  DEFAULT_DIVERSITY_CONFIG,
  type DiversityConfig,
} from '../../src/tools/search-diversity.js';
import type { SearchResult } from '../../src/tools/search-types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeResult(
  id: string,
  score: number,
  collection: string,
  overrides: Partial<SearchResult['metadata']> = {}
): SearchResult {
  return {
    id,
    score,
    collection,
    content: `content-${id}`,
    metadata: { ...overrides },
  };
}

function makeProjectResult(id: string, score: number, tenantId: string): SearchResult {
  return makeResult(id, score, 'projects', { tenant_id: tenantId });
}

function makeLibraryResult(id: string, score: number, libraryName: string): SearchResult {
  return makeResult(id, score, 'libraries', { library_name: libraryName });
}

// ---------------------------------------------------------------------------
// extractSource
// ---------------------------------------------------------------------------

describe('extractSource', () => {
  it('uses library_name when present', () => {
    const r = makeLibraryResult('1', 0.9, 'my-lib');
    expect(extractSource(r)).toBe('libraries:my-lib');
  });

  it('uses tenant_id for project results', () => {
    const r = makeProjectResult('1', 0.9, 'tenant-abc');
    expect(extractSource(r)).toBe('projects:tenant-abc');
  });

  it('falls back to unknown when no discriminator present', () => {
    const r = makeResult('1', 0.9, 'projects');
    expect(extractSource(r)).toBe('projects:unknown');
  });
});

// ---------------------------------------------------------------------------
// computeDiversityScore
// ---------------------------------------------------------------------------

describe('computeDiversityScore', () => {
  it('returns 1.0 for empty list', () => {
    expect(computeDiversityScore([])).toBe(1.0);
  });

  it('returns 1.0 when all results from unique sources', () => {
    const results = [
      makeProjectResult('1', 0.9, 'tenant-a'),
      makeProjectResult('2', 0.8, 'tenant-b'),
      makeLibraryResult('3', 0.7, 'lib-x'),
    ];
    expect(computeDiversityScore(results)).toBeCloseTo(1.0);
  });

  it('returns < 1.0 when sources repeat', () => {
    const results = [
      makeProjectResult('1', 0.9, 'tenant-a'),
      makeProjectResult('2', 0.8, 'tenant-a'),
      makeProjectResult('3', 0.7, 'tenant-a'),
    ];
    // 1 unique / 3 total = 0.333
    expect(computeDiversityScore(results)).toBeCloseTo(1 / 3);
  });
});

// ---------------------------------------------------------------------------
// diversifyResults — disabled config
// ---------------------------------------------------------------------------

describe('diversifyResults — disabled', () => {
  it('returns results unchanged when disabled', () => {
    const results = [
      makeProjectResult('1', 0.9, 'tenant-a'),
      makeProjectResult('2', 0.8, 'tenant-a'),
      makeProjectResult('3', 0.7, 'tenant-a'),
      makeProjectResult('4', 0.6, 'tenant-a'),
    ];
    const config: DiversityConfig = { enabled: false, maxPerSource: 3, scoreTierThreshold: 0.05 };
    const { results: out } = diversifyResults(results, config);
    expect(out).toBe(results); // same reference
    expect(out).toHaveLength(4);
  });

  it('returns results unchanged for empty list', () => {
    const { results: out, diversityScore } = diversifyResults([]);
    expect(out).toHaveLength(0);
    expect(diversityScore).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// diversifyResults — maxPerSource cap
// ---------------------------------------------------------------------------

describe('diversifyResults — maxPerSource enforcement', () => {
  it('caps single-source results to maxPerSource', () => {
    // 6 results all from the same source
    const results = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65].map((score, i) =>
      makeProjectResult(`r${i}`, score, 'tenant-a')
    );
    const config: DiversityConfig = { ...DEFAULT_DIVERSITY_CONFIG, maxPerSource: 3 };
    const { results: out } = diversifyResults(results, config);
    expect(out).toHaveLength(3);
    out.forEach((r) => expect(r.metadata.tenant_id).toBe('tenant-a'));
  });

  it('returns max 3 per source from 10 results split 2 sources', () => {
    // 5 from source-a, 5 from source-b, interleaved by score
    const results: SearchResult[] = [];
    for (let i = 0; i < 5; i++) {
      results.push(makeProjectResult(`a${i}`, 0.9 - i * 0.05, 'tenant-a'));
      results.push(makeProjectResult(`b${i}`, 0.87 - i * 0.05, 'tenant-b'));
    }
    // Sort by score descending (as the pipeline does before calling us)
    results.sort((a, b) => b.score - a.score);

    const config: DiversityConfig = { ...DEFAULT_DIVERSITY_CONFIG, maxPerSource: 3 };
    const { results: out } = diversifyResults(results, config);

    const aCount = out.filter((r) => r.metadata.tenant_id === 'tenant-a').length;
    const bCount = out.filter((r) => r.metadata.tenant_id === 'tenant-b').length;

    expect(aCount).toBeLessThanOrEqual(3);
    expect(bCount).toBeLessThanOrEqual(3);
    expect(out.length).toBeLessThanOrEqual(6);
  });
});

// ---------------------------------------------------------------------------
// diversifyResults — diversity score
// ---------------------------------------------------------------------------

describe('diversifyResults — diversity score', () => {
  it('returns diversity score of 1.0 when all sources are unique', () => {
    const results = [
      makeProjectResult('1', 0.9, 'tenant-a'),
      makeLibraryResult('2', 0.85, 'lib-x'),
      makeProjectResult('3', 0.8, 'tenant-b'),
    ];
    const { diversityScore } = diversifyResults(results, DEFAULT_DIVERSITY_CONFIG);
    expect(diversityScore).toBeCloseTo(1.0);
  });

  it('returns diversity score < 1 when multiple results share a source', () => {
    // 2 from tenant-a, 1 from lib-x → after cap still 2 from a, 1 from lib
    const results = [
      makeProjectResult('1', 0.9, 'tenant-a'),
      makeProjectResult('2', 0.85, 'tenant-a'),
      makeLibraryResult('3', 0.8, 'lib-x'),
    ];
    const config: DiversityConfig = { ...DEFAULT_DIVERSITY_CONFIG, maxPerSource: 3 };
    const { results: out, diversityScore } = diversifyResults(results, config);
    // 2 unique / 3 total = 0.666
    expect(out).toHaveLength(3);
    expect(diversityScore).toBeCloseTo(2 / 3);
  });

  it('returns diversity_score of 1.0 for empty results', () => {
    const { diversityScore } = diversifyResults([]);
    expect(diversityScore).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// diversifyResults — tier interleaving
// ---------------------------------------------------------------------------

describe('diversifyResults — tier interleaving', () => {
  it('interleaves two sources within the same score tier', () => {
    // All within 0.05 of each other → single tier
    const results = [
      makeProjectResult('a1', 0.9, 'tenant-a'),
      makeProjectResult('a2', 0.88, 'tenant-a'),
      makeLibraryResult('b1', 0.89, 'lib-x'),
      makeLibraryResult('b2', 0.87, 'lib-x'),
    ];
    results.sort((a, b) => b.score - a.score);

    const config: DiversityConfig = { ...DEFAULT_DIVERSITY_CONFIG, maxPerSource: 3 };
    const { results: out } = diversifyResults(results, config);

    // Results should alternate between sources (a, b, a, b) rather than (a, a, b, b)
    const sources = out.map(extractSource);
    // First two should not be from the same source
    expect(sources[0]).not.toBe(sources[1]);
  });

  it('preserves score ordering across tiers', () => {
    // Two distinct tiers separated by a score gap > 0.05
    const highTier = [
      makeProjectResult('h1', 0.95, 'tenant-a'),
      makeProjectResult('h2', 0.93, 'tenant-a'),
    ];
    const lowTier = [makeLibraryResult('l1', 0.7, 'lib-x'), makeLibraryResult('l2', 0.68, 'lib-x')];
    const results = [...highTier, ...lowTier];

    const config: DiversityConfig = { ...DEFAULT_DIVERSITY_CONFIG, maxPerSource: 3 };
    const { results: out } = diversifyResults(results, config);

    // High-tier results must precede low-tier results
    const highIds = new Set(['h1', 'h2']);
    const lowIds = new Set(['l1', 'l2']);
    let seenLow = false;
    for (const r of out) {
      if (lowIds.has(r.id)) seenLow = true;
      if (seenLow && highIds.has(r.id)) {
        throw new Error(`High-tier result ${r.id} appeared after a low-tier result`);
      }
    }
  });
});
