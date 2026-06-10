/**
 * Tests for blendPoolScores — the rerank score-mixing kernel.
 *
 * w=1 must reproduce the legacy pure-reranker order and w=0 the pre-rerank
 * (RRF + path boost) order; intermediate weights interpolate over min-max
 * normalized signals so the two score scales (RRF ~0.01–0.03 vs cross-encoder
 * sigmoid/logit) cannot dominate each other by magnitude alone.
 */

import { describe, it, expect } from 'vitest';
import { blendPoolScores } from '../../src/tools/search-helpers.js';

/** Pool indices sorted by blended score descending. */
function order(blended: Map<number, number>): number[] {
  return [...blended.entries()].sort((a, b) => b[1] - a[1]).map(([index]) => index);
}

describe('blendPoolScores', () => {
  // Pre-rerank pool order is 0, 1, 2; the reranker prefers 1, 2, 0.
  const base = [0.9, 0.6, 0.3];
  const rerank = new Map([
    [0, 0.1],
    [1, 0.95],
    [2, 0.5],
  ]);

  it('w=1 yields the pure reranker order', () => {
    expect(order(blendPoolScores(base, rerank, 1))).toEqual([1, 2, 0]);
  });

  it('w=0 yields the pre-rerank order', () => {
    expect(order(blendPoolScores(base, rerank, 0))).toEqual([0, 1, 2]);
  });

  it('w=0.5 blends both signals over normalized scores', () => {
    // norm(base) = [1, 0.5, 0]; norm(rerank) = [0, 1, ~0.47]
    // blended    = [0.5, 0.75, ~0.24] → order 1, 0, 2
    const blended = blendPoolScores(base, rerank, 0.5);
    expect(order(blended)).toEqual([1, 0, 2]);
    expect(blended.get(0)).toBeCloseTo(0.5);
    expect(blended.get(1)).toBeCloseTo(0.75);
  });

  it('returns only indices the reranker scored', () => {
    const partial = new Map([[1, 0.9]]);
    expect([...blendPoolScores(base, partial, 0.5).keys()]).toEqual([1]);
  });

  it('ignores out-of-range reranker indices', () => {
    const outOfRange = new Map([
      [5, 0.9],
      [1, 0.4],
    ]);
    expect([...blendPoolScores(base, outOfRange, 1).keys()]).toEqual([1]);
  });

  it('cancels degenerate (constant) signals instead of dividing by zero', () => {
    // Flat base → ordering follows the reranker.
    expect(order(blendPoolScores([0.5, 0.5, 0.5], rerank, 0.5))).toEqual([1, 2, 0]);
    // Flat rerank → ordering follows the base.
    const flatRerank = new Map([
      [0, 0.7],
      [1, 0.7],
      [2, 0.7],
    ]);
    const blended = blendPoolScores(base, flatRerank, 0.5);
    expect(order(blended)).toEqual([0, 1, 2]);
    for (const value of blended.values()) {
      expect(Number.isFinite(value)).toBe(true);
    }
  });

  it('clamps weight outside [0, 1]', () => {
    expect(order(blendPoolScores(base, rerank, 5))).toEqual([1, 2, 0]);
    expect(order(blendPoolScores(base, rerank, -1))).toEqual([0, 1, 2]);
  });
});
