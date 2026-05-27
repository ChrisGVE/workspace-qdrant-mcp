import { describe, expect, it } from 'vitest';

import {
  classifySemanticSearchQuality,
  evaluateSearchResults,
  globToRegExp,
  matchesGlob,
  normalizeBenchmarkPath,
  summarizeModeRuns,
  type SemanticSearchModeRun,
} from '../../src/benchmarks/semantic-search.js';
import type { SearchResult } from '../../src/tools/search.js';

function createResult(
  id: string,
  filePath: string,
  score: number,
  useRelativePath = true
): SearchResult {
  return {
    id,
    score,
    collection: 'projects',
    content: '',
    metadata: useRelativePath
      ? { relative_path: filePath }
      : { file_path: filePath },
  };
}

describe('semantic-search benchmark helpers', () => {
  it('normalizes absolute and relative paths against the workspace root', () => {
    expect(
      normalizeBenchmarkPath(
        'C:\\Users\\alber\\Documents\\repositorios\\workspace-qdrant-mcp\\src\\tools\\search.ts',
        'C:\\Users\\alber\\Documents\\repositorios\\workspace-qdrant-mcp'
      )
    ).toBe('src/tools/search.ts');
    expect(normalizeBenchmarkPath('./docs/plans/search.md', 'C:\\repo')).toBe('docs/plans/search.md');
  });

  it('converts glob patterns to regex and matches normalized paths', () => {
    const regex = globToRegExp('src/**/*.ts');
    expect(regex.test('src/typescript/mcp-server/src/tools/search.ts')).toBe(true);
    expect(matchesGlob('docs/plans/2026-05-25-search-quality-next-steps.md', 'docs/**/*.md')).toBe(true);
    expect(matchesGlob('src/typescript/mcp-server/src/tools/search.ts', 'docs/**/*.md')).toBe(false);
  });

  it('evaluates search results with top-k hit, duplicate, precision, and recall metrics', () => {
    const results: SearchResult[] = [
      createResult(
        '1',
        'src/typescript/mcp-server/src/tools/search.ts',
        0.99
      ),
      createResult(
        '2',
        'C:\\Users\\alber\\Documents\\repositorios\\workspace-qdrant-mcp\\docs\\plans\\2026-05-25-search-quality-next-steps.md',
        0.95,
        false
      ),
      createResult(
        '3',
        'src/typescript/mcp-server/src/tools/search.ts',
        0.9
      ),
      createResult(
        '4',
        'src/typescript/mcp-server/src/tools/search-qdrant.ts',
        0.85
      ),
    ];

    const evaluation = evaluateSearchResults(
      results,
      [
        'src/typescript/mcp-server/src/tools/search.ts',
        'docs/plans/2026-05-25-search-quality-next-steps.md',
      ],
      'C:\\Users\\alber\\Documents\\repositorios\\workspace-qdrant-mcp',
      10
    );

    expect(evaluation.top1Hit).toBe(true);
    expect(evaluation.top3Hit).toBe(true);
    expect(evaluation.top10Hit).toBe(true);
    expect(evaluation.firstRelevantRank).toBe(1);
    expect(evaluation.mrr).toBe(1);
    expect(evaluation.precisionAt10).toBeCloseTo(2 / 3, 5);
    expect(evaluation.recallAt10).toBe(1);
    expect(evaluation.duplicateRate).toBeCloseTo(0.25, 5);
    expect(evaluation.matchedExpectedFiles).toEqual([
      'src/typescript/mcp-server/src/tools/search.ts',
      'docs/plans/2026-05-25-search-quality-next-steps.md',
    ]);
    expect(evaluation.missingExpectedFiles).toEqual([]);
    expect(evaluation.topPaths).toEqual([
      'src/typescript/mcp-server/src/tools/search.ts',
      'docs/plans/2026-05-25-search-quality-next-steps.md',
      'src/typescript/mcp-server/src/tools/search-qdrant.ts',
    ]);
  });

  it('summarizes runs and classifies semantic quality against the documented thresholds', () => {
    const runs: SemanticSearchModeRun[] = [
      {
        mode: 'semantic',
        searchOptions: { query: 'alpha', limit: 10, scope: 'project', mode: 'semantic' },
        status: 'ok',
        total: 2,
        collectionsSearched: ['projects'],
        latencySamplesMs: [10, 20],
        latencyMeanMs: 15,
        latencyMedianMs: 15,
        latencyP95Ms: 19.5,
        topResults: [],
        evaluation: {
          expectedFiles: ['a.ts'],
          matchedExpectedFiles: ['a.ts'],
          missingExpectedFiles: [],
          rawTopPaths: ['a.ts'],
          topPaths: ['a.ts'],
          top1Hit: true,
          top3Hit: true,
          top10Hit: true,
          firstRelevantRank: 1,
          precisionAt10: 1,
          recallAt10: 1,
          duplicateRate: 0,
          mrr: 1,
        },
      },
      {
        mode: 'semantic',
        searchOptions: { query: 'beta', limit: 10, scope: 'project', mode: 'semantic' },
        status: 'ok',
        total: 2,
        collectionsSearched: ['projects'],
        latencySamplesMs: [30],
        latencyMeanMs: 30,
        latencyMedianMs: 30,
        latencyP95Ms: 30,
        topResults: [],
        evaluation: {
          expectedFiles: ['b.ts'],
          matchedExpectedFiles: ['b.ts'],
          missingExpectedFiles: [],
          rawTopPaths: ['x.ts', 'b.ts'],
          topPaths: ['x.ts', 'b.ts'],
          top1Hit: false,
          top3Hit: true,
          top10Hit: true,
          firstRelevantRank: 2,
          precisionAt10: 0.9,
          recallAt10: 1,
          duplicateRate: 0,
          mrr: 0.5,
        },
      },
    ];

    const summary = summarizeModeRuns(runs);
    expect(summary.runs).toBe(2);
    expect(summary.top1HitRate).toBe(0.5);
    expect(summary.top3HitRate).toBe(1);
    expect(summary.top10HitRate).toBe(1);
    expect(summary.avgLatencyMs).toBe(20);
    expect(summary.p95LatencyMs).toBe(29);

    const verdict = classifySemanticSearchQuality(summary);
    expect(verdict.grade).toBe('good');
    expect(verdict.reasons).toEqual([]);
  });
});
