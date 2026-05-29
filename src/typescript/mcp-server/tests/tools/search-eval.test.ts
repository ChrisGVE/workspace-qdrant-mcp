/**
 * Tests for the `search_eval` tool handler — it shapes the benchmark harness
 * output and enforces tenant resolution. Uses a mock runner so no live index
 * is needed.
 */

import { describe, it, expect, vi } from 'vitest';
import { runSearchEval } from '../../src/tools/search-eval.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';
import type { SearchBenchmarkRunner } from '../../src/benchmarks/semantic-search.js';

/** Runner that always returns `top` at rank 1 and one filler hit. */
function makeRunner(topRelPath: string): SearchBenchmarkRunner {
  return {
    collectionExists: vi.fn().mockResolvedValue(true),
    search: vi.fn(async (options: { query: string; mode?: string }) => ({
      results: [
        {
          id: '1',
          score: 0.9,
          collection: 'projects',
          content: '',
          metadata: { relative_path: topRelPath },
        },
        {
          id: '2',
          score: 0.8,
          collection: 'projects',
          content: '',
          metadata: { relative_path: 'src/unrelated/filler.ts' },
        },
      ],
      total: 2,
      query: options.query,
      mode: options.mode ?? 'semantic',
      scope: 'project',
      collections_searched: ['projects'],
      status: 'ok',
    })),
  } as unknown as SearchBenchmarkRunner;
}

function makeDetector(projectId?: string): ProjectDetector {
  return {
    getProjectInfo: vi.fn().mockResolvedValue(projectId ? { projectId } : null),
  } as unknown as ProjectDetector;
}

describe('runSearchEval', () => {
  it('evaluates inline cases and returns per-mode metrics + verdict', async () => {
    const res = await runSearchEval(makeRunner('src/tools/search.ts'), makeDetector('p1'), {
      cases: [{ query: 'where is the search tool', expectedFiles: ['src/tools/search.ts'] }],
    });

    expect(res.success).toBe(true);
    expect(res.datasetSource).toBe('inline');
    expect(res.queryCount).toBe(1);
    expect(res.projectId).toBe('p1');
    // Expected file is at rank 1 → 100% top1/top3.
    expect(res.modes?.semantic?.top1).toBe(100);
    expect(res.modes?.semantic?.top3).toBe(100);
    expect(res.modes?.semantic?.duplicateRate).toBe(0);
    expect(res.perQuery?.[0]?.semantic).toMatchObject({ top1: true, top3: true, top10: true });
    expect(res.verdict?.grade).toBeDefined();
  });

  it('reports a miss when the expected file is absent', async () => {
    const res = await runSearchEval(makeRunner('src/unrelated/other.ts'), makeDetector('p1'), {
      cases: [{ query: 'q', expectedFiles: ['src/tools/search.ts'] }],
    });
    expect(res.success).toBe(true);
    expect(res.modes?.semantic?.top10).toBe(0);
    expect(res.perQuery?.[0]?.semantic).toMatchObject({ top1: false, top10: false });
  });

  it('refuses project scope when no tenant can be resolved', async () => {
    const res = await runSearchEval(makeRunner('a.ts'), makeDetector(undefined), {
      cases: [{ query: 'x', expectedFiles: ['a.ts'] }],
    });
    expect(res.success).toBe(false);
    expect(res.error).toMatch(/project/i);
  });

  it('includes top paths when includeTopPaths=true', async () => {
    const res = await runSearchEval(makeRunner('src/tools/search.ts'), makeDetector('p1'), {
      cases: [{ query: 'q', expectedFiles: ['src/tools/search.ts'] }],
      includeTopPaths: true,
    });
    const sem = res.perQuery?.[0]?.semantic as Record<string, unknown>;
    expect(sem.topPaths).toEqual(['src/tools/search.ts', 'src/unrelated/filler.ts']);
  });
});
