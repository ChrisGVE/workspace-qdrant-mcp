/**
 * `search_eval` MCP tool — benchmark semantic-search quality in-process.
 *
 * Runs a set of known-item queries (each with `expectedFiles`) through the
 * live SearchTool and returns hit@k / recall@10 / MRR / duplicate-rate per
 * mode plus the quality verdict. This is the agent-facing measure→edit→measure
 * loop: it reuses the same harness as `npm run benchmark:semantic` but runs
 * inside the MCP server (real DB + Qdrant), so no host DB snapshot is needed.
 *
 * See docs/testing/semantic-search-benchmarking.md.
 */

import { existsSync } from 'node:fs';
import { join } from 'node:path';

import type { ProjectDetector } from '../utils/project-detector.js';
import { getEffectiveCwd } from '../utils/request-context.js';
import type { SearchScope } from './search-types.js';
import {
  loadSemanticSearchBenchmarkDataset,
  runSemanticSearchBenchmark,
  type BenchmarkMode,
  type SearchBenchmarkRunner,
  type SemanticSearchBenchmarkDataset,
  type SemanticSearchModeSummary,
} from '../benchmarks/semantic-search.js';

/** One ad-hoc evaluation case (a query + the files that should rank for it). */
export interface SearchEvalCase {
  id?: string;
  query: string;
  expectedFiles: string[];
}

/** All three modes are always executed by the harness; we report each. */
const REPORT_MODES: readonly BenchmarkMode[] = ['semantic', 'hybrid', 'exact'];

/** Relative path to the bundled dataset under the bind-mounted repo. */
const BUNDLED_DATASET_REL = 'src/typescript/mcp-server/scripts/benchmark-data/semantic-search-quality.yaml';

/** Round a 0–1 rate to a 1-decimal percentage. */
function pct(rate: number): number {
  return Math.round(rate * 1000) / 10;
}

/** Query-id prefixes that name a dataset category (see the bundled dataset's
 *  description). Anything else — including the original 2026-05 known-item
 *  set — reports under "orig". */
const CATEGORY_PREFIXES = ['pt', 'sym', 'impl', 'doc', 'real'] as const;

/** Category of a query id: its known prefix before the first '-', else "orig". */
export function categoryOf(id: string): string {
  const dash = id.indexOf('-');
  if (dash > 0 && (CATEGORY_PREFIXES as readonly string[]).includes(id.slice(0, dash))) {
    return id.slice(0, dash);
  }
  return 'orig';
}

interface CategoryHitCounts {
  n: number;
  top1: number;
  top3: number;
  top10: number;
}

/** Per-category hit rates for one ranked mode, as percentages. */
function shapeCategory(counts: CategoryHitCounts): Record<string, number> {
  return {
    n: counts.n,
    top1: pct(counts.top1 / counts.n),
    top3: pct(counts.top3 / counts.n),
    top10: pct(counts.top10 / counts.n),
  };
}

function shapeModeSummary(summary: SemanticSearchModeSummary): Record<string, number> {
  return {
    top1: pct(summary.top1HitRate),
    top3: pct(summary.top3HitRate),
    top10: pct(summary.top10HitRate),
    recallAt10: pct(summary.recallAt10),
    precisionAt10: pct(summary.precisionAt10),
    mrr: Math.round(summary.mrr * 100) / 100,
    duplicateRate: pct(summary.duplicateRate),
    avgLatencyMs: Math.round(summary.avgLatencyMs * 10) / 10,
  };
}

/** Build a one-shot dataset from inline cases. */
function datasetFromCases(cases: readonly SearchEvalCase[]): SemanticSearchBenchmarkDataset {
  return {
    name: 'ad-hoc',
    description: 'Inline cases passed to the search_eval tool.',
    queries: cases.map((c, index) => ({
      id: c.id?.trim() || `case-${index + 1}`,
      query: c.query,
      expectedFiles: c.expectedFiles ?? [],
    })),
  };
}

export interface SearchEvalResult {
  success: boolean;
  error?: string;
  datasetSource?: string;
  queryCount?: number;
  projectId?: string;
  verdict?: { grade: string; reasons: string[] };
  modes?: Record<string, Record<string, number>>;
  /** Hit rates per dataset category (query-id prefix: pt/sym/impl/doc/real,
   *  else "orig") for the ranked modes — so a weak category (e.g. PT) is
   *  visible instead of silently dragging the aggregate verdict. */
  byCategory?: Record<string, { semantic: Record<string, number>; hybrid: Record<string, number> }>;
  perQuery?: Array<Record<string, unknown>>;
}

/**
 * Resolve the tenant for a project-scoped eval. Mirrors the search/exact
 * tools: explicit projectId wins, else detect from the effective cwd, else
 * (for non-`all` scope) fail rather than search the wrong/every tenant.
 */
async function resolveTenant(
  projectId: string | undefined,
  scope: SearchScope,
  projectDetector: ProjectDetector
): Promise<{ tenantId?: string } | { error: string }> {
  if (scope === 'all') return {};
  if (projectId) return { tenantId: projectId };
  const info = await projectDetector.getProjectInfo(getEffectiveCwd(), false, {
    fallbackToSoleProject: true,
  });
  if (info?.projectId) return { tenantId: info.projectId };
  return {
    error:
      'Could not resolve a project for the eval. Pass `projectId`, run from a registered ' +
      'project directory (with `cwd`), or set `scope: "all"`.',
  };
}

/**
 * Execute the `search_eval` tool. `runner` is the live SearchTool (typed as the
 * minimal benchmark-runner contract so it stays unit-testable with a mock).
 */
export async function runSearchEval(
  runner: SearchBenchmarkRunner,
  projectDetector: ProjectDetector,
  args: Record<string, unknown> | undefined
): Promise<SearchEvalResult> {
  const scope = (args?.['scope'] as SearchScope | undefined) ?? 'project';
  const tenant = await resolveTenant(args?.['projectId'] as string | undefined, scope, projectDetector);
  if ('error' in tenant) return { success: false, error: tenant.error };

  // Dataset: inline `cases` take precedence; otherwise fall back to the bundled
  // dataset reachable via the bind-mounted repo (WQM_REPO_DIR).
  const rawCases = args?.['cases'] as SearchEvalCase[] | undefined;
  let dataset: SemanticSearchBenchmarkDataset;
  let datasetSource: string;
  if (Array.isArray(rawCases) && rawCases.length > 0) {
    dataset = datasetFromCases(rawCases);
    datasetSource = 'inline';
  } else {
    const repoDir = process.env['WQM_REPO_DIR'];
    const dsPath = repoDir ? join(repoDir, BUNDLED_DATASET_REL) : undefined;
    if (!dsPath || !existsSync(dsPath)) {
      return {
        success: false,
        error:
          'No `cases` provided and the bundled dataset is not reachable ' +
          '(WQM_REPO_DIR unset or file missing). Pass `cases: [{ query, expectedFiles }]`.',
      };
    }
    dataset = loadSemanticSearchBenchmarkDataset(dsPath);
    datasetSource = dsPath;
  }

  const limit = (args?.['limit'] as number | undefined) ?? 10;
  const topK = (args?.['topK'] as number | undefined) ?? 10;
  const includeTopPaths = (args?.['includeTopPaths'] as boolean | undefined) ?? false;
  const rerank = args?.['rerank'] as boolean | undefined;
  const rerankWeight = args?.['rerankWeight'] as number | undefined;

  const report = await runSemanticSearchBenchmark(runner, dataset, {
    // relative_path on each hit carries the repo-relative path, so the
    // workspace-root prefix strip is a no-op — pass empty.
    workspaceRoot: '',
    scope,
    limit,
    topK,
    warmupRuns: 0,
    iterations: 1,
    modes: REPORT_MODES,
    datasetSourcePath: datasetSource,
    ...(tenant.tenantId ? { projectId: tenant.tenantId } : {}),
    ...(rerank !== undefined ? { rerank } : {}),
    ...(rerankWeight !== undefined ? { rerankWeight } : {}),
  });

  // Per-category hit rates (semantic + hybrid) keyed by query-id prefix.
  const byCategory: Record<string, { semantic: CategoryHitCounts; hybrid: CategoryHitCounts }> = {};
  for (const q of report.queries) {
    const cat = categoryOf(q.id);
    const bucket = (byCategory[cat] ??= {
      semantic: { n: 0, top1: 0, top3: 0, top10: 0 },
      hybrid: { n: 0, top1: 0, top3: 0, top10: 0 },
    });
    for (const mode of ['semantic', 'hybrid'] as const) {
      const ev = q.modes[mode].evaluation;
      bucket[mode].n += 1;
      bucket[mode].top1 += ev.top1Hit ? 1 : 0;
      bucket[mode].top3 += ev.top3Hit ? 1 : 0;
      bucket[mode].top10 += ev.top10Hit ? 1 : 0;
    }
  }

  return {
    success: true,
    datasetSource,
    queryCount: report.summary.queryCount,
    ...(tenant.tenantId ? { projectId: tenant.tenantId } : {}),
    verdict: {
      grade: report.summary.semanticVerdict.grade,
      reasons: report.summary.semanticVerdict.reasons,
    },
    modes: {
      semantic: shapeModeSummary(report.summary.modes.semantic),
      hybrid: shapeModeSummary(report.summary.modes.hybrid),
      exact: shapeModeSummary(report.summary.modes.exact),
    },
    byCategory: Object.fromEntries(
      Object.entries(byCategory).map(([cat, counts]) => [
        cat,
        { semantic: shapeCategory(counts.semantic), hybrid: shapeCategory(counts.hybrid) },
      ])
    ),
    perQuery: report.queries.map((q) => {
      const ev = q.modes.semantic.evaluation;
      const hv = q.modes.hybrid.evaluation;
      return {
        id: q.id,
        query: q.query,
        expected: q.expectedFiles,
        bestMode: q.bestMode,
        semantic: {
          top1: ev.top1Hit,
          top3: ev.top3Hit,
          top10: ev.top10Hit,
          firstRelevantRank: ev.firstRelevantRank ?? null,
          ...(includeTopPaths ? { topPaths: ev.rawTopPaths } : {}),
        },
        // Hybrid mirror of the semantic block so per-category rates can be
        // derived for BOTH ranked modes from one run.
        hybrid: {
          top1: hv.top1Hit,
          top3: hv.top3Hit,
          top10: hv.top10Hit,
          firstRelevantRank: hv.firstRelevantRank ?? null,
        },
      };
    }),
  };
}
