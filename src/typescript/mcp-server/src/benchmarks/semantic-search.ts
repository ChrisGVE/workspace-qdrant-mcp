/**
 * Semantic search quality benchmark helpers.
 *
 * This module loads a curated dataset, runs it against the SearchTool
 * in `semantic`, `hybrid`, and `exact` modes, and computes quality
 * metrics that are useful for deciding whether semantic search is
 * actually good on real project queries.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { parse as parseYaml } from 'yaml';

import { determineCollections } from '../tools/search-filters.js';
import { PROJECTS_COLLECTION } from '../tools/search-types.js';
import type { SearchMode, SearchOptions, SearchResponse, SearchResult, SearchScope } from '../tools/search.js';

export const SEMANTIC_SEARCH_BENCHMARK_MODES = ['semantic', 'hybrid', 'exact'] as const;

export type BenchmarkMode = (typeof SEMANTIC_SEARCH_BENCHMARK_MODES)[number];

// Verdict gates on the two INDEPENDENT known-item signals: recall@10
// (did we surface the relevant files at all) and top-3 useful rate (are
// they ranked high enough to be seen). precision@10 is intentionally NOT a
// gate — with only 1–2 relevant files per query it is ≈ recall@10 ×
// (meanExpected/10) ≈ recall × 0.19, i.e. a rescaled copy of recall, so
// gating on both would double-count one signal (and the old 0.84 bar was
// mathematically unreachable since precision@10 caps at ~0.19). It stays in
// the reported metrics for visibility.
export const DEFAULT_SEMANTIC_QUALITY_THRESHOLDS = {
  top3UsefulRate: 0.8,
  recallAt10: 0.7,
} as const;

export interface SemanticSearchBenchmarkDatasetDefaults {
  scope?: SearchScope;
  collection?: string;
  includeLibraries?: boolean;
  limit?: number;
  projectId?: string;
}

export interface SemanticSearchBenchmarkQuery {
  id: string;
  query: string;
  expectedFiles: string[];
  scope?: SearchScope;
  collection?: string;
  includeLibraries?: boolean;
  limit?: number;
  projectId?: string;
  branch?: string;
  fileType?: string;
  pathGlob?: string;
  component?: string;
  libraryName?: string;
  tag?: string;
  tags?: string[];
}

export interface SemanticSearchBenchmarkDataset {
  name: string;
  description?: string;
  defaults?: SemanticSearchBenchmarkDatasetDefaults;
  queries: SemanticSearchBenchmarkQuery[];
}

export interface SemanticSearchBenchmarkRunConfig {
  workspaceRoot: string;
  projectId?: string;
  scope?: SearchScope;
  collection?: string;
  includeLibraries?: boolean;
  limit?: number;
  topK?: number;
  warmupRuns?: number;
  iterations?: number;
  modes?: readonly BenchmarkMode[];
  queryIds?: readonly string[];
  datasetSourcePath?: string;
}

export interface SearchBenchmarkRunner {
  search(options: SearchOptions): Promise<SearchResponse>;
  collectionExists?(collectionName: string): Promise<boolean>;
}

export interface SearchHitSummary {
  rank: number;
  filePath: string;
  score: number;
  collection: string;
}

export interface SearchEvaluation {
  expectedFiles: string[];
  matchedExpectedFiles: string[];
  missingExpectedFiles: string[];
  rawTopPaths: string[];
  topPaths: string[];
  top1Hit: boolean;
  top3Hit: boolean;
  top10Hit: boolean;
  firstRelevantRank: number | undefined;
  precisionAt10: number;
  recallAt10: number;
  duplicateRate: number;
  mrr: number;
}

export interface SemanticSearchModeRun {
  mode: BenchmarkMode;
  searchOptions: SearchOptions;
  status: 'ok' | 'uncertain';
  statusReason: string | undefined;
  total: number;
  collectionsSearched: string[];
  latencySamplesMs: number[];
  latencyMeanMs: number;
  latencyMedianMs: number;
  latencyP95Ms: number;
  topResults: SearchHitSummary[];
  evaluation: SearchEvaluation;
}

export interface SemanticSearchQueryRun {
  id: string;
  query: string;
  expectedFiles: string[];
  bestMode: BenchmarkMode;
  semanticRescuedByHybridTop10: boolean;
  semanticRescuedByExactTop10: boolean;
  modes: Record<BenchmarkMode, SemanticSearchModeRun>;
}

export interface SemanticSearchModeSummary {
  runs: number;
  top1HitRate: number;
  top3HitRate: number;
  top10HitRate: number;
  precisionAt10: number;
  recallAt10: number;
  mrr: number;
  duplicateRate: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
}

export interface SemanticSearchVerdict {
  grade: 'good' | 'mixed' | 'poor';
  reasons: string[];
  thresholds: typeof DEFAULT_SEMANTIC_QUALITY_THRESHOLDS;
}

export interface SemanticSearchBenchmarkSummary {
  queryCount: number;
  modes: Record<BenchmarkMode, SemanticSearchModeSummary>;
  semanticVerdict: SemanticSearchVerdict;
  rescueCounts: {
    hybridRescuedSemanticTop10: number;
    exactRescuedSemanticTop10: number;
  };
}

export interface SemanticSearchBenchmarkReport {
  generatedAt: string;
  workspaceRoot: string;
  dataset: {
    name: string;
    description?: string;
    sourcePath?: string;
    queryCount: number;
    defaults?: SemanticSearchBenchmarkDatasetDefaults;
  };
  config: Omit<SemanticSearchBenchmarkRunConfig, 'workspaceRoot'> & { workspaceRoot: string };
  summary: SemanticSearchBenchmarkSummary;
  queries: SemanticSearchQueryRun[];
}

interface RawSemanticSearchBenchmarkDataset {
  name?: unknown;
  description?: unknown;
  defaults?: unknown;
  queries?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asString(value: unknown, fieldName: string): string {
  if (typeof value !== 'string') {
    throw new Error(`Benchmark dataset field "${fieldName}" must be a string.`);
  }
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    throw new Error(`Benchmark dataset field "${fieldName}" cannot be empty.`);
  }
  return trimmed;
}

function asOptionalString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function asOptionalBoolean(value: unknown, fieldName: string): boolean | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'boolean') {
    throw new Error(`Benchmark dataset field "${fieldName}" must be a boolean.`);
  }
  return value;
}

function asOptionalPositiveInteger(value: unknown, fieldName: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(`Benchmark dataset field "${fieldName}" must be a positive integer.`);
  }
  return value;
}

function asStringArray(value: unknown, fieldName: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`Benchmark dataset field "${fieldName}" must be an array of strings.`);
  }
  const result: string[] = [];
  for (const item of value) {
    if (typeof item !== 'string') {
      throw new Error(`Benchmark dataset field "${fieldName}" must contain only strings.`);
    }
    const trimmed = item.trim();
    if (trimmed.length === 0) {
      throw new Error(`Benchmark dataset field "${fieldName}" cannot contain empty strings.`);
    }
    result.push(trimmed);
  }
  return result;
}

function asOptionalStringArray(value: unknown, fieldName: string): string[] | undefined {
  if (value === undefined) return undefined;
  return asStringArray(value, fieldName);
}

function asSearchScope(value: unknown, fieldName: string): SearchScope | undefined {
  if (value === undefined) return undefined;
  if (value !== 'project' && value !== 'global' && value !== 'all') {
    throw new Error(
      `Benchmark dataset field "${fieldName}" must be one of "project", "global", or "all".`
    );
  }
  return value;
}

function dedupeStrings(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    if (seen.has(value)) continue;
    seen.add(value);
    result.push(value);
  }
  return result;
}

function escapeRegexChar(char: string): string {
  return char.replace(/[\\^$+?.()|[\]{}]/g, '\\$&');
}

export function globToRegExp(glob: string): RegExp {
  let pattern = '^';
  for (let index = 0; index < glob.length; index += 1) {
    const char = glob[index];
    if (char === undefined) {
      continue;
    }
    if (char === '*') {
      if (glob[index + 1] === '*') {
        if (glob[index + 2] === '/') {
          pattern += '(?:.*/)?';
          index += 2;
          continue;
        }
        pattern += '.*';
        index += 1;
        continue;
      }
      pattern += '[^/]*';
      continue;
    }
    if (char === '?') {
      pattern += '[^/]';
      continue;
    }
    pattern += escapeRegexChar(char);
  }
  pattern += '$';
  return new RegExp(pattern);
}

export function matchesGlob(value: string, glob: string): boolean {
  return globToRegExp(glob).test(value);
}

export function normalizeBenchmarkPath(input: string, workspaceRoot?: string): string {
  const trimmed = input.trim();
  if (trimmed.length === 0) return '';

  // Work in forward-slash space. node:path's resolve/relative are
  // platform-specific (POSIX on Linux/CI) and mishandle Windows-style inputs
  // like `C:\...`, so strip the workspace-root prefix by string comparison
  // instead — identical behavior on every host.
  const toPosix = (p: string): string => p.replace(/\\/g, '/');
  const stripEnds = (p: string): string => p.replace(/^\.\/+/, '').replace(/\/+$/, '');

  let candidate = stripEnds(toPosix(trimmed));
  if (workspaceRoot) {
    const root = stripEnds(toPosix(workspaceRoot.trim()));
    if (root.length > 0) {
      if (candidate === root) {
        candidate = '';
      } else if (candidate.startsWith(`${root}/`)) {
        candidate = candidate.slice(root.length + 1);
      }
    }
  }

  return candidate.replace(/^\/+/, '').replace(/^\.\/+/, '');
}

function extractResultPath(result: SearchResult): string | undefined {
  const relativePath = result.metadata['relative_path'];
  if (typeof relativePath === 'string' && relativePath.trim().length > 0) {
    return relativePath;
  }
  const filePath = result.metadata['file_path'];
  if (typeof filePath === 'string' && filePath.trim().length > 0) {
    return filePath;
  }
  const fallbackPath = result.metadata['path'];
  if (typeof fallbackPath === 'string' && fallbackPath.trim().length > 0) {
    return fallbackPath;
  }
  return undefined;
}

function expectedMatcher(expected: string): (value: string) => boolean {
  if (/[*?[{]/.test(expected)) {
    const regex = globToRegExp(expected);
    return (value: string) => regex.test(value);
  }
  return (value: string) => value === expected;
}

function normalizeAndDedupeExpectedFiles(expectedFiles: readonly string[], workspaceRoot: string): string[] {
  return dedupeStrings(expectedFiles.map((filePath) => normalizeBenchmarkPath(filePath, workspaceRoot)));
}

function summarizeTopResults(
  results: readonly SearchResult[],
  workspaceRoot: string,
  topK: number
): SearchHitSummary[] {
  const summaries: SearchHitSummary[] = [];
  const seen = new Set<string>();
  for (let index = 0; index < results.length && summaries.length < topK; index += 1) {
    const result = results[index];
    if (!result) continue;
    const path = extractResultPath(result);
    if (!path) continue;
    const normalizedPath = normalizeBenchmarkPath(path, workspaceRoot);
    if (normalizedPath.length === 0 || seen.has(normalizedPath)) continue;
    seen.add(normalizedPath);
    summaries.push({
      rank: index + 1,
      filePath: normalizedPath,
      score: result.score,
      collection: result.collection,
    });
  }
  return summaries;
}

function percentile(values: readonly number[], pct: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * pct;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower] ?? 0;
  const lowerValue = sorted[lower] ?? 0;
  const upperValue = sorted[upper] ?? lowerValue;
  return lowerValue + (upperValue - lowerValue) * (position - lower);
}

function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function countMatches(
  expectedMatchers: readonly { expected: string; matches: (value: string) => boolean }[],
  values: readonly string[]
): { matchedExpectedFiles: string[]; missingExpectedFiles: string[] } {
  const matchedExpectedFiles: string[] = [];
  for (const expected of expectedMatchers) {
    if (values.some((value) => expected.matches(value))) {
      matchedExpectedFiles.push(expected.expected);
    }
  }
  const matchedSet = new Set(matchedExpectedFiles);
  const missingExpectedFiles = expectedMatchers
    .map((entry) => entry.expected)
    .filter((expected) => !matchedSet.has(expected));
  return { matchedExpectedFiles, missingExpectedFiles };
}

export function evaluateSearchResults(
  results: readonly SearchResult[],
  expectedFiles: readonly string[],
  workspaceRoot: string,
  topK = 10
): SearchEvaluation {
  const normalizedExpectedFiles = normalizeAndDedupeExpectedFiles(expectedFiles, workspaceRoot);
  const expectedMatchers = normalizedExpectedFiles.map((expected) => ({
    expected,
    matches: expectedMatcher(expected),
  }));
  const topResults = summarizeTopResults(results, workspaceRoot, topK);
  const rawTopPaths = results
    .slice(0, topK)
    .map((result) => extractResultPath(result))
    .filter((path): path is string => typeof path === 'string')
    .map((path) => normalizeBenchmarkPath(path, workspaceRoot))
    .filter((path) => path.length > 0);
  const topPaths = topResults.map((result) => result.filePath);
  const { matchedExpectedFiles, missingExpectedFiles } = countMatches(expectedMatchers, topPaths);

  let firstRelevantRank: number | undefined;
  for (let index = 0; index < rawTopPaths.length; index += 1) {
    const path = rawTopPaths[index];
    if (!path) continue;
    if (expectedMatchers.some((expected) => expected.matches(path))) {
      firstRelevantRank = index + 1;
      break;
    }
  }

  const firstRawPath = rawTopPaths[0];
  const top1Hit = firstRawPath
    ? expectedMatchers.some((expected) => expected.matches(firstRawPath))
    : false;
  const top3Hit = rawTopPaths
    .slice(0, 3)
    .some((path) => expectedMatchers.some((expected) => expected.matches(path)));
  const top10Hit = rawTopPaths.some((path) => expectedMatchers.some((expected) => expected.matches(path)));

  const relevantUniqueHits = topPaths.filter((path) =>
    expectedMatchers.some((expected) => expected.matches(path))
  );
  const precisionAt10 = topPaths.length > 0 ? relevantUniqueHits.length / topPaths.length : 0;
  const recallAt10 = normalizedExpectedFiles.length > 0
    ? matchedExpectedFiles.length / normalizedExpectedFiles.length
    : 0;
  const duplicateRate = rawTopPaths.length > 0 ? 1 - topPaths.length / rawTopPaths.length : 0;
  const mrr = firstRelevantRank ? 1 / firstRelevantRank : 0;

  return {
    expectedFiles: normalizedExpectedFiles,
    matchedExpectedFiles,
    missingExpectedFiles,
    rawTopPaths,
    topPaths,
    top1Hit,
    top3Hit,
    top10Hit,
    firstRelevantRank,
    precisionAt10,
    recallAt10,
    duplicateRate,
    mrr,
  };
}

export function summarizeModeRuns(runs: readonly SemanticSearchModeRun[]): SemanticSearchModeSummary {
  if (runs.length === 0) {
    throw new Error('Cannot summarize an empty benchmark mode run set.');
  }

  return {
    runs: runs.length,
    top1HitRate: mean(runs.map((run) => (run.evaluation.top1Hit ? 1 : 0))),
    top3HitRate: mean(runs.map((run) => (run.evaluation.top3Hit ? 1 : 0))),
    top10HitRate: mean(runs.map((run) => (run.evaluation.top10Hit ? 1 : 0))),
    precisionAt10: mean(runs.map((run) => run.evaluation.precisionAt10)),
    recallAt10: mean(runs.map((run) => run.evaluation.recallAt10)),
    mrr: mean(runs.map((run) => run.evaluation.mrr)),
    duplicateRate: mean(runs.map((run) => run.evaluation.duplicateRate)),
    avgLatencyMs: mean(runs.flatMap((run) => run.latencySamplesMs)),
    p95LatencyMs: percentile(runs.flatMap((run) => run.latencySamplesMs), 0.95),
  };
}

export function classifySemanticSearchQuality(
  summary: SemanticSearchModeSummary,
  thresholds: typeof DEFAULT_SEMANTIC_QUALITY_THRESHOLDS = DEFAULT_SEMANTIC_QUALITY_THRESHOLDS
): SemanticSearchVerdict {
  const reasons: string[] = [];
  if (summary.top3HitRate < thresholds.top3UsefulRate) {
    reasons.push(
      `top-3 useful rate ${formatPercent(summary.top3HitRate)} is below ${formatPercent(
        thresholds.top3UsefulRate
      )}`
    );
  }
  if (summary.recallAt10 < thresholds.recallAt10) {
    reasons.push(
      `recall@10 ${formatPercent(summary.recallAt10)} is below ${formatPercent(thresholds.recallAt10)}`
    );
  }

  const grade: SemanticSearchVerdict['grade'] =
    reasons.length === 0 ? 'good' : reasons.length === 1 ? 'mixed' : 'poor';

  return {
    grade,
    reasons,
    thresholds,
  };
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number): string {
  return value.toFixed(1);
}

function resolveOptionalNumber(value: number | undefined, fallback: number): number {
  return value ?? fallback;
}

function parseDatasetDefaults(defaults: unknown): SemanticSearchBenchmarkDatasetDefaults | undefined {
  if (defaults === undefined) return undefined;
  if (!isRecord(defaults)) {
    throw new Error('Benchmark dataset "defaults" must be an object when present.');
  }
  const result: SemanticSearchBenchmarkDatasetDefaults = {};
  const scope = asSearchScope(defaults.scope, 'defaults.scope');
  if (scope !== undefined) result.scope = scope;
  const collection = asOptionalString(defaults.collection);
  if (collection !== undefined) result.collection = collection;
  const includeLibraries = asOptionalBoolean(defaults.includeLibraries, 'defaults.includeLibraries');
  if (includeLibraries !== undefined) result.includeLibraries = includeLibraries;
  const limit = asOptionalPositiveInteger(defaults.limit, 'defaults.limit');
  if (limit !== undefined) result.limit = limit;
  const projectId = asOptionalString(defaults.projectId);
  if (projectId !== undefined) result.projectId = projectId;
  return result;
}

function parseDatasetQuery(rawQuery: unknown, index: number): SemanticSearchBenchmarkQuery {
  if (!isRecord(rawQuery)) {
    throw new Error(`Benchmark dataset query at index ${index} must be an object.`);
  }

  const id = asString(rawQuery.id, `queries[${index}].id`);
  const query = asString(rawQuery.query, `queries[${index}].query`);
  const expectedFiles = asStringArray(rawQuery.expectedFiles, `queries[${index}].expectedFiles`);
  const scope = asSearchScope(rawQuery.scope, `queries[${index}].scope`);
  const collection = asOptionalString(rawQuery.collection);
  const includeLibraries = asOptionalBoolean(
    rawQuery.includeLibraries,
    `queries[${index}].includeLibraries`
  );
  const limit = asOptionalPositiveInteger(rawQuery.limit, `queries[${index}].limit`);
  const projectId = asOptionalString(rawQuery.projectId);
  const branch = asOptionalString(rawQuery.branch);
  const fileType = asOptionalString(rawQuery.fileType);
  const pathGlob = asOptionalString(rawQuery.pathGlob);
  const component = asOptionalString(rawQuery.component);
  const libraryName = asOptionalString(rawQuery.libraryName);
  const tag = asOptionalString(rawQuery.tag);
  const tags = asOptionalStringArray(rawQuery.tags, `queries[${index}].tags`);

  return {
    id,
    query,
    expectedFiles,
    ...(scope !== undefined ? { scope } : {}),
    ...(collection !== undefined ? { collection } : {}),
    ...(includeLibraries !== undefined ? { includeLibraries } : {}),
    ...(limit !== undefined ? { limit } : {}),
    ...(projectId !== undefined ? { projectId } : {}),
    ...(branch !== undefined ? { branch } : {}),
    ...(fileType !== undefined ? { fileType } : {}),
    ...(pathGlob !== undefined ? { pathGlob } : {}),
    ...(component !== undefined ? { component } : {}),
    ...(libraryName !== undefined ? { libraryName } : {}),
    ...(tag !== undefined ? { tag } : {}),
    ...(tags !== undefined ? { tags } : {}),
  };
}

export function loadSemanticSearchBenchmarkDataset(filePath: string): SemanticSearchBenchmarkDataset {
  const absolutePath = resolve(filePath);
  const content = readFileSync(absolutePath, 'utf8');
  const parsed = parseYaml(content) as unknown;
  if (!isRecord(parsed)) {
    throw new Error(`Benchmark dataset file ${absolutePath} must contain a YAML object.`);
  }

  const rawDataset = parsed as RawSemanticSearchBenchmarkDataset;
  const name = asString(rawDataset.name, 'name');
  const description = asOptionalString(rawDataset.description);
  const defaults = parseDatasetDefaults(rawDataset.defaults);
  if (!Array.isArray(rawDataset.queries)) {
    throw new Error('Benchmark dataset must contain a top-level "queries" array.');
  }

  const queries = rawDataset.queries.map((entry, index) => parseDatasetQuery(entry, index));
  const seenIds = new Set<string>();
  for (const query of queries) {
    if (seenIds.has(query.id)) {
      throw new Error(`Benchmark dataset query ids must be unique; duplicate id "${query.id}" found.`);
    }
    seenIds.add(query.id);
  }

  return {
    name,
    ...(description !== undefined ? { description } : {}),
    ...(defaults !== undefined ? { defaults } : {}),
    queries,
  };
}

function mergeQuerySettings(
  query: SemanticSearchBenchmarkQuery,
  defaults: SemanticSearchBenchmarkDatasetDefaults | undefined,
  config: SemanticSearchBenchmarkRunConfig
): {
  scope: SearchScope;
  collection: string | undefined;
  includeLibraries: boolean;
  limit: number;
  projectId: string | undefined;
  branch: string | undefined;
  fileType: string | undefined;
  pathGlob: string | undefined;
  component: string | undefined;
  libraryName: string | undefined;
  tag: string | undefined;
  tags: string[] | undefined;
} {
  const scope = query.scope ?? config.scope ?? defaults?.scope ?? 'project';
  const collection = query.collection ?? config.collection ?? defaults?.collection;
  const includeLibraries = query.includeLibraries ?? config.includeLibraries ?? defaults?.includeLibraries ?? false;
  const limit = resolveOptionalNumber(query.limit, resolveOptionalNumber(config.limit, defaults?.limit ?? 10));
  const projectId = query.projectId ?? config.projectId ?? defaults?.projectId;
  return {
    scope,
    collection,
    includeLibraries,
    limit,
    projectId,
    branch: query.branch,
    fileType: query.fileType,
    pathGlob: query.pathGlob,
    component: query.component,
    libraryName: query.libraryName,
    tag: query.tag,
    tags: query.tags,
  };
}

function buildSearchOptions(
  query: SemanticSearchBenchmarkQuery,
  mode: BenchmarkMode,
  config: SemanticSearchBenchmarkRunConfig,
  defaults: SemanticSearchBenchmarkDatasetDefaults | undefined
): SearchOptions {
  const merged = mergeQuerySettings(query, defaults, config);
  const options: SearchOptions = {
    query: query.query,
    limit: merged.limit,
    scope: merged.scope,
  };
  if (merged.collection !== undefined) options.collection = merged.collection;
  if (merged.projectId !== undefined) options.projectId = merged.projectId;
  if (merged.includeLibraries) options.includeLibraries = true;
  if (merged.branch !== undefined) options.branch = merged.branch;
  if (merged.fileType !== undefined) options.fileType = merged.fileType;
  if (merged.pathGlob !== undefined) options.pathGlob = merged.pathGlob;
  if (merged.component !== undefined) options.component = merged.component;
  if (merged.libraryName !== undefined) options.libraryName = merged.libraryName;
  if (merged.tag !== undefined) options.tag = merged.tag;
  if (merged.tags !== undefined && merged.tags.length > 0) options.tags = merged.tags;

  if (mode === 'exact') {
    options.exact = true;
  } else {
    options.mode = mode as SearchMode;
  }

  return options;
}

function dedupeBenchmarkModes(modes: readonly BenchmarkMode[] | undefined): BenchmarkMode[] {
  if (!modes || modes.length === 0) return [...SEMANTIC_SEARCH_BENCHMARK_MODES];
  const filtered: BenchmarkMode[] = [];
  for (const mode of modes) {
    if (SEMANTIC_SEARCH_BENCHMARK_MODES.includes(mode) && !filtered.includes(mode)) {
      filtered.push(mode);
    }
  }
  return filtered;
}

async function preflightCollections(
  runner: SearchBenchmarkRunner,
  dataset: SemanticSearchBenchmarkDataset,
  queries: readonly SemanticSearchBenchmarkQuery[],
  config: SemanticSearchBenchmarkRunConfig,
  modes: readonly BenchmarkMode[]
): Promise<void> {
  if (!runner.collectionExists) return;

  const requiredCollections = new Set<string>();
  for (const query of queries) {
    const merged = mergeQuerySettings(query, dataset.defaults, config);
    const effectiveCollections = determineCollections(
      merged.collection,
      merged.scope,
      merged.includeLibraries
    );
    for (const collection of effectiveCollections) {
      requiredCollections.add(collection);
    }
    if (modes.includes('exact')) {
      requiredCollections.add(PROJECTS_COLLECTION);
    }
  }

  const missingCollections: string[] = [];
  for (const collection of requiredCollections) {
    const exists = await runner.collectionExists(collection);
    if (!exists) missingCollections.push(collection);
  }

  if (missingCollections.length > 0) {
    throw new Error(
      `Required Qdrant collections are missing: ${missingCollections.join(', ')}. ` +
        'Index the project or adjust the benchmark scope before running the semantic-search benchmark.'
    );
  }
}

function buildModeRun(
  mode: BenchmarkMode,
  response: SearchResponse,
  elapsedSamples: number[],
  workspaceRoot: string,
  topK: number,
  expectedFiles: readonly string[]
): SemanticSearchModeRun {
  const latencyMeanMs = mean(elapsedSamples);
  const latencyMedianMs = percentile(elapsedSamples, 0.5);
  const latencyP95Ms = percentile(elapsedSamples, 0.95);
  return {
    mode,
    searchOptions: {
      query: response.query,
      scope: response.scope,
      limit: response.results.length,
      ...(mode === 'exact'
        ? { exact: true }
        : { mode: response.mode as SearchMode }),
    },
    status: response.status ?? 'ok',
    statusReason: response.status_reason,
    total: response.total,
    collectionsSearched: response.collections_searched,
    latencySamplesMs: elapsedSamples,
    latencyMeanMs,
    latencyMedianMs,
    latencyP95Ms,
    topResults: summarizeTopResults(response.results, workspaceRoot, topK),
    evaluation: evaluateSearchResults(response.results, expectedFiles, workspaceRoot, topK),
  };
}

async function runOneMode(
  runner: SearchBenchmarkRunner,
  mode: BenchmarkMode,
  query: SemanticSearchBenchmarkQuery,
  config: SemanticSearchBenchmarkRunConfig,
  defaults: SemanticSearchBenchmarkDatasetDefaults | undefined,
  workspaceRoot: string
): Promise<SemanticSearchModeRun> {
  const searchOptions = buildSearchOptions(query, mode, config, defaults);
  const warmupRuns = resolveOptionalNumber(config.warmupRuns, 1);
  const iterations = resolveOptionalNumber(config.iterations, 1);
  const topK = resolveOptionalNumber(config.topK, searchOptions.limit ?? 10);

  for (let warmup = 0; warmup < warmupRuns; warmup += 1) {
    await runner.search(searchOptions);
  }

  const latencySamplesMs: number[] = [];
  let measuredResponse: SearchResponse | undefined;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const start = performance.now();
    const response = await runner.search(searchOptions);
    latencySamplesMs.push(performance.now() - start);
    if (!measuredResponse) {
      measuredResponse = response;
    }
  }

  if (!measuredResponse) {
    throw new Error(`Benchmark mode "${mode}" did not produce any measured responses.`);
  }

  return buildModeRun(
    mode,
    measuredResponse,
    latencySamplesMs,
    workspaceRoot,
    topK,
    query.expectedFiles
  );
}

function selectBestMode(modes: Record<BenchmarkMode, SemanticSearchModeRun>): BenchmarkMode {
  const priority: Record<BenchmarkMode, number> = {
    semantic: 0,
    hybrid: 1,
    exact: 2,
  };
  const candidates = [...SEMANTIC_SEARCH_BENCHMARK_MODES].map((mode) => modes[mode]);
  candidates.sort((left, right) => {
    if (left.evaluation.firstRelevantRank === undefined && right.evaluation.firstRelevantRank !== undefined) {
      return 1;
    }
    if (left.evaluation.firstRelevantRank !== undefined && right.evaluation.firstRelevantRank === undefined) {
      return -1;
    }
    const leftRank = left.evaluation.firstRelevantRank ?? Number.POSITIVE_INFINITY;
    const rightRank = right.evaluation.firstRelevantRank ?? Number.POSITIVE_INFINITY;
    if (leftRank !== rightRank) return leftRank - rightRank;
    if (left.evaluation.mrr !== right.evaluation.mrr) {
      return right.evaluation.mrr - left.evaluation.mrr;
    }
    return priority[left.mode] - priority[right.mode];
  });
  return candidates[0]?.mode ?? 'semantic';
}

function countSemanticRescueEvents(queryRuns: readonly SemanticSearchQueryRun[]): {
  hybridRescuedSemanticTop10: number;
  exactRescuedSemanticTop10: number;
} {
  let hybridRescuedSemanticTop10 = 0;
  let exactRescuedSemanticTop10 = 0;
  for (const queryRun of queryRuns) {
    if (queryRun.semanticRescuedByHybridTop10) hybridRescuedSemanticTop10 += 1;
    if (queryRun.semanticRescuedByExactTop10) exactRescuedSemanticTop10 += 1;
  }
  return {
    hybridRescuedSemanticTop10,
    exactRescuedSemanticTop10,
  };
}

export async function runSemanticSearchBenchmark(
  runner: SearchBenchmarkRunner,
  dataset: SemanticSearchBenchmarkDataset,
  config: SemanticSearchBenchmarkRunConfig
): Promise<SemanticSearchBenchmarkReport> {
  const modes = dedupeBenchmarkModes(config.modes);
  const filteredQueries = config.queryIds
    ? dataset.queries.filter((query) => config.queryIds?.includes(query.id))
    : [...dataset.queries];
  if (filteredQueries.length === 0) {
    throw new Error('The semantic search benchmark dataset does not contain any queries to run.');
  }

  await preflightCollections(runner, dataset, filteredQueries, config, modes);

  const queryRuns: SemanticSearchQueryRun[] = [];
  for (const query of filteredQueries) {
    const merged = mergeQuerySettings(query, dataset.defaults, config);
    if (merged.scope !== 'all' && !merged.projectId) {
      throw new Error(
        `Query "${query.id}" requires a projectId because its effective scope is "${merged.scope}". ` +
          'Pass --project-id or add a projectId to the dataset/defaults.'
      );
    }

    const semantic = await runOneMode(
      runner,
      'semantic',
      query,
      config,
      dataset.defaults,
      config.workspaceRoot
    );
    const hybrid = await runOneMode(
      runner,
      'hybrid',
      query,
      config,
      dataset.defaults,
      config.workspaceRoot
    );
    const exact = await runOneMode(
      runner,
      'exact',
      query,
      config,
      dataset.defaults,
      config.workspaceRoot
    );
    const modesByName = { semantic, hybrid, exact };
    queryRuns.push({
      id: query.id,
      query: query.query,
      expectedFiles: semantic.evaluation.expectedFiles,
      bestMode: selectBestMode(modesByName),
      semanticRescuedByHybridTop10:
        !semantic.evaluation.top10Hit && hybrid.evaluation.top10Hit,
      semanticRescuedByExactTop10:
        !semantic.evaluation.top10Hit && exact.evaluation.top10Hit,
      modes: modesByName,
    });
  }

  const summary: SemanticSearchBenchmarkSummary = {
    queryCount: queryRuns.length,
    modes: {
      semantic: summarizeModeRuns(queryRuns.map((queryRun) => queryRun.modes.semantic)),
      hybrid: summarizeModeRuns(queryRuns.map((queryRun) => queryRun.modes.hybrid)),
      exact: summarizeModeRuns(queryRuns.map((queryRun) => queryRun.modes.exact)),
    },
    semanticVerdict: classifySemanticSearchQuality(
      summarizeModeRuns(queryRuns.map((queryRun) => queryRun.modes.semantic))
    ),
    rescueCounts: countSemanticRescueEvents(queryRuns),
  };

  const report: SemanticSearchBenchmarkReport = {
    generatedAt: new Date().toISOString(),
    workspaceRoot: config.workspaceRoot,
    dataset: {
      name: dataset.name,
      queryCount: queryRuns.length,
      ...(dataset.description !== undefined ? { description: dataset.description } : {}),
      ...(config.datasetSourcePath !== undefined ? { sourcePath: config.datasetSourcePath } : {}),
      ...(dataset.defaults !== undefined ? { defaults: dataset.defaults } : {}),
    },
    config: {
      workspaceRoot: config.workspaceRoot,
      ...(config.projectId !== undefined ? { projectId: config.projectId } : {}),
      ...(config.scope !== undefined ? { scope: config.scope } : {}),
      ...(config.collection !== undefined ? { collection: config.collection } : {}),
      ...(config.includeLibraries !== undefined ? { includeLibraries: config.includeLibraries } : {}),
      ...(config.limit !== undefined ? { limit: config.limit } : {}),
      ...(config.topK !== undefined ? { topK: config.topK } : {}),
      ...(config.warmupRuns !== undefined ? { warmupRuns: config.warmupRuns } : {}),
      ...(config.iterations !== undefined ? { iterations: config.iterations } : {}),
      modes,
      ...(config.queryIds !== undefined ? { queryIds: config.queryIds } : {}),
      ...(config.datasetSourcePath !== undefined ? { datasetSourcePath: config.datasetSourcePath } : {}),
    },
    summary,
    queries: queryRuns,
  };

  return report;
}

function formatModeSummaryLine(mode: BenchmarkMode, summary: SemanticSearchModeSummary): string {
  return [
    mode.padEnd(8),
    formatPercent(summary.top1HitRate).padStart(8),
    formatPercent(summary.top3HitRate).padStart(8),
    formatPercent(summary.top10HitRate).padStart(8),
    formatPercent(summary.precisionAt10).padStart(9),
    formatPercent(summary.recallAt10).padStart(8),
    summary.mrr.toFixed(2).padStart(6),
    formatPercent(summary.duplicateRate).padStart(7),
    formatNumber(summary.avgLatencyMs).padStart(8),
    formatNumber(summary.p95LatencyMs).padStart(8),
  ].join('  ');
}

function formatModeHits(run: SemanticSearchModeRun): string {
  const hits = [
    run.evaluation.top1Hit ? 'Y' : 'N',
    run.evaluation.top3Hit ? 'Y' : 'N',
    run.evaluation.top10Hit ? 'Y' : 'N',
  ].join('/');
  return `${hits}${run.status === 'uncertain' ? '!' : ''}`;
}

export function formatSemanticSearchBenchmarkReport(report: SemanticSearchBenchmarkReport): string {
  const lines: string[] = [];
  lines.push('Semantic search quality benchmark');
  lines.push(`Dataset: ${report.dataset.name}`);
  if (report.dataset.description) lines.push(`Description: ${report.dataset.description}`);
  if (report.dataset.sourcePath) lines.push(`Dataset file: ${report.dataset.sourcePath}`);
  lines.push(`Workspace root: ${report.workspaceRoot}`);
  lines.push(`Queries: ${report.summary.queryCount}`);
  lines.push('');
  lines.push(
    [
      'Mode'.padEnd(8),
      'Top1'.padStart(8),
      'Top3'.padStart(8),
      'Top10'.padStart(8),
      'Prec@10'.padStart(9),
      'Rec@10'.padStart(8),
      'MRR'.padStart(6),
      'Dup'.padStart(7),
      'Avg ms'.padStart(8),
      'P95 ms'.padStart(8),
    ].join('  ')
  );
  lines.push(formatModeSummaryLine('semantic', report.summary.modes.semantic));
  lines.push(formatModeSummaryLine('hybrid', report.summary.modes.hybrid));
  lines.push(formatModeSummaryLine('exact', report.summary.modes.exact));
  lines.push('');
  lines.push(
    `Semantic verdict: ${report.summary.semanticVerdict.grade.toUpperCase()} ` +
      (report.summary.semanticVerdict.reasons.length > 0
        ? `(${report.summary.semanticVerdict.reasons.join('; ')})`
        : '(all threshold checks passed)')
  );
  lines.push(
    `Rescues: hybrid rescued semantic on ${report.summary.rescueCounts.hybridRescuedSemanticTop10}/${report.summary.queryCount} queries; ` +
      `exact rescued semantic on ${report.summary.rescueCounts.exactRescuedSemanticTop10}/${report.summary.queryCount} queries`
  );
  lines.push('');
  lines.push('Per query:');
  for (const query of report.queries) {
    lines.push(
      `- ${query.id}: semantic ${formatModeHits(query.modes.semantic)} | ` +
        `hybrid ${formatModeHits(query.modes.hybrid)} | ` +
        `exact ${formatModeHits(query.modes.exact)} | best=${query.bestMode}`
    );
  }
  return lines.join('\n');
}

export function ensureDirectoryForFile(filePath: string): void {
  const targetDirectory = dirname(filePath);
  if (!existsSync(targetDirectory)) {
    mkdirSync(targetDirectory, { recursive: true });
  }
}

export function writeSemanticSearchBenchmarkReport(
  report: SemanticSearchBenchmarkReport,
  outputPath: string
): void {
  ensureDirectoryForFile(outputPath);
  writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
}

export function findRepositoryRoot(startDir: string): string {
  let current = resolve(startDir);
  while (true) {
    if (existsSync(join(current, '.git'))) {
      return current;
    }
    const parent = dirname(current);
    if (parent === current) {
      throw new Error(`Could not locate repository root starting from ${startDir}`);
    }
    current = parent;
  }
}
