#!/usr/bin/env node

import { writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

import { DaemonClient } from '../src/clients/daemon-client.js';
import { SqliteStateManager } from '../src/clients/sqlite-state-manager.js';
import { loadConfig } from '../src/config.js';
import { SearchTool } from '../src/tools/search.js';
import { ProjectDetector } from '../src/utils/project-detector.js';
import {
  findRepositoryRoot,
  loadSemanticSearchBenchmarkDataset,
  runSemanticSearchBenchmark,
  ensureDirectoryForFile,
  type SemanticSearchBenchmarkReport,
  type SemanticSearchBenchmarkRunConfig,
} from '../src/benchmarks/semantic-search.js';

type SearchScope = 'project' | 'global' | 'all';

export interface SweepScenario {
  name: string;
  rerank?: boolean;
  rerankWeight?: number;
}

interface SweepRun {
  scenario: SweepScenario;
  report: SemanticSearchBenchmarkReport;
}

function parseInteger(value: string | undefined, fallback: number, fieldName: string): number {
  if (value === undefined) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`Option --${fieldName} must be a non-negative integer.`);
  }
  return parsed;
}

function parsePositiveInteger(
  value: string | undefined,
  fallback: number,
  fieldName: string
): number {
  if (value === undefined) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Option --${fieldName} must be a positive integer.`);
  }
  return parsed;
}

function parseWeight(value: string, fieldName: string): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`${fieldName} must be a number between 0 and 1.`);
  }
  return parsed;
}

function resolvePathFromCwd(value: string | undefined): string | undefined {
  if (value === undefined) return undefined;
  if (value === '~') return homedir();
  const homeMatch = /^~[\/](.*)$/.exec(value);
  if (homeMatch) return resolve(homedir(), homeMatch[1]);
  return resolve(process.cwd(), value);
}

function parseBoolean(value: string, fieldName: string): boolean {
  if (/^(1|true|yes|y|on)$/i.test(value)) return true;
  if (/^(0|false|no|n|off)$/i.test(value)) return false;
  throw new Error(`${fieldName} must be true/false or on/off.`);
}

export function parseScenarioSpec(spec: string): SweepScenario {
  const trimmed = spec.trim();
  if (trimmed.length === 0) throw new Error('Scenario cannot be empty.');

  const [rawName, rawSettings] = trimmed.includes(':')
    ? (trimmed.split(/:(.*)/s).slice(0, 2) as [string, string])
    : [trimmed, ''];
  const name = rawName.trim();
  if (name.length === 0) throw new Error(`Scenario ${spec} is missing a name.`);

  const scenario: SweepScenario = { name };
  const settings = rawSettings.trim();
  const shorthand = settings.length > 0 ? settings : name;

  if (/^(current|default)$/i.test(shorthand)) return scenario;
  if (/^(off|no-rerank|rerank-off)$/i.test(shorthand)) return { ...scenario, rerank: false };
  if (/^(on|rerank|rerank-on)$/i.test(shorthand)) return { ...scenario, rerank: true };
  if (/^(w|weight)=/i.test(shorthand)) {
    const weight = parseWeight(shorthand.split('=')[1] ?? '', `Scenario ${name} weight`);
    return { ...scenario, rerank: true, rerankWeight: weight };
  }
  if (/^[0-9.]+$/.test(shorthand)) {
    return {
      ...scenario,
      rerank: true,
      rerankWeight: parseWeight(shorthand, `Scenario ${name} weight`),
    };
  }

  for (const part of settings
    .split(',')
    .map((p) => p.trim())
    .filter(Boolean)) {
    const [rawKey, rawValue] = part.split(/=(.*)/s).slice(0, 2) as [string, string | undefined];
    const key = rawKey.trim().toLowerCase();
    const value = rawValue?.trim();
    if (!value) throw new Error(`Scenario ${name} setting ${rawKey} is missing a value.`);
    if (key === 'rerank') scenario.rerank = parseBoolean(value, `Scenario ${name} rerank`);
    else if (key === 'weight' || key === 'rerankweight' || key === 'rerank-weight') {
      scenario.rerankWeight = parseWeight(value, `Scenario ${name} weight`);
      if (scenario.rerank === undefined) scenario.rerank = true;
    } else {
      throw new Error(`Unsupported scenario setting: ${rawKey}`);
    }
  }

  return scenario;
}

export function buildDefaultScenarios(weightsCsv: string | undefined): SweepScenario[] {
  const weights = (weightsCsv ?? '0.25,0.5,1')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => parseWeight(value, '--weights'));

  return [
    { name: 'current' },
    { name: 'rerank-off', rerank: false },
    ...weights.map((weight) => ({
      name: `rerank-${weight}`,
      rerank: true,
      rerankWeight: weight,
    })),
  ];
}

function fmtPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function fmtNumber(value: number): string {
  return value.toFixed(1);
}

function scenarioLabel(scenario: SweepScenario): string {
  const parts = [scenario.name];
  if (scenario.rerank !== undefined) parts.push(`rerank=${scenario.rerank}`);
  if (scenario.rerankWeight !== undefined) parts.push(`w=${scenario.rerankWeight}`);
  return parts.join(' ');
}

function formatSweepTable(runs: readonly SweepRun[]): string {
  const lines: string[] = [];
  lines.push('Semantic search parameter sweep');
  lines.push(`Scenarios: ${runs.length}`);
  if (runs[0]) lines.push(`Queries: ${runs[0].report.summary.queryCount}`);
  lines.push('');
  lines.push(
    [
      'Scenario'.padEnd(26),
      'Sem Top1'.padStart(8),
      'Sem Top3'.padStart(8),
      'Sem Top10'.padStart(9),
      'Sem Rec'.padStart(8),
      'Sem MRR'.padStart(7),
      'Hyb Top3'.padStart(8),
      'Hyb Rec'.padStart(8),
      'Avg ms'.padStart(8),
      'Verdict'.padStart(8),
    ].join('  ')
  );

  for (const run of runs) {
    const semantic = run.report.summary.modes.semantic;
    const hybrid = run.report.summary.modes.hybrid;
    lines.push(
      [
        scenarioLabel(run.scenario).slice(0, 26).padEnd(26),
        fmtPercent(semantic.top1HitRate).padStart(8),
        fmtPercent(semantic.top3HitRate).padStart(8),
        fmtPercent(semantic.top10HitRate).padStart(9),
        fmtPercent(semantic.recallAt10).padStart(8),
        semantic.mrr.toFixed(2).padStart(7),
        fmtPercent(hybrid.top3HitRate).padStart(8),
        fmtPercent(hybrid.recallAt10).padStart(8),
        fmtNumber(semantic.avgLatencyMs).padStart(8),
        run.report.summary.semanticVerdict.grade.padStart(8),
      ].join('  ')
    );
  }
  return lines.join('\n');
}

function serializeSweep(runs: readonly SweepRun[]): Record<string, unknown> {
  return {
    generatedAt: new Date().toISOString(),
    scenarios: runs.map((run) => ({
      scenario: run.scenario,
      summary: run.report.summary,
      config: run.report.config,
    })),
    reports: runs.map((run) => ({
      scenario: run.scenario,
      report: run.report,
    })),
  };
}

function validScope(value: string | undefined): SearchScope | undefined {
  return value === 'project' || value === 'global' || value === 'all' ? value : undefined;
}

async function main(): Promise<void> {
  const scriptPath = fileURLToPath(import.meta.url);
  const scriptDir = dirname(scriptPath);
  const defaultDatasetPath = resolve(scriptDir, 'benchmark-data', 'semantic-search-quality.yaml');
  const defaultWorkspaceRoot = findRepositoryRoot(scriptDir);

  const parsed = parseArgs({
    options: {
      dataset: { type: 'string' },
      'workspace-root': { type: 'string' },
      output: { type: 'string' },
      'project-id': { type: 'string' },
      'qdrant-url': { type: 'string' },
      'qdrant-api-key': { type: 'string' },
      'daemon-host': { type: 'string' },
      'daemon-port': { type: 'string' },
      'database-path': { type: 'string' },
      scope: { type: 'string' },
      collection: { type: 'string' },
      limit: { type: 'string' },
      topk: { type: 'string' },
      warmup: { type: 'string' },
      iterations: { type: 'string' },
      'include-libraries': { type: 'boolean' },
      'query-id': { type: 'string', multiple: true },
      scenario: { type: 'string', multiple: true },
      weights: { type: 'string' },
      help: { type: 'boolean' },
    },
    allowPositionals: false,
  });

  if (parsed.values.help) {
    console.log(
      [
        'Usage:',
        '  npm run benchmark:semantic:sweep -- [options]',
        '',
        'Common options mirror benchmark:semantic: --workspace-root, --project-id, --qdrant-url,',
        '  --daemon-host, --daemon-port, --database-path, --query-id, --limit, --topk.',
        '',
        'Sweep options:',
        '  --weights <csv>             Default rerank weights to test (default: 0.25,0.5,1)',
        '  --scenario <spec>           Repeatable. Examples:',
        '                              current',
        '                              off:rerank=false',
        '                              weak:rerank=true,weight=0.25',
        '                              pure:w=1',
        '  --output <path>             Write combined JSON report',
        '  --help                      Show this message',
      ].join('\n')
    );
    return;
  }

  const config = loadConfig();
  const datasetPath = resolvePathFromCwd(parsed.values.dataset) ?? defaultDatasetPath;
  const workspaceRoot = resolvePathFromCwd(parsed.values['workspace-root']) ?? defaultWorkspaceRoot;
  const outputPath = resolvePathFromCwd(parsed.values.output);
  const dataset = loadSemanticSearchBenchmarkDataset(datasetPath);
  const queryIds = parsed.values['query-id']
    ? parsed.values['query-id'].map((value) => value.trim()).filter(Boolean)
    : undefined;
  const scenarios = parsed.values.scenario?.length
    ? parsed.values.scenario.map(parseScenarioSpec)
    : buildDefaultScenarios(parsed.values.weights);

  const baseConfig: SemanticSearchBenchmarkRunConfig = {
    workspaceRoot,
    projectId: parsed.values['project-id']?.trim() || undefined,
    scope: validScope(parsed.values.scope),
    collection: parsed.values.collection?.trim() || undefined,
    includeLibraries: parsed.values['include-libraries'] ?? undefined,
    limit: parsePositiveInteger(parsed.values.limit, 10, 'limit'),
    topK: parsePositiveInteger(parsed.values.topk, 10, 'topk'),
    warmupRuns: parseInteger(parsed.values.warmup, 1, 'warmup'),
    iterations: parsePositiveInteger(parsed.values.iterations, 1, 'iterations'),
    queryIds,
    datasetSourcePath: datasetPath,
  };

  const daemonHost = parsed.values['daemon-host']?.trim() || config.daemon.grpcHost;
  const daemonPort = parsed.values['daemon-port']
    ? parsePositiveInteger(parsed.values['daemon-port'], config.daemon.grpcPort, 'daemon-port')
    : config.daemon.grpcPort;
  const qdrantUrl = parsed.values['qdrant-url']?.trim() || config.qdrant.url;
  const qdrantApiKey = parsed.values['qdrant-api-key']?.trim() || config.qdrant.apiKey;
  const databasePath = parsed.values['database-path']
    ? resolvePathFromCwd(parsed.values['database-path'])
    : config.database.path;

  const stateManager = new SqliteStateManager({ dbPath: databasePath });
  const projectDetector = new ProjectDetector({ stateManager });
  const daemonClient = new DaemonClient({ host: daemonHost, port: daemonPort, timeoutMs: 5000 });
  const searchTool = new SearchTool(
    {
      qdrantUrl,
      ...(qdrantApiKey ? { qdrantApiKey } : {}),
    },
    daemonClient,
    stateManager,
    projectDetector
  );

  try {
    await daemonClient.connect();

    const selectedQueries = baseConfig.queryIds
      ? dataset.queries.filter((query) => baseConfig.queryIds?.includes(query.id))
      : dataset.queries;
    const needsProjectId = selectedQueries.some((query) => {
      const scope = query.scope ?? baseConfig.scope ?? dataset.defaults?.scope ?? 'project';
      const queryProjectId = query.projectId ?? baseConfig.projectId ?? dataset.defaults?.projectId;
      return scope !== 'all' && !queryProjectId;
    });

    let resolvedProjectId = baseConfig.projectId;
    if (!resolvedProjectId && needsProjectId) {
      resolvedProjectId =
        (await projectDetector.getCurrentProjectId(workspaceRoot, false)) ?? undefined;
      if (!resolvedProjectId) {
        throw new Error(
          'Unable to resolve a projectId for project-scoped semantic-search benchmarking. ' +
            'Pass --project-id or run from a workspace root that is already registered with the daemon.'
        );
      }
    }

    const runs: SweepRun[] = [];
    for (const scenario of scenarios) {
      console.error(`Running scenario: ${scenarioLabel(scenario)}`);
      const report = await runSemanticSearchBenchmark(searchTool, dataset, {
        ...baseConfig,
        projectId: resolvedProjectId,
        ...(scenario.rerank !== undefined ? { rerank: scenario.rerank } : {}),
        ...(scenario.rerankWeight !== undefined ? { rerankWeight: scenario.rerankWeight } : {}),
      });
      runs.push({ scenario, report });
    }

    console.log(formatSweepTable(runs));

    if (outputPath) {
      ensureDirectoryForFile(outputPath);
      writeFileSync(outputPath, `${JSON.stringify(serializeSweep(runs), null, 2)}\n`, 'utf8');
      console.log(`\nJSON sweep report written to ${outputPath}`);
    }
  } finally {
    daemonClient.close();
    stateManager.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
