#!/usr/bin/env node

import { homedir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { DaemonClient } from '../src/clients/daemon-client.js';
import { SqliteStateManager } from '../src/clients/sqlite-state-manager.js';
import { loadConfig } from '../src/config.js';
import { SearchTool } from '../src/tools/search.js';
import { ProjectDetector } from '../src/utils/project-detector.js';
import {
  findRepositoryRoot,
  formatSemanticSearchBenchmarkReport,
  loadSemanticSearchBenchmarkDataset,
  runSemanticSearchBenchmark,
  writeSemanticSearchBenchmarkReport,
} from '../src/benchmarks/semantic-search.js';

function parseInteger(value: string | undefined, fallback: number, fieldName: string): number {
  if (value === undefined) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`Option --${fieldName} must be a non-negative integer.`);
  }
  return parsed;
}

function parsePositiveInteger(value: string | undefined, fallback: number, fieldName: string): number {
  if (value === undefined) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Option --${fieldName} must be a positive integer.`);
  }
  return parsed;
}

function resolvePathFromCwd(value: string | undefined): string | undefined {
  if (value === undefined) return undefined;
  if (value === '~') return homedir();
  const homeMatch = /^~[\\/](.*)$/.exec(value);
  if (homeMatch) {
    return resolve(homedir(), homeMatch[1]);
  }
  return resolve(process.cwd(), value);
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
      help: { type: 'boolean' },
    },
    allowPositionals: false,
  });

  if (parsed.values.help) {
    console.log([
      'Usage:',
      '  npm run benchmark:semantic -- [options]',
      '',
      'Options:',
      '  --dataset <path>            Path to the benchmark YAML file',
      '  --workspace-root <path>     Workspace root used to normalize file paths',
      '  --project-id <id>           Project tenant id (skips auto-detection)',
      '  --scope <project|global|all>',
      '  --collection <name>         Override the default collection',
      '  --limit <n>                 Search limit per query (default: 10)',
      '  --topk <n>                  Evaluation cutoff (default: same as limit)',
      '  --warmup <n>                Warmup runs per mode (default: 1)',
      '  --iterations <n>            Measured runs per mode (default: 1)',
      '  --include-libraries         Include libraries in project-scope search',
      '  --output <path>             Write the report as JSON',
      '  --qdrant-url <url>          Override the Qdrant endpoint',
      '  --qdrant-api-key <key>      Override the Qdrant API key',
      '  --daemon-host <host>        Override the daemon gRPC host',
      '  --daemon-port <port>        Override the daemon gRPC port',
      '  --database-path <path>      Override the SQLite database path',
      '  --query-id <id>             Run only the selected query id (repeatable)',
      '  --help                      Show this message',
    ].join('\n'));
    return;
  }

  const config = loadConfig();

  const datasetPath = resolvePathFromCwd(parsed.values.dataset) ?? defaultDatasetPath;
  const workspaceRoot = resolvePathFromCwd(parsed.values['workspace-root']) ?? defaultWorkspaceRoot;
  const outputPath = resolvePathFromCwd(parsed.values.output);
  const dataset = loadSemanticSearchBenchmarkDataset(datasetPath);
  const queryIds = parsed.values['query-id'] ? parsed.values['query-id'].map((value) => value.trim()).filter(Boolean) : undefined;

  const benchmarkConfig = {
    workspaceRoot,
    projectId: parsed.values['project-id']?.trim() || undefined,
    scope: parsed.values.scope === 'project' || parsed.values.scope === 'global' || parsed.values.scope === 'all'
      ? parsed.values.scope
      : undefined,
    collection: parsed.values.collection?.trim() || undefined,
    includeLibraries: parsed.values['include-libraries'] ?? undefined,
    limit: parsePositiveInteger(parsed.values.limit, 10, 'limit'),
    topK: parsePositiveInteger(parsed.values.topk, 10, 'topk'),
    warmupRuns: parseInteger(parsed.values.warmup, 1, 'warmup'),
    iterations: parsePositiveInteger(parsed.values.iterations, 1, 'iterations'),
    queryIds,
    datasetSourcePath: datasetPath,
  } as const;

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
  const stateInit = stateManager.initialize();
  if (stateInit.status !== 'ok') {
    console.warn(`SQLite state unavailable for benchmark: ${stateInit.reason ?? 'unknown'}`);
  }
  const projectDetector = new ProjectDetector({ stateManager });
  const daemonClient = new DaemonClient({
    host: daemonHost,
    port: daemonPort,
    timeoutMs: 5000,
  });
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

    const needsProjectId = (benchmarkConfig.queryIds ? dataset.queries.filter((query) => benchmarkConfig.queryIds?.includes(query.id)) : dataset.queries).some((query) => {
      const scope = query.scope ?? benchmarkConfig.scope ?? dataset.defaults?.scope ?? 'project';
      const queryProjectId = query.projectId ?? benchmarkConfig.projectId ?? dataset.defaults?.projectId;
      return scope !== 'all' && !queryProjectId;
    });

    let resolvedProjectId = benchmarkConfig.projectId;
    if (!resolvedProjectId && needsProjectId) {
      resolvedProjectId = (await projectDetector.getCurrentProjectId(workspaceRoot, false)) ?? undefined;
      if (!resolvedProjectId) {
        throw new Error(
          'Unable to resolve a projectId for project-scoped semantic-search benchmarking. ' +
            'Pass --project-id or run from a workspace root that is already registered with the daemon.'
        );
      }
    }

    const report = await runSemanticSearchBenchmark(searchTool, dataset, {
      ...benchmarkConfig,
      projectId: resolvedProjectId,
    });
    const reportText = formatSemanticSearchBenchmarkReport(report);
    console.log(reportText);

    if (outputPath) {
      writeSemanticSearchBenchmarkReport(report, outputPath);
      console.log(`\nJSON report written to ${outputPath}`);
    }
  } finally {
    daemonClient.close();
    stateManager.close();
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
