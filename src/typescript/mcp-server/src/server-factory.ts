/**
 * Factory helpers for constructing server components
 */

import { DaemonClient } from './clients/daemon-client.js';
import { SearchDbReader } from './clients/search-db-reader.js';
import { SqliteStateManager } from './clients/sqlite-state-manager.js';
import { ProjectDetector } from './utils/project-detector.js';
import { HealthMonitor } from './utils/health-monitor.js';
import { SearchTool } from './tools/search.js';
import { RetrieveTool } from './tools/retrieve.js';
import { RulesTool } from './tools/rules.js';
import { StoreTool } from './tools/store.js';
import { ScratchpadTool } from './tools/scratchpad.js';
import { GrepTool } from './tools/grep.js';
import { ListFilesTool } from './tools/list-files/index.js';
import type { ServerConfig } from './types/index.js';
import { DEFAULT_CONFIG } from './types/generated-defaults.js';

export interface ServerComponents {
  daemonClient: DaemonClient;
  stateManager: SqliteStateManager;
  searchDbReader: SearchDbReader;
  projectDetector: ProjectDetector;
  healthMonitor: HealthMonitor;
  searchTool: SearchTool;
  retrieveTool: RetrieveTool;
  rulesTool: RulesTool;
  storeTool: StoreTool;
  scratchpadTool: ScratchpadTool;
  grepTool: GrepTool;
  listTool: ListFilesTool;
  qdrantConfig: { qdrantUrl: string; qdrantApiKey?: string };
}

/** Build Qdrant config, conditionally including API key for exactOptionalPropertyTypes. */
function buildQdrantConfig(config: ServerConfig): { qdrantUrl: string; qdrantApiKey?: string } {
  const qdrantUrl = config.qdrant?.url ?? DEFAULT_CONFIG.qdrant.url;
  const result: { qdrantUrl: string; qdrantApiKey?: string } = { qdrantUrl };
  if (config.qdrant?.apiKey) result.qdrantApiKey = config.qdrant.apiKey;
  return result;
}

/** Create all MCP tool instances. */
function createTools(
  qdrantConfig: { qdrantUrl: string; qdrantApiKey?: string },
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  projectDetector: ProjectDetector,
  config: ServerConfig
) {
  const searchTool = new SearchTool(qdrantConfig, daemonClient, stateManager, projectDetector);
  const retrieveTool = new RetrieveTool(qdrantConfig, projectDetector, daemonClient);
  const rulesConfig: { qdrantUrl: string; qdrantApiKey?: string; duplicationThreshold?: number } = {
    ...qdrantConfig,
  };
  if (config.rules?.duplicationThreshold !== undefined) {
    rulesConfig.duplicationThreshold = config.rules.duplicationThreshold;
  }
  const rulesTool = new RulesTool(rulesConfig, daemonClient, stateManager, projectDetector);
  const grepTool = new GrepTool(daemonClient, projectDetector);
  const storeTool = new StoreTool({}, stateManager);
  const scratchpadTool = new ScratchpadTool(qdrantConfig, stateManager, projectDetector);
  const listTool = new ListFilesTool(stateManager, projectDetector);
  return { searchTool, retrieveTool, rulesTool, grepTool, storeTool, scratchpadTool, listTool };
}

/** Instantiate all server components from config. */
export function buildServerComponents(config: ServerConfig): ServerComponents {
  // DaemonClient is constructed eagerly but connection is lazy (on first RPC call).
  // All consumers handle null/unavailable daemon gracefully via fire-and-forget patterns.
  // gRPC timeout must cover the slowest daemon RPC. `register_project` is the
  // current bottleneck: it eagerly spawns per-language LSP servers and
  // pyright's `initialize` handshake regularly takes ~10s before timing out
  // (and the daemon retries other languages serially), so the whole call
  // routinely lands in the 20-30s range. Five seconds is well below the floor
  // and produces empty `response=` lines in the hook log. Override with
  // `WQM_DAEMON_TIMEOUT_MS` when tuning for a faster (or slower) daemon.
  const daemonTimeoutMs = Number(process.env['WQM_DAEMON_TIMEOUT_MS'] ?? '30000');
  const daemonClient = new DaemonClient({
    host: config.daemon.grpcHost,
    port: config.daemon.grpcPort,
    timeoutMs: Number.isFinite(daemonTimeoutMs) && daemonTimeoutMs > 0 ? daemonTimeoutMs : 30000,
  });
  console.error(`[wqm] daemon gRPC endpoint: ${config.daemon.grpcHost}:${config.daemon.grpcPort}`);
  const stateManager = new SqliteStateManager({
    dbPath: config.database.path.replace('~', process.env['HOME'] ?? ''),
  });
  stateManager.setDaemonClient(daemonClient);
  // search.db lives next to state.db (mirroring the Rust convention in
  // `search_db_path_from_state`). Lazy-init: opens on first admin call.
  const searchDbReader = new SearchDbReader();
  const projectDetector = new ProjectDetector({ stateManager });
  const qdrantConfig = buildQdrantConfig(config);
  const healthMonitor = new HealthMonitor(qdrantConfig, daemonClient);
  const tools = createTools(qdrantConfig, daemonClient, stateManager, projectDetector, config);

  return {
    daemonClient,
    stateManager,
    searchDbReader,
    projectDetector,
    healthMonitor,
    qdrantConfig,
    ...tools,
  };
}
