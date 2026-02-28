/**
 * Factory helpers for constructing server components
 */

import { DaemonClient } from './clients/daemon-client.js';
import { SqliteStateManager } from './clients/sqlite-state-manager.js';
import { ProjectDetector } from './utils/project-detector.js';
import { HealthMonitor } from './utils/health-monitor.js';
import { SearchTool } from './tools/search.js';
import { RetrieveTool } from './tools/retrieve.js';
import { RulesTool } from './tools/rules.js';
import { StoreTool } from './tools/store.js';
import { GrepTool } from './tools/grep.js';
import { ListFilesTool } from './tools/list-files/index.js';
import type { ServerConfig } from './types/index.js';

export interface ServerComponents {
  daemonClient: DaemonClient;
  stateManager: SqliteStateManager;
  projectDetector: ProjectDetector;
  healthMonitor: HealthMonitor;
  searchTool: SearchTool;
  retrieveTool: RetrieveTool;
  rulesTool: RulesTool;
  storeTool: StoreTool;
  grepTool: GrepTool;
  listTool: ListFilesTool;
  qdrantConfig: { qdrantUrl: string; qdrantApiKey?: string };
}

/** Build Qdrant config, conditionally including API key for exactOptionalPropertyTypes. */
function buildQdrantConfig(config: ServerConfig): { qdrantUrl: string; qdrantApiKey?: string } {
  const qdrantUrl = config.qdrant?.url ?? 'http://localhost:6333';
  const result: { qdrantUrl: string; qdrantApiKey?: string } = { qdrantUrl };
  if (config.qdrant?.apiKey) result.qdrantApiKey = config.qdrant.apiKey;
  return result;
}

/** Create all MCP tool instances. */
function createTools(
  qdrantConfig: { qdrantUrl: string; qdrantApiKey?: string },
  daemonClient: DaemonClient, stateManager: SqliteStateManager,
  projectDetector: ProjectDetector, config: ServerConfig,
) {
  const searchTool = new SearchTool(qdrantConfig, daemonClient, stateManager, projectDetector);
  const retrieveTool = new RetrieveTool(qdrantConfig, projectDetector);
  const rulesConfig: { qdrantUrl: string; qdrantApiKey?: string; duplicationThreshold?: number } = { ...qdrantConfig };
  if (config.rules?.duplicationThreshold !== undefined) {
    rulesConfig.duplicationThreshold = config.rules.duplicationThreshold;
  }
  const rulesTool = new RulesTool(rulesConfig, daemonClient, stateManager, projectDetector);
  const grepTool = new GrepTool(daemonClient, projectDetector);
  const storeTool = new StoreTool({}, stateManager);
  const listTool = new ListFilesTool(stateManager, projectDetector);
  return { searchTool, retrieveTool, rulesTool, grepTool, storeTool, listTool };
}

/** Instantiate all server components from config. */
export function buildServerComponents(config: ServerConfig): ServerComponents {
  const daemonClient = new DaemonClient({ port: config.daemon.grpcPort, timeoutMs: 5000 });
  const stateManager = new SqliteStateManager({
    dbPath: config.database.path.replace('~', process.env['HOME'] ?? ''),
  });
  const projectDetector = new ProjectDetector({ stateManager });
  const qdrantConfig = buildQdrantConfig(config);
  const healthMonitor = new HealthMonitor(qdrantConfig, daemonClient);
  const tools = createTools(qdrantConfig, daemonClient, stateManager, projectDetector, config);

  return { daemonClient, stateManager, projectDetector, healthMonitor, qdrantConfig, ...tools };
}
