/**
 * Factory helpers for constructing server components
 */

import { DaemonClient } from './clients/daemon-client.js';
import { SqliteStateManager } from './clients/sqlite-state-manager.js';
import { ProjectDetector } from './utils/project-detector.js';
import { HealthMonitor } from './utils/health-monitor.js';
import { SearchTool } from './tools/search.js';
import { RetrieveTool } from './tools/retrieve.js';
import { MemoryTool } from './tools/memory.js';
import { StoreTool } from './tools/store.js';
import { GrepTool } from './tools/grep.js';
import type { ServerConfig } from './types/index.js';

export interface ServerComponents {
  daemonClient: DaemonClient;
  stateManager: SqliteStateManager;
  projectDetector: ProjectDetector;
  healthMonitor: HealthMonitor;
  searchTool: SearchTool;
  retrieveTool: RetrieveTool;
  memoryTool: MemoryTool;
  storeTool: StoreTool;
  grepTool: GrepTool;
  qdrantConfig: { qdrantUrl: string; qdrantApiKey?: string };
}

/**
 * Instantiate all server components from config
 */
export function buildServerComponents(config: ServerConfig): ServerComponents {
  const qdrantUrl = config.qdrant?.url ?? 'http://localhost:6333';
  const qdrantApiKey = config.qdrant?.apiKey;

  const daemonClient = new DaemonClient({
    port: config.daemon.grpcPort,
    timeoutMs: 5000,
  });

  const stateManager = new SqliteStateManager({
    dbPath: config.database.path.replace('~', process.env['HOME'] ?? ''),
  });

  const projectDetector = new ProjectDetector({ stateManager });

  // Build Qdrant config conditionally to satisfy exactOptionalPropertyTypes
  const qdrantConfig: { qdrantUrl: string; qdrantApiKey?: string } = { qdrantUrl };
  if (qdrantApiKey) qdrantConfig.qdrantApiKey = qdrantApiKey;

  const healthMonitor = new HealthMonitor(qdrantConfig, daemonClient);

  const searchTool = new SearchTool(
    qdrantConfig,
    daemonClient,
    stateManager,
    projectDetector
  );

  const retrieveTool = new RetrieveTool(qdrantConfig, projectDetector);

  const memoryTool = new MemoryTool(
    qdrantConfig,
    daemonClient,
    stateManager,
    projectDetector
  );

  const grepTool = new GrepTool(daemonClient, projectDetector);

  // StoreTool is for libraries collection ONLY per spec
  const storeTool = new StoreTool({}, stateManager);

  return {
    daemonClient,
    stateManager,
    projectDetector,
    healthMonitor,
    searchTool,
    retrieveTool,
    memoryTool,
    storeTool,
    grepTool,
    qdrantConfig,
  };
}
