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
    qdrantConfig: {
        qdrantUrl: string;
        qdrantApiKey?: string;
    };
}
/** Instantiate all server components from config. */
export declare function buildServerComponents(config: ServerConfig): ServerComponents;
//# sourceMappingURL=server-factory.d.ts.map