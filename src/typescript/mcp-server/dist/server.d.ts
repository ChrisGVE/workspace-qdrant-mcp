/**
 * MCP Server with session lifecycle management
 *
 * Implements WorkspaceQdrantMcpServer class with:
 * - Session start: project detection, daemon registration, heartbeat start
 * - Session end: heartbeat stop, project deprioritization
 * - Graceful degradation when daemon unavailable
 *
 * Uses @modelcontextprotocol/sdk for MCP protocol handling.
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import type { ServerConfig } from './types/index.js';
import type { SessionState, ServerOptions, ServerMode, HttpTransportOptions } from './server-types.js';
export type { SessionState, ServerOptions, ServerMode, HttpTransportOptions, } from './server-types.js';
/**
 * Workspace Qdrant MCP Server
 *
 * Manages the MCP server lifecycle including session management,
 * project registration with the daemon, and heartbeat maintenance.
 */
export declare class WorkspaceQdrantMcpServer {
    private readonly server;
    private readonly components;
    private sessionState;
    private readonly mode;
    private readonly httpOptions;
    private readonly authConfig;
    private httpHandle;
    private isInitialized;
    constructor(options: ServerOptions);
    private setupHandlers;
    private handleToolCall;
    start(): Promise<void>;
    stop(): Promise<void>;
    getMode(): ServerMode;
    /**
     * Tear down all resources for the current session.
     *
     * Safe to call multiple times. Only the first invocation runs cleanup;
     * subsequent calls no-op via the `cleaned` flag (F-049). This prevents
     * double-decrement of the `wqm_mcp_session_count` metric when both the
     * `onclose` handler and `stop()` fire for the same session.
     */
    private cleanupSession;
    private seedDefaultRule;
    getSessionState(): Readonly<SessionState>;
    isReady(): boolean;
    isDaemonConnected(): boolean;
    getMcpServer(): Server;
    getDaemonClient(): import("./clients/daemon-client.js").DaemonClient;
    getStateManager(): import("./clients/sqlite-state-manager.js").SqliteStateManager;
    getProjectDetector(): import("./utils/project-detector.js").ProjectDetector;
    getHealthMonitor(): import("./utils/health-monitor.js").HealthMonitor;
}
/**
 * Create and start the MCP server.
 *
 * @param config    Resolved server configuration.
 * @param modeOrStdio  Either a `ServerMode` string, or (legacy) a boolean
 *                     `stdio` flag. `true` → stdio, `false` → test.
 * @param httpOptions  HTTP transport options; required when `modeOrStdio === 'http'`.
 */
export declare function createServer(config: ServerConfig, modeOrStdio?: ServerMode | boolean, httpOptions?: HttpTransportOptions): Promise<WorkspaceQdrantMcpServer>;
//# sourceMappingURL=server.d.ts.map