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
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';

import type { ServerConfig } from './types/index.js';
import { logInfo, logError, logDebug } from './utils/logger.js';
import {
  SERVER_NAME,
  SERVER_VERSION,
  DEFAULT_HTTP_HOST,
  DEFAULT_HTTP_PORT,
  DEFAULT_HTTP_PATH,
} from './server-types.js';
import type {
  SessionState,
  ServerOptions,
  ServerMode,
  HttpTransportOptions,
} from './server-types.js';
export type {
  SessionState,
  ServerOptions,
  ServerMode,
  HttpTransportOptions,
} from './server-types.js';
import { startMcpHttpServer, stopMcpHttpServer } from './mcp-http-server.js';
import type { McpHttpServerHandle } from './mcp-http-server.js';
import type { AuthConfig } from './auth-middleware.js';
import { loadAuthConfig, requireAuth } from './auth-middleware.js';

import { buildServerComponents } from './server-factory.js';
import type { ServerComponents } from './server-factory.js';
import { getToolDefinitions } from './tool-definitions/index.js';
import { initializeSession, startHeartbeat, sendHeartbeat, cleanup } from './session-lifecycle.js';
import { recordSessionStart, recordSessionEnd } from './telemetry/metrics.js';
import { dispatchToolCall } from './tool-dispatcher.js';
import { seedDefaultRule } from './rule-seeder.js';

/**
 * Workspace Qdrant MCP Server
 *
 * Manages the MCP server lifecycle including session management,
 * project registration with the daemon, and heartbeat maintenance.
 */
export class WorkspaceQdrantMcpServer {
  private readonly server: Server;
  private readonly components: ServerComponents;
  private readonly config: ServerConfig;

  private sessionState: SessionState = {
    sessionId: '',
    projectId: null,
    projectPath: null,
    watchPath: null,
    isWorktree: false,
    heartbeatInterval: null,
    daemonConnected: false,
    cleaned: false,
  };

  private readonly mode: ServerMode;
  private readonly httpOptions: HttpTransportOptions;
  private readonly authConfig: AuthConfig;
  private httpHandle: McpHttpServerHandle | null = null;
  private isInitialized = false;

  constructor(options: ServerOptions) {
    this.config = options.config;
    this.mode = resolveMode(options);
    this.httpOptions = {
      host: options.http?.host ?? DEFAULT_HTTP_HOST,
      port: options.http?.port ?? DEFAULT_HTTP_PORT,
      path: options.http?.path ?? DEFAULT_HTTP_PATH,
      ...(options.http?.tls ? { tls: options.http.tls } : {}),
    };
    this.authConfig = options.auth ?? loadAuthConfig();
    this.components = buildServerComponents(options.config);

    this.server = this.createMcpServer();

    this.setupHandlers(this.server, this.components, this.sessionState);
  }

  private createMcpServer(): Server {
    return new Server(
      { name: SERVER_NAME, version: SERVER_VERSION },
      {
        capabilities: { tools: {} },
        instructions: [
          "This server provides access to the user's indexed codebase and knowledge libraries.",
          "ALWAYS use the `search` tool before answering questions about the user's code, project structure, or library documentation.",
          'Use the `rules` tool to check for behavioral preferences before starting work.',
          'Use `retrieve` to access specific documents when you know the document ID.',
          'Use `list` to browse project file/folder structure — start with format "summary" to get an overview.',
          'Collections: projects (indexed code), libraries (reference docs), rules (behavioral rules).',
        ].join(' '),
      }
    );
  }

  private setupHandlers(
    server: Server,
    components: ServerComponents,
    sessionState: SessionState
  ): void {
    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: getToolDefinitions(),
    }));

    server.setRequestHandler(CallToolRequestSchema, async (request) => {
      return this.handleToolCall(request.params.name, request.params.arguments, components, sessionState);
    });

    server.onerror = (error): void => {
      logError('MCP server error', error);
    };

    server.onclose = (): void => {
      logInfo('MCP server closed');
      void this.cleanupSession(sessionState, components);
    };
  }

  private async handleToolCall(
    toolName: string,
    args: Record<string, unknown> | undefined,
    components: ServerComponents,
    sessionState: SessionState
  ): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    return dispatchToolCall(toolName, args, components, sessionState);
  }

  private createSessionState(): SessionState {
    return {
      sessionId: '',
      projectId: null,
      projectPath: null,
      watchPath: null,
      isWorktree: false,
      heartbeatInterval: null,
      daemonConnected: false,
      cleaned: false,
    };
  }

  private async createHttpSessionServer(): Promise<Server> {
    const sessionState = this.createSessionState();
    const components = buildServerComponents(this.config);
    const { stateManager, daemonClient, projectDetector, healthMonitor } = components;
    const initResult = stateManager.initialize();
    if (initResult.status === 'degraded') {
      logInfo('State manager degraded', { reason: initResult.reason });
    }
    await initializeSession(sessionState, daemonClient, projectDetector, () =>
      startHeartbeat(sessionState, () => sendHeartbeat(sessionState, daemonClient))
    );

    healthMonitor.start();
    logDebug('Health monitoring started');
    await seedDefaultRule(components.rulesTool);
    recordSessionStart();

    const server = this.createMcpServer();
    this.setupHandlers(server, components, sessionState);
    return server;
  }

  async start(): Promise<void> {
    const { stateManager, daemonClient, projectDetector, healthMonitor } = this.components;
    try {
      const initResult = stateManager.initialize();
      if (initResult.status === 'degraded') {
        logInfo('State manager degraded', { reason: initResult.reason });
      }

      healthMonitor.start();
      logDebug('Health monitoring started');

      // Seed default search-first rule on fresh installation
      await this.seedDefaultRule();

      if (this.mode === 'stdio') {
        await initializeSession(this.sessionState, daemonClient, projectDetector, () =>
          startHeartbeat(this.sessionState, () => sendHeartbeat(this.sessionState, daemonClient))
        );
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        logInfo('MCP server started', { mode: 'stdio' });
        recordSessionStart();
      } else if (this.mode === 'http') {
        requireAuth(this.authConfig);
        this.httpHandle = await startMcpHttpServer(
          () => this.createHttpSessionServer(),
          this.httpOptions,
          this.authConfig
        );
        logInfo('MCP server started', {
          mode: 'http',
          host: this.httpOptions.host,
          port: this.httpOptions.port,
          path: this.httpOptions.path,
        });
      } else {
        logInfo('MCP server started', { mode: 'test' });
      }

      this.isInitialized = true;
    } catch (error) {
      logError('Failed to start MCP server', error);
      throw error;
    }
  }

  async stop(): Promise<void> {
    logInfo('Stopping MCP server');
    if (this.mode !== 'http') {
      await this.cleanupSession(this.sessionState, this.components);
    }
    if (this.httpHandle) {
      await stopMcpHttpServer(this.httpHandle);
      this.httpHandle = null;
    }
    await this.server.close();
    logInfo('MCP server stopped');
  }

  getMode(): ServerMode {
    return this.mode;
  }

  /**
   * Tear down all resources for the current session.
   *
   * Safe to call multiple times. Only the first invocation runs cleanup;
   * subsequent calls no-op via the `cleaned` flag (F-049). This prevents
   * double-decrement of the `wqm_mcp_session_count` metric when both the
   * `onclose` handler and `stop()` fire for the same session.
   */
  private async cleanupSession(
    sessionState: SessionState,
    components: ServerComponents
  ): Promise<void> {
    if (sessionState.cleaned) return;
    sessionState.cleaned = true;

    const { daemonClient, stateManager, healthMonitor } = components;
    await cleanup(sessionState, daemonClient, stateManager, healthMonitor);
    recordSessionEnd();
  }

  private async seedDefaultRule(): Promise<void> {
    return seedDefaultRule(this.components.rulesTool);
  }

  // ---- Accessors ----

  getSessionState(): Readonly<SessionState> {
    return { ...this.sessionState };
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  isDaemonConnected(): boolean {
    return this.sessionState.daemonConnected;
  }

  getMcpServer(): Server {
    return this.server;
  }

  getDaemonClient() {
    return this.components.daemonClient;
  }

  getStateManager() {
    return this.components.stateManager;
  }

  getProjectDetector() {
    return this.components.projectDetector;
  }

  getHealthMonitor() {
    return this.components.healthMonitor;
  }
}

/**
 * Resolve the effective transport mode from `ServerOptions`.
 *
 * Precedence: explicit `mode` wins. If omitted, the legacy `stdio` boolean
 * maps `false` to `'test'` and everything else to `'stdio'`.
 */
function resolveMode(options: ServerOptions): ServerMode {
  if (options.mode) {
    return options.mode;
  }
  if (options.stdio === false) {
    return 'test';
  }
  return 'stdio';
}

/**
 * Create and start the MCP server.
 *
 * @param config    Resolved server configuration.
 * @param modeOrStdio  Either a `ServerMode` string, or (legacy) a boolean
 *                     `stdio` flag. `true` → stdio, `false` → test.
 * @param httpOptions  HTTP transport options; required when `modeOrStdio === 'http'`.
 */
export async function createServer(
  config: ServerConfig,
  modeOrStdio: ServerMode | boolean = true,
  httpOptions?: HttpTransportOptions
): Promise<WorkspaceQdrantMcpServer> {
  const options: ServerOptions = { config };
  if (typeof modeOrStdio === 'string') {
    options.mode = modeOrStdio;
  } else {
    options.stdio = modeOrStdio;
  }
  if (httpOptions) {
    options.http = httpOptions;
  }
  const server = new WorkspaceQdrantMcpServer(options);
  await server.start();
  return server;
}
