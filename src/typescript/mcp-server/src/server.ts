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
import { logInfo, logError, logDebug, logToolCall } from './utils/logger.js';
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
import { TENANT_GLOBAL } from './constants/tenants.js';
import type { ServerComponents } from './server-factory.js';
import { getToolDefinitions } from './tool-definitions/index.js';
import {
  buildSearchOptions,
  buildRetrieveOptions,
  buildRuleOptions,
  buildStoreOptions,
  buildGrepOptions,
  buildListOptions,
} from './tool-builders/index.js';
import { storeUrl, storeScratchpad } from './store-handlers.js';
import {
  initializeSession,
  registerProjectFromTool,
  startHeartbeat,
  sendHeartbeat,
  cleanup,
} from './session-lifecycle.js';
import { withToolMetrics, recordSessionStart, recordSessionEnd } from './telemetry/metrics.js';

/**
 * Workspace Qdrant MCP Server
 *
 * Manages the MCP server lifecycle including session management,
 * project registration with the daemon, and heartbeat maintenance.
 */
export class WorkspaceQdrantMcpServer {
  private readonly server: Server;
  private readonly components: ServerComponents;

  private sessionState: SessionState = {
    sessionId: '',
    projectId: null,
    projectPath: null,
    watchPath: null,
    isWorktree: false,
    heartbeatInterval: null,
    daemonConnected: false,
  };

  private readonly mode: ServerMode;
  private readonly httpOptions: HttpTransportOptions;
  private readonly authConfig: AuthConfig;
  private httpHandle: McpHttpServerHandle | null = null;
  private isInitialized = false;

  constructor(options: ServerOptions) {
    this.mode = resolveMode(options);
    this.httpOptions = {
      host: options.http?.host ?? DEFAULT_HTTP_HOST,
      port: options.http?.port ?? DEFAULT_HTTP_PORT,
      path: options.http?.path ?? DEFAULT_HTTP_PATH,
    };
    this.authConfig = options.auth ?? loadAuthConfig();
    this.components = buildServerComponents(options.config);

    this.server = new Server(
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

    this.setupHandlers();
  }

  private setupHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: getToolDefinitions(),
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      return this.handleToolCall(request.params.name, request.params.arguments);
    });

    this.server.onerror = (error): void => {
      logError('MCP server error', error);
    };

    this.server.onclose = (): void => {
      logInfo('MCP server closed');
      this.cleanupSession();
    };
  }

  private async handleToolCall(
    toolName: string,
    args: Record<string, unknown> | undefined
  ): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    const startTime = Date.now();
    const {
      searchTool,
      retrieveTool,
      rulesTool,
      storeTool,
      grepTool,
      listTool,
      healthMonitor,
      daemonClient,
      stateManager,
    } = this.components;

    // Implicit heartbeat — fire-and-forget to avoid latency
    sendHeartbeat(this.sessionState, daemonClient);

    // Unknown tool check — outside metrics wrapper to avoid recording unknown names
    const knownTools = ['search', 'retrieve', 'rules', 'store', 'grep', 'list'];
    if (!knownTools.includes(toolName)) {
      logToolCall(toolName, Date.now() - startTime, false, { error: 'Unknown tool' });
      return { content: [{ type: 'text', text: `Unknown tool: ${toolName}` }], isError: true };
    }

    try {
      const result = await withToolMetrics(toolName, async () => {
        switch (toolName) {
          case 'search': {
            const searchResult = await searchTool.search(buildSearchOptions(args));
            return healthMonitor.augmentSearchResults({ success: true, ...searchResult });
          }
          case 'retrieve':
            return retrieveTool.retrieve(buildRetrieveOptions(args));
          case 'rules':
            return rulesTool.execute(buildRuleOptions(args));
          case 'store': {
            const storeType = (args?.['type'] as string) ?? 'library';
            if (storeType === 'project') {
              return registerProjectFromTool(args, this.sessionState, daemonClient);
            } else if (storeType === 'url') {
              return storeUrl(args, stateManager, this.sessionState);
            } else if (storeType === 'scratchpad') {
              return storeScratchpad(args, stateManager, this.sessionState);
            } else {
              return storeTool.store(buildStoreOptions(args, this.sessionState));
            }
          }
          case 'grep':
            return grepTool.grep(buildGrepOptions(args));
          case 'list':
            return listTool.list(buildListOptions(args));
          default:
            // Unreachable: knownTools guard above covers all cases
            throw new Error(`Unexpected tool: ${toolName}`);
        }
      });

      logToolCall(toolName, Date.now() - startTime, true);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logToolCall(toolName, Date.now() - startTime, false, { error: errorMessage });
      return { content: [{ type: 'text', text: `Error: ${errorMessage}` }], isError: true };
    }
  }

  async start(): Promise<void> {
    const { stateManager, daemonClient, projectDetector, healthMonitor } = this.components;
    try {
      const initResult = stateManager.initialize();
      if (initResult.status === 'degraded') {
        logInfo('State manager degraded', { reason: initResult.reason });
      }

      await initializeSession(this.sessionState, daemonClient, projectDetector, () =>
        startHeartbeat(this.sessionState, () => sendHeartbeat(this.sessionState, daemonClient))
      );

      healthMonitor.start();
      logDebug('Health monitoring started');

      // Seed default search-first rule on fresh installation
      await this.seedDefaultRule();

      if (this.mode === 'stdio') {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        logInfo('MCP server started', { mode: 'stdio' });
      } else if (this.mode === 'http') {
        requireAuth(this.authConfig);
        this.httpHandle = await startMcpHttpServer(this.server, this.httpOptions, this.authConfig);
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
      recordSessionStart();
    } catch (error) {
      logError('Failed to start MCP server', error);
      throw error;
    }
  }

  async stop(): Promise<void> {
    logInfo('Stopping MCP server');
    await this.cleanupSession();
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

  private async cleanupSession(): Promise<void> {
    const { daemonClient, stateManager, healthMonitor } = this.components;
    await cleanup(this.sessionState, daemonClient, stateManager, healthMonitor);
    recordSessionEnd();
  }

  /**
   * Seed a default "search-first" rule if the rules collection is empty.
   * Only runs once per fresh installation; skipped if any rule already exists.
   */
  private async seedDefaultRule(): Promise<void> {
    const { rulesTool } = this.components;
    try {
      const listResult = await rulesTool.execute({ action: 'list', scope: TENANT_GLOBAL });
      if (!listResult.success || (listResult.rules && listResult.rules.length > 0)) {
        return; // Rules exist or list failed — skip seeding
      }

      const addResult = await rulesTool.execute({
        action: 'add',
        label: 'search-first',
        title: 'Always search before answering',
        content: [
          'When asked about the codebase, project structure, library documentation,',
          'or any topic that might be covered in the indexed knowledge base,',
          'ALWAYS use the workspace-qdrant search tool first.',
          'Do not rely on training data for project-specific questions.',
          'Use scope="project" for code questions and includeLibraries=true for broader knowledge queries.',
        ].join(' '),
        scope: TENANT_GLOBAL,
        priority: 100,
      });

      if (addResult.success) {
        logInfo('Created default search-first behavioral rule');
      }
    } catch (error) {
      logDebug('Skipped default rule seeding', { reason: String(error) });
    }
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
