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
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import type { ServerConfig } from './types/index.js';
import { logInfo, logError, logDebug, logToolCall } from './utils/logger.js';
import { SERVER_NAME, SERVER_VERSION } from './server-types.js';
import type { SessionState, ServerOptions } from './server-types.js';
export type { SessionState, ServerOptions } from './server-types.js';

import { buildServerComponents } from './server-factory.js';
import type { ServerComponents } from './server-factory.js';
import { getToolDefinitions } from './tool-definitions.js';
import {
  buildSearchOptions,
  buildRetrieveOptions,
  buildRuleOptions,
  buildStoreOptions,
  buildGrepOptions,
} from './tool-builders.js';
import { storeUrl, storeScratchpad } from './store-handlers.js';
import {
  initializeSession,
  registerProjectFromTool,
  startHeartbeat,
  sendHeartbeat,
  cleanup,
} from './session-lifecycle.js';

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
    heartbeatInterval: null,
    daemonConnected: false,
  };

  private isStdioMode: boolean;
  private isInitialized = false;

  constructor(options: ServerOptions) {
    this.isStdioMode = options.stdio ?? true;
    this.components = buildServerComponents(options.config);

    this.server = new Server(
      { name: SERVER_NAME, version: SERVER_VERSION },
      {
        capabilities: { tools: {} },
        instructions: [
          'This server provides access to the user\'s indexed codebase and knowledge libraries.',
          'ALWAYS use the `search` tool before answering questions about the user\'s code, project structure, or library documentation.',
          'Use the `rules` tool to check for behavioral preferences before starting work.',
          'Use `retrieve` to access specific documents when you know the document ID.',
          'Collections: projects (indexed code), libraries (reference docs), memory (behavioral rules).',
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
    const { searchTool, retrieveTool, rulesTool, storeTool, grepTool, healthMonitor, daemonClient, stateManager } = this.components;

    // Implicit heartbeat — fire-and-forget to avoid latency
    sendHeartbeat(this.sessionState, daemonClient);

    try {
      let result: unknown;

      switch (toolName) {
        case 'search': {
          const searchResult = await searchTool.search(buildSearchOptions(args));
          result = healthMonitor.augmentSearchResults({ success: true, ...searchResult });
          break;
        }
        case 'retrieve':
          result = await retrieveTool.retrieve(buildRetrieveOptions(args));
          break;
        case 'rules':
          result = await rulesTool.execute(buildRuleOptions(args));
          break;
        case 'store': {
          const storeType = (args?.['type'] as string) ?? 'library';
          if (storeType === 'project') {
            result = await registerProjectFromTool(args, this.sessionState, daemonClient);
          } else if (storeType === 'url') {
            result = await storeUrl(args, stateManager, this.sessionState);
          } else if (storeType === 'scratchpad') {
            result = await storeScratchpad(args, stateManager, this.sessionState);
          } else {
            result = await storeTool.store(buildStoreOptions(args, this.sessionState));
          }
          break;
        }
        case 'grep':
          result = await grepTool.grep(buildGrepOptions(args));
          break;
        default:
          logToolCall(toolName, Date.now() - startTime, false, { error: 'Unknown tool' });
          return { content: [{ type: 'text', text: `Unknown tool: ${toolName}` }], isError: true };
      }

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

      await initializeSession(
        this.sessionState,
        daemonClient,
        projectDetector,
        () => startHeartbeat(this.sessionState, () => sendHeartbeat(this.sessionState, daemonClient))
      );

      healthMonitor.start();
      logDebug('Health monitoring started');

      // Seed default search-first rule on fresh installation
      await this.seedDefaultRule();

      if (this.isStdioMode) {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        logInfo('MCP server started', { mode: 'stdio' });
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
    await this.cleanupSession();
    await this.server.close();
    logInfo('MCP server stopped');
  }

  private async cleanupSession(): Promise<void> {
    const { daemonClient, stateManager, healthMonitor } = this.components;
    await cleanup(this.sessionState, daemonClient, stateManager, healthMonitor);
  }

  /**
   * Seed a default "search-first" rule if the rules collection is empty.
   * Only runs once per fresh installation; skipped if any rule already exists.
   */
  private async seedDefaultRule(): Promise<void> {
    const { rulesTool } = this.components;
    try {
      const listResult = await rulesTool.execute({ action: 'list', scope: 'global' });
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
        scope: 'global',
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
 * Create and start the MCP server
 */
export async function createServer(
  config: ServerConfig,
  stdio = true
): Promise<WorkspaceQdrantMcpServer> {
  const server = new WorkspaceQdrantMcpServer({ config, stdio });
  await server.start();
  return server;
}
