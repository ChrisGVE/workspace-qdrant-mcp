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
import { resolveBodyCwdOverride, runWithRequestContext } from './utils/request-context.js';
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
    currentBranch: null,
    lastBranchRefreshAt: 0,
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
          "This server exposes the user's indexed codebase, libraries, behavioral rules, scratchpad, and project/branch registry.",
          'Start of session: call `rules` with action="list" to load behavioral preferences before any non-trivial work. When you discover a durable project convention (build/test commands, preferred libraries, patterns to follow or avoid), record it with `rules` action="add" so it persists — reserve the scratchpad for one-off task context.',
          'Discovery — call `search` FIRST for any question about this codebase, project structure, or library docs; do not answer from training data. Defaults: scope="project", limit=10. Widen to scope="all" or includeLibraries=true only after a project-scoped query returns nothing useful. Use mode="semantic" for concept queries; for a known identifier or exact string use exact=true (FTS5 substring) or the `grep` tool — NOT mode="keyword", which is sparse BM25 relevance and will not reliably match a verbatim token.',
          'Query formulation — write `search` queries in ENGLISH regardless of the conversation language: the embedding model is English-only, so non-English queries match same-language docs instead of code and recall collapses. Use vocabulary close to the expected identifiers/comments. When you want the implementation rather than docs or tests, add fileType="code" (other values: docs, text, config, data, build, web, slides) or a pathGlob like "src/**/*.rs" — documentation and test files often outrank the implementation otherwise.',
          'Exact lookups — use `grep` for regex / exact substring across the project (faster and cheaper than `search` with exact=true for known strings). Use `list` (start with format="summary") to understand layout before drilling in. Use `retrieve` when you already know the document ID/metadata — do not re-search.',
          'Project context — `search`, `grep`, `list`, `retrieve`, and `rules` auto-detect the current project from your working directory. Over HTTP the server cannot observe it, so pass your absolute working directory in the `cwd` argument on each such call (unless you already pass an explicit `projectId`). Omitting both can yield "Could not detect project".',
          'Writes — `store` writes to `scratchpad` (notes, snippets) or `libraries` (only when the user explicitly asks). The server does NOT write project code to the `projects` collection — that is daemon-owned via file watching. To register/activate a project, use `store` with type="project".',
          'Project memory (scratchpad) — as you work, proactively record durable project knowledge with `store` type="scratchpad": decisions and their rationale, non-obvious gotchas, conventions, and anything worth recalling in a later session. Keep each note self-contained. Notes are project-scoped and resurface AUTOMATICALLY — the project-scoped `search` recall lane appends the most relevant notes after the code hits, so you need not query the scratchpad explicitly. To revise or remove a note, use the `scratchpad` tool (update/delete) rather than creating near-duplicates.',
          'Code intelligence (graph) — for relationship questions ("what calls X", "what breaks if I change Y") and before refactoring or renaming a widely-used symbol, use the `graph` tool (`impact` for blast radius; relationship queries for callers/dependencies) rather than guessing — it surfaces edges that `search`/`grep` miss.',
          'Embeddings — `embedding` is a low-level helper; prefer `search` unless you specifically need a raw vector.',
          'Branches & worktrees — project registration is automatic on session start; the server tracks the current branch via heartbeat. Use `workspace_index` for observability (read-only actions: list_projects, project_status, status_all, list_branches, agent_branch_status, observe_*, incremental_check*). `search` defaults to the current branch; pass `branch="<name>"` or `branch="*"` to widen explicitly — do not widen silently. When working on an agent/feature branch (especially in a parallel worktree), register it via `start_agent_branch` with `branchName`, `purpose`, `createdBy`, and `useWorktree=true` if applicable; close out with `finish_agent_branch` (merged) or `abandon_agent_branch` (discarded). Mutating actions require DOUBLE opt-in (allowMutation:true AND WQM_INDEX_MANAGER_ALLOW_MUTATION=1) and explicit user confirmation, because they affect persistent shared state. `sync_current_branch` is for git hooks only — agents must not call it. Multi-clone: tenant_ids are stable per clone; if results come from the wrong clone, pass `projectId` explicitly.',
          'Collections: projects (indexed code, daemon-written), libraries (reference docs), rules (behavioral rules), scratchpad (ad-hoc notes).',
          'Budget: default to scope="project" with small limits; escalate only when needed.',
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
    // Body-level host-CWD fallback. The HTTP transport binds the host cwd from
    // the `x-mcp-host-cwd` header, which always wins. But a client may be
    // unable to send that header per session (e.g. Claude Code over HTTP has no
    // dynamic header for the cwd). In that case an agent can pass its working
    // directory in the tool's `cwd` argument; bind it into the request context
    // so getEffectiveCwd() — and thus project auto-detection — picks it up
    // exactly as if it were the header. Precedence stays:
    //   header > body `cwd` > WQM_DEFAULT_HOST_CWD > process.cwd().
    const override = resolveBodyCwdOverride(
      typeof args?.['cwd'] === 'string' ? (args['cwd'] as string) : undefined
    );
    if (override) {
      return runWithRequestContext({ hostCwd: override }, () =>
        dispatchToolCall(toolName, args, components, sessionState)
      );
    }
    return dispatchToolCall(toolName, args, components, sessionState);
  }

  private createSessionState(): SessionState {
    return {
      sessionId: '',
      projectId: null,
      projectPath: null,
      watchPath: null,
      isWorktree: false,
      currentBranch: null,
      lastBranchRefreshAt: 0,
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
          this.authConfig,
          // Admin UI deps. Mounting only on the http transport (stdio
          // doesn't expose HTTP routes). Reuses the same daemonClient
          // and stateManager that the MCP tools already use, so admin
          // CRUD and the agent observe identical state.
          {
            daemonClient: this.components.daemonClient,
            stateManager: this.components.stateManager,
            searchDbReader: this.components.searchDbReader,
            rulesTool: this.components.rulesTool,
            authConfig: this.authConfig,
          }
        );
        logInfo('MCP server started', {
          mode: 'http',
          host: this.httpOptions.host,
          port: this.httpOptions.port,
          path: this.httpOptions.path,
        });
      } else {
        // test mode: run full session init (daemon + transport are mocked in
        // tests) but do not bind a real stdio/http transport.
        await initializeSession(this.sessionState, daemonClient, projectDetector, () =>
          startHeartbeat(this.sessionState, () => sendHeartbeat(this.sessionState, daemonClient))
        );
        logInfo('MCP server started', { mode: 'test' });
        recordSessionStart();
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
    try {
      await cleanup(sessionState, daemonClient, stateManager, healthMonitor);
    } finally {
      // Decrement unconditionally. If cleanup() ever throws, skipping this would
      // leak the wqm_mcp_session_count gauge upward with no way to recover (the
      // `cleaned` guard above blocks any retry). The decrement must survive a
      // failed teardown.
      recordSessionEnd();
    }
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
