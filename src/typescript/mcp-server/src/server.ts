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
import { randomUUID } from 'node:crypto';

import { DaemonClient } from './clients/daemon-client.js';
import { SqliteStateManager } from './clients/sqlite-state-manager.js';
import { ProjectDetector } from './utils/project-detector.js';
import { HealthMonitor } from './utils/health-monitor.js';
import {
  setSessionId,
  logInfo,
  logError,
  logDebug,
  logToolCall,
  logSessionEvent,
  logDaemonStatus,
} from './utils/logger.js';
import { SearchTool } from './tools/search.js';
import { RetrieveTool } from './tools/retrieve.js';
import { MemoryTool } from './tools/memory.js';
import { StoreTool } from './tools/store.js';
import type { ServerConfig } from './types/index.js';

// Heartbeat interval: 3 hours (in milliseconds)
const HEARTBEAT_INTERVAL_MS = 3 * 60 * 60 * 1000;

// Server name and version for MCP protocol
import { BUILD_NUMBER } from './build-info.js';
const SERVER_NAME = 'workspace-qdrant-mcp';
const SERVER_VERSION = `0.1.0-beta1 (${BUILD_NUMBER})`;

export interface SessionState {
  sessionId: string;
  projectId: string | null;
  projectPath: string | null;
  heartbeatInterval: ReturnType<typeof setInterval> | null;
  daemonConnected: boolean;
}

export interface ServerOptions {
  config: ServerConfig;
  stdio?: boolean;
}

/**
 * Workspace Qdrant MCP Server
 *
 * Manages the MCP server lifecycle including session management,
 * project registration with the daemon, and heartbeat maintenance.
 */
export class WorkspaceQdrantMcpServer {
  private readonly config: ServerConfig;
  private readonly server: Server;
  private readonly daemonClient: DaemonClient;
  private readonly stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;

  // Tools
  private readonly searchTool: SearchTool;
  private readonly retrieveTool: RetrieveTool;
  private readonly memoryTool: MemoryTool;
  private readonly storeTool: StoreTool;

  // Health monitoring
  private readonly healthMonitor: HealthMonitor;

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
    this.config = options.config;
    this.isStdioMode = options.stdio ?? true;

    const qdrantUrl = this.config.qdrant?.url ?? 'http://localhost:6333';
    const qdrantApiKey = this.config.qdrant?.apiKey;

    // Initialize clients
    this.daemonClient = new DaemonClient({
      port: this.config.daemon.grpcPort,
      timeoutMs: 5000,
    });

    this.stateManager = new SqliteStateManager({
      dbPath: this.config.database.path.replace('~', process.env['HOME'] ?? ''),
    });

    this.projectDetector = new ProjectDetector({
      stateManager: this.stateManager,
    });

    // Build Qdrant config conditionally to satisfy exactOptionalPropertyTypes
    const qdrantConfig: { qdrantUrl: string; qdrantApiKey?: string } = { qdrantUrl };
    if (qdrantApiKey) qdrantConfig.qdrantApiKey = qdrantApiKey;

    // Initialize health monitor
    this.healthMonitor = new HealthMonitor(qdrantConfig, this.daemonClient);

    // Initialize tools (all use daemonClient, stateManager, projectDetector)
    this.searchTool = new SearchTool(
      qdrantConfig,
      this.daemonClient,
      this.stateManager,
      this.projectDetector
    );

    this.retrieveTool = new RetrieveTool(qdrantConfig, this.projectDetector);

    this.memoryTool = new MemoryTool(
      qdrantConfig,
      this.daemonClient,
      this.stateManager,
      this.projectDetector
    );

    // StoreTool is for libraries collection ONLY per spec
    // Project content is handled by daemon file watching, not this tool
    this.storeTool = new StoreTool(
      {},
      this.stateManager
    );

    // Create MCP server
    this.server = new Server(
      {
        name: SERVER_NAME,
        version: SERVER_VERSION,
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  /**
   * Set up MCP protocol handlers
   */
  private setupHandlers(): void {
    // Register tool list handler
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: this.getToolDefinitions(),
    }));

    // Register tool invocation handler
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      return this.handleToolCall(request.params.name, request.params.arguments);
    });

    // Handle server errors
    this.server.onerror = (error): void => {
      logError('MCP server error', error);
    };

    // Handle server close
    this.server.onclose = (): void => {
      logInfo('MCP server closed');
      this.cleanup();
    };
  }

  /**
   * Get tool definitions for ListTools response
   */
  private getToolDefinitions() {
    return [
      {
        name: 'search',
        description: 'Search for documents using hybrid semantic and keyword search',
        inputSchema: {
          type: 'object' as const,
          properties: {
            query: {
              type: 'string',
              description: 'The search query text',
            },
            collection: {
              type: 'string',
              enum: ['projects', 'libraries', 'memory'],
              description: 'Specific collection to search',
            },
            mode: {
              type: 'string',
              enum: ['hybrid', 'semantic', 'keyword'],
              description: 'Search mode (default: hybrid)',
            },
            scope: {
              type: 'string',
              enum: ['project', 'global', 'all'],
              description: 'Search scope: project (current), global, or all (default: project)',
            },
            limit: {
              type: 'number',
              description: 'Maximum results to return (default: 10)',
            },
            projectId: {
              type: 'string',
              description: 'Specific project ID to search',
            },
            libraryName: {
              type: 'string',
              description: 'Library name when searching libraries collection',
            },
            branch: {
              type: 'string',
              description: 'Filter by branch name',
            },
            fileType: {
              type: 'string',
              description: 'Filter by file type',
            },
            includeLibraries: {
              type: 'boolean',
              description: 'Include libraries in search (default: false)',
            },
          },
          required: ['query'],
        },
      },
      {
        name: 'retrieve',
        description: 'Retrieve documents by ID or metadata filter',
        inputSchema: {
          type: 'object' as const,
          properties: {
            documentId: {
              type: 'string',
              description: 'Document ID to retrieve',
            },
            collection: {
              type: 'string',
              enum: ['projects', 'libraries', 'memory'],
              description: 'Collection to retrieve from (default: projects)',
            },
            filter: {
              type: 'object',
              additionalProperties: { type: 'string' },
              description: 'Metadata filter key-value pairs',
            },
            limit: {
              type: 'number',
              description: 'Maximum results (default: 10)',
            },
            offset: {
              type: 'number',
              description: 'Pagination offset (default: 0)',
            },
            projectId: {
              type: 'string',
              description: 'Project ID for projects collection',
            },
            libraryName: {
              type: 'string',
              description: 'Library name for libraries collection',
            },
          },
        },
      },
      {
        name: 'memory',
        description: 'Manage behavioral rules (add, update, remove, list)',
        inputSchema: {
          type: 'object' as const,
          properties: {
            action: {
              type: 'string',
              enum: ['add', 'update', 'remove', 'list'],
              description: 'Action to perform',
            },
            content: {
              type: 'string',
              description: 'Rule content (required for add/update)',
            },
            label: {
              type: 'string',
              description: 'Rule label (max 15 chars, format: word-word-word, e.g., "prefer-uv", "use-pytest"). Required for update/remove.',
            },
            scope: {
              type: 'string',
              enum: ['global', 'project'],
              description: 'Rule scope (default: global)',
            },
            projectId: {
              type: 'string',
              description: 'Project ID for project-scoped rules',
            },
            title: {
              type: 'string',
              description: 'Rule title (max 50 chars)',
            },
            tags: {
              type: 'array',
              items: { type: 'string' },
              description: 'Tags for categorization (max 5 tags, max 20 chars each)',
            },
            priority: {
              type: 'number',
              description: 'Rule priority (higher = more important)',
            },
            limit: {
              type: 'number',
              description: 'Max rules to return for list (default: 50)',
            },
          },
          required: ['action'],
        },
      },
      {
        name: 'store',
        description: 'Store content or register a project. Use type "library" (default) to store reference documentation, or type "project" to register a project directory for file watching and ingestion.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            type: {
              type: 'string',
              enum: ['library', 'project'],
              description: 'What to store: "library" for reference docs (default), "project" to register a project directory',
            },
            content: {
              type: 'string',
              description: 'Content to store (required for type "library")',
            },
            libraryName: {
              type: 'string',
              description: 'Library name (required for type "library")',
            },
            path: {
              type: 'string',
              description: 'Project directory path (required for type "project")',
            },
            name: {
              type: 'string',
              description: 'Project display name (optional for type "project", defaults to directory name)',
            },
            title: {
              type: 'string',
              description: 'Content title (for type "library")',
            },
            url: {
              type: 'string',
              description: 'Source URL (for web content)',
            },
            filePath: {
              type: 'string',
              description: 'Source file path',
            },
            sourceType: {
              type: 'string',
              enum: ['user_input', 'web', 'file', 'scratchbook', 'note'],
              description: 'Source type (default: user_input)',
            },
            metadata: {
              type: 'object',
              additionalProperties: { type: 'string' },
              description: 'Additional metadata',
            },
          },
        },
      },
    ];
  }

  /**
   * Handle tool invocation
   */
  private async handleToolCall(
    toolName: string,
    args: Record<string, unknown> | undefined
  ): Promise<{ content: Array<{ type: string; text: string }>; isError?: boolean }> {
    const startTime = Date.now();

    try {
      let result: unknown;

      switch (toolName) {
        case 'search': {
          const searchResult = await this.searchTool.search(
            this.buildSearchOptions(args)
          );
          // Augment with health status (add success: true to make it compatible)
          result = this.healthMonitor.augmentSearchResults({
            success: true,
            ...searchResult,
          });
          break;
        }

        case 'retrieve': {
          result = await this.retrieveTool.retrieve(this.buildRetrieveOptions(args));
          break;
        }

        case 'memory': {
          result = await this.memoryTool.execute(this.buildMemoryOptions(args));
          break;
        }

        case 'store': {
          const storeType = (args?.['type'] as string) ?? 'library';
          if (storeType === 'project') {
            result = await this.registerProjectFromTool(args);
          } else {
            result = await this.storeTool.store(this.buildStoreOptions(args));
          }
          break;
        }

        default:
          logToolCall(toolName, Date.now() - startTime, false, { error: 'Unknown tool' });
          return {
            content: [{ type: 'text', text: `Unknown tool: ${toolName}` }],
            isError: true,
          };
      }

      logToolCall(toolName, Date.now() - startTime, true);

      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logToolCall(toolName, Date.now() - startTime, false, { error: errorMessage });
      return {
        content: [{ type: 'text', text: `Error: ${errorMessage}` }],
        isError: true,
      };
    }
  }

  /**
   * Build search options from tool arguments
   */
  private buildSearchOptions(args: Record<string, unknown> | undefined): {
    query: string;
    collection?: string;
    mode?: 'hybrid' | 'semantic' | 'keyword';
    scope?: 'project' | 'global' | 'all';
    limit?: number;
    projectId?: string;
    libraryName?: string;
    branch?: string;
    fileType?: string;
    includeLibraries?: boolean;
  } {
    const options: {
      query: string;
      collection?: string;
      mode?: 'hybrid' | 'semantic' | 'keyword';
      scope?: 'project' | 'global' | 'all';
      limit?: number;
      projectId?: string;
      libraryName?: string;
      branch?: string;
      fileType?: string;
      includeLibraries?: boolean;
    } = {
      query: (args?.['query'] as string) ?? '',
    };

    const collection = args?.['collection'] as string | undefined;
    if (collection) options.collection = collection;

    const mode = args?.['mode'] as string | undefined;
    if (mode === 'hybrid' || mode === 'semantic' || mode === 'keyword') {
      options.mode = mode;
    }

    const scope = args?.['scope'] as string | undefined;
    if (scope === 'project' || scope === 'global' || scope === 'all') {
      options.scope = scope;
    }

    const limit = args?.['limit'] as number | undefined;
    if (limit !== undefined) options.limit = limit;

    const projectId = args?.['projectId'] as string | undefined;
    if (projectId) options.projectId = projectId;

    const libraryName = args?.['libraryName'] as string | undefined;
    if (libraryName) options.libraryName = libraryName;

    const branch = args?.['branch'] as string | undefined;
    if (branch) options.branch = branch;

    const fileType = args?.['fileType'] as string | undefined;
    if (fileType) options.fileType = fileType;

    const includeLibraries = args?.['includeLibraries'] as boolean | undefined;
    if (includeLibraries !== undefined) options.includeLibraries = includeLibraries;

    return options;
  }

  /**
   * Build retrieve options from tool arguments
   */
  private buildRetrieveOptions(args: Record<string, unknown> | undefined): {
    documentId?: string;
    collection?: 'projects' | 'libraries' | 'memory';
    filter?: Record<string, string>;
    limit?: number;
    offset?: number;
    projectId?: string;
    libraryName?: string;
  } {
    const options: {
      documentId?: string;
      collection?: 'projects' | 'libraries' | 'memory';
      filter?: Record<string, string>;
      limit?: number;
      offset?: number;
      projectId?: string;
      libraryName?: string;
    } = {};

    const documentId = args?.['documentId'] as string | undefined;
    if (documentId) options.documentId = documentId;

    const collection = args?.['collection'] as string | undefined;
    if (collection === 'projects' || collection === 'libraries' || collection === 'memory') {
      options.collection = collection;
    }

    const filter = args?.['filter'] as Record<string, string> | undefined;
    if (filter) options.filter = filter;

    const limit = args?.['limit'] as number | undefined;
    if (limit !== undefined) options.limit = limit;

    const offset = args?.['offset'] as number | undefined;
    if (offset !== undefined) options.offset = offset;

    const projectId = args?.['projectId'] as string | undefined;
    if (projectId) options.projectId = projectId;

    const libraryName = args?.['libraryName'] as string | undefined;
    if (libraryName) options.libraryName = libraryName;

    return options;
  }

  /**
   * Build memory options from tool arguments
   */
  private buildMemoryOptions(args: Record<string, unknown> | undefined): {
    action: 'add' | 'update' | 'remove' | 'list';
    content?: string;
    label?: string;
    scope?: 'global' | 'project';
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
    limit?: number;
  } {
    const action = args?.['action'] as string;
    if (action !== 'add' && action !== 'update' && action !== 'remove' && action !== 'list') {
      throw new Error(`Invalid memory action: ${action}`);
    }

    const options: {
      action: 'add' | 'update' | 'remove' | 'list';
      content?: string;
      label?: string;
      scope?: 'global' | 'project';
      projectId?: string;
      title?: string;
      tags?: string[];
      priority?: number;
      limit?: number;
    } = { action };

    const content = args?.['content'] as string | undefined;
    if (content) options.content = content;

    const label = args?.['label'] as string | undefined;
    if (label) options.label = label;

    const scope = args?.['scope'] as string | undefined;
    if (scope === 'global' || scope === 'project') {
      options.scope = scope;
    }

    const projectId = args?.['projectId'] as string | undefined;
    if (projectId) options.projectId = projectId;

    const title = args?.['title'] as string | undefined;
    if (title) options.title = title;

    const tags = args?.['tags'] as string[] | undefined;
    if (tags) options.tags = tags;

    const priority = args?.['priority'] as number | undefined;
    if (priority !== undefined) options.priority = priority;

    const limit = args?.['limit'] as number | undefined;
    if (limit !== undefined) options.limit = limit;

    return options;
  }

  /**
   * Build store options from tool arguments
   *
   * Supports both projects and libraries collections.
   * Store tool is for libraries collection ONLY per spec.
   */
  private buildStoreOptions(args: Record<string, unknown> | undefined): {
    content: string;
    libraryName: string;
    title?: string;
    url?: string;
    filePath?: string;
    sourceType?: 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';
    metadata?: Record<string, string>;
  } {
    const content = args?.['content'] as string;
    if (!content) {
      throw new Error('Content is required for store operation');
    }

    const libraryName = args?.['libraryName'] as string;
    if (!libraryName) {
      throw new Error('libraryName is required - store tool is for libraries collection only');
    }

    const options: {
      content: string;
      libraryName: string;
      title?: string;
      url?: string;
      filePath?: string;
      sourceType?: 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';
      metadata?: Record<string, string>;
    } = { content, libraryName };

    const title = args?.['title'] as string | undefined;
    if (title) options.title = title;

    const url = args?.['url'] as string | undefined;
    if (url) options.url = url;

    const filePath = args?.['filePath'] as string | undefined;
    if (filePath) options.filePath = filePath;

    const sourceType = args?.['sourceType'] as string | undefined;
    if (sourceType === 'user_input' || sourceType === 'web' || sourceType === 'file' || sourceType === 'scratchbook' || sourceType === 'note') {
      options.sourceType = sourceType;
    }

    const metadata = args?.['metadata'] as Record<string, string> | undefined;
    if (metadata) options.metadata = metadata;

    return options;
  }

  /**
   * Start the MCP server
   */
  async start(): Promise<void> {
    try {
      // Initialize SQLite state manager
      const initResult = this.stateManager.initialize();
      if (initResult.status === 'degraded') {
        logInfo('State manager degraded', { reason: initResult.reason });
      }

      // Perform session initialization (project detection, daemon registration)
      // Must happen before health monitoring so daemon connection is established
      await this.initializeSession();

      // Start health monitoring after daemon connection attempt
      this.healthMonitor.start();
      logDebug('Health monitoring started');

      // Start the transport
      if (this.isStdioMode) {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        logInfo('MCP server started', { mode: 'stdio' });
      } else {
        // Non-stdio mode: don't connect transport (for testing)
        // In production, HTTP transport would be set up here
        logInfo('MCP server started', { mode: 'test' });
      }

      this.isInitialized = true;
    } catch (error) {
      logError('Failed to start MCP server', error);
      throw error;
    }
  }

  /**
   * Initialize session: detect project, register with daemon, start heartbeat
   */
  private async initializeSession(): Promise<void> {
    // Generate session ID and set for logging correlation
    this.sessionState.sessionId = randomUUID();
    setSessionId(this.sessionState.sessionId);

    logSessionEvent('start', { session_id: this.sessionState.sessionId });

    // Detect current project via longest-prefix matching in daemon's database
    const cwd = process.cwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, true);

    if (projectInfo) {
      this.sessionState.projectPath = projectInfo.projectPath;
      this.sessionState.projectId = projectInfo.projectId;
      logDebug('Project detected', {
        project_path: projectInfo.projectPath,
        project_id: projectInfo.projectId,
      });
    } else {
      logDebug('No project detected from cwd', { cwd });
    }

    // Try to connect to daemon
    try {
      await this.daemonClient.connect();
      this.sessionState.daemonConnected = true;
      logDaemonStatus(true);

      // Register/activate project with daemon (even for new projects without projectId)
      if (this.sessionState.projectPath) {
        await this.registerProject();
      }

      // Start heartbeat
      this.startHeartbeat();
    } catch (error) {
      this.sessionState.daemonConnected = false;
      logDaemonStatus(false, { reason: 'connection_failed' });
      logError('Daemon connection error', error);
    }
  }

  /**
   * Register/activate the current project with the daemon
   *
   * Only re-activates EXISTING projects (register_if_new: false).
   * New projects must be registered explicitly via CLI or other means,
   * preventing accidental registration of home directories or config folders.
   */
  private async registerProject(): Promise<void> {
    if (!this.sessionState.projectPath) {
      return;
    }

    try {
      const response = await this.daemonClient.registerProject({
        path: this.sessionState.projectPath,
        project_id: this.sessionState.projectId ?? '', // Empty for new projects
        name: this.sessionState.projectPath.split('/').pop() ?? 'unknown',
        register_if_new: false, // Never auto-register new projects on session start
      });

      // If the project was not registered (not found and register_if_new=false),
      // log it and continue without error - the MCP server still works without daemon
      if (!response.is_active && !response.created) {
        logInfo('Project not registered with daemon, skipping activation', {
          project_path: this.sessionState.projectPath,
          project_id: response.project_id,
        });
        return;
      }

      // Store the project ID from daemon response (important for existing projects)
      if (response.project_id && !this.sessionState.projectId) {
        this.sessionState.projectId = response.project_id;
        logDebug('Project ID assigned by daemon', { project_id: response.project_id });
      }

      logSessionEvent('register', {
        project_id: response.project_id,
        project_path: this.sessionState.projectPath,
        created: response.created,
        priority: response.priority,
        is_active: response.is_active,
        newly_registered: response.newly_registered,
      });
    } catch (error) {
      logError('Failed to register project', error, {
        project_path: this.sessionState.projectPath,
      });
    }
  }

  /**
   * Register a project via the store tool (type: "project")
   *
   * Unlike session-start registration (register_if_new: false), this explicitly
   * registers new projects with register_if_new: true.
   */
  private async registerProjectFromTool(
    args: Record<string, unknown> | undefined
  ): Promise<{
    success: boolean;
    project_id: string;
    created: boolean;
    is_active: boolean;
    message: string;
  }> {
    const path = args?.['path'] as string;
    if (!path) {
      throw new Error('path is required for store type "project"');
    }

    if (!this.sessionState.daemonConnected) {
      throw new Error('Daemon is not connected — cannot register project');
    }

    const name = (args?.['name'] as string) ?? path.split('/').pop() ?? 'unknown';

    const response = await this.daemonClient.registerProject({
      path,
      project_id: '',
      name,
      register_if_new: true,
    });

    logSessionEvent('register', {
      project_id: response.project_id,
      project_path: path,
      created: response.created,
      priority: response.priority,
      is_active: response.is_active,
      newly_registered: response.newly_registered,
    });

    return {
      success: true,
      project_id: response.project_id,
      created: response.newly_registered,
      is_active: response.is_active,
      message: response.newly_registered
        ? `Project registered and activated: ${path}`
        : `Project already registered and activated: ${path}`,
    };
  }

  /**
   * Start the heartbeat interval
   */
  private startHeartbeat(): void {
    if (this.sessionState.heartbeatInterval) {
      clearInterval(this.sessionState.heartbeatInterval);
    }

    // Send heartbeat immediately
    this.sendHeartbeat();

    // Set up interval for future heartbeats
    this.sessionState.heartbeatInterval = setInterval(() => {
      this.sendHeartbeat();
    }, HEARTBEAT_INTERVAL_MS);

    logDebug('Heartbeat started', { interval_minutes: HEARTBEAT_INTERVAL_MS / 1000 / 60 });
  }

  /**
   * Send a heartbeat to the daemon
   */
  private async sendHeartbeat(): Promise<void> {
    if (!this.sessionState.projectId || !this.sessionState.daemonConnected) {
      return;
    }

    try {
      const response = await this.daemonClient.heartbeat({
        project_id: this.sessionState.projectId,
      });

      if (response.acknowledged) {
        logSessionEvent('heartbeat', {
          project_id: this.sessionState.projectId,
          acknowledged: true,
        });
      }
    } catch (error) {
      logError('Heartbeat failed', error, { project_id: this.sessionState.projectId });
      // Mark daemon as disconnected on heartbeat failure
      this.sessionState.daemonConnected = false;
      logDaemonStatus(false, { reason: 'heartbeat_failed' });
    }
  }

  /**
   * Stop the MCP server gracefully
   */
  async stop(): Promise<void> {
    logInfo('Stopping MCP server');
    await this.cleanup();
    await this.server.close();
    logInfo('MCP server stopped');
  }

  /**
   * Clean up resources on session end
   */
  private async cleanup(): Promise<void> {
    // Stop health monitoring
    this.healthMonitor.stop();
    logDebug('Health monitoring stopped');

    // Stop heartbeat
    if (this.sessionState.heartbeatInterval) {
      clearInterval(this.sessionState.heartbeatInterval);
      this.sessionState.heartbeatInterval = null;
      logDebug('Heartbeat stopped');
    }

    // Deprioritize project with daemon
    if (this.sessionState.projectId && this.sessionState.daemonConnected) {
      try {
        const response = await this.daemonClient.deprioritizeProject({
          project_id: this.sessionState.projectId,
        });
        logSessionEvent('deprioritize', {
          project_id: this.sessionState.projectId,
          is_active: response.is_active,
          new_priority: response.new_priority,
        });
      } catch (error) {
        logError('Failed to deprioritize project', error, {
          project_id: this.sessionState.projectId,
        });
      }
    }

    // Close daemon client
    this.daemonClient.close();

    // Close state manager
    this.stateManager.close();

    logSessionEvent('end', { session_id: this.sessionState.sessionId });
  }

  /**
   * Get current session state
   */
  getSessionState(): Readonly<SessionState> {
    return { ...this.sessionState };
  }

  /**
   * Check if server is initialized
   */
  isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Check if daemon is connected
   */
  isDaemonConnected(): boolean {
    return this.sessionState.daemonConnected;
  }

  /**
   * Get the underlying MCP Server instance (for tool registration)
   */
  getMcpServer(): Server {
    return this.server;
  }

  /**
   * Get the daemon client (for tools that need daemon access)
   */
  getDaemonClient(): DaemonClient {
    return this.daemonClient;
  }

  /**
   * Get the state manager (for tools that need SQLite access)
   */
  getStateManager(): SqliteStateManager {
    return this.stateManager;
  }

  /**
   * Get the project detector (for project resolution)
   */
  getProjectDetector(): ProjectDetector {
    return this.projectDetector;
  }

  /**
   * Get the health monitor (for health status)
   */
  getHealthMonitor(): HealthMonitor {
    return this.healthMonitor;
  }
}

/**
 * Create and start the MCP server
 */
export async function createServer(config: ServerConfig, stdio = true): Promise<WorkspaceQdrantMcpServer> {
  const server = new WorkspaceQdrantMcpServer({ config, stdio });
  await server.start();
  return server;
}
