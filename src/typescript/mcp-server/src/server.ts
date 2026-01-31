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
import { randomUUID } from 'node:crypto';

import { DaemonClient } from './clients/daemon-client.js';
import { SqliteStateManager } from './clients/sqlite-state-manager.js';
import { ProjectDetector } from './utils/project-detector.js';
import type { ServerConfig } from './types/index.js';

// Heartbeat interval: 3 hours (in milliseconds)
const HEARTBEAT_INTERVAL_MS = 3 * 60 * 60 * 1000;

// Server name and version for MCP protocol
const SERVER_NAME = 'workspace-qdrant-mcp';
const SERVER_VERSION = '0.1.0';

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
    // Tool handlers will be registered when tools are implemented
    // For now, just log the capability

    // Handle server errors
    this.server.onerror = (error): void => {
      this.logError('MCP server error:', error);
    };

    // Handle server close
    this.server.onclose = (): void => {
      this.log('MCP server closed');
      this.cleanup();
    };
  }

  /**
   * Start the MCP server
   */
  async start(): Promise<void> {
    try {
      // Initialize SQLite state manager
      const initResult = this.stateManager.initialize();
      if (initResult.status === 'degraded') {
        this.log(`State manager degraded: ${initResult.reason}`);
      }

      // Perform session initialization (project detection, daemon registration)
      await this.initializeSession();

      // Start the transport
      if (this.isStdioMode) {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        this.log('MCP server started (stdio mode)');
      } else {
        // Non-stdio mode: don't connect transport (for testing)
        // In production, HTTP transport would be set up here
        this.log('MCP server started (no transport - test mode)');
      }

      this.isInitialized = true;
    } catch (error) {
      this.logError('Failed to start MCP server:', error);
      throw error;
    }
  }

  /**
   * Initialize session: detect project, register with daemon, start heartbeat
   */
  private async initializeSession(): Promise<void> {
    // Generate session ID
    this.sessionState.sessionId = randomUUID();
    this.log(`Session initialized: ${this.sessionState.sessionId}`);

    // Detect current project
    const cwd = process.cwd();
    const projectRoot = this.projectDetector.findProjectRoot(cwd);

    if (projectRoot) {
      this.sessionState.projectPath = projectRoot;
      this.log(`Project detected: ${projectRoot}`);

      // Try to get project info from daemon's database
      const projectInfo = await this.projectDetector.getProjectInfo(projectRoot, true);
      if (projectInfo) {
        this.sessionState.projectId = projectInfo.projectId;
        this.log(`Project ID: ${projectInfo.projectId}`);
      }
    } else {
      this.log('No project detected from cwd');
    }

    // Try to connect to daemon
    try {
      await this.daemonClient.connect();
      this.sessionState.daemonConnected = true;
      this.log('Connected to daemon');

      // Register project with daemon
      if (this.sessionState.projectPath && this.sessionState.projectId) {
        await this.registerProject();
      }

      // Start heartbeat
      this.startHeartbeat();
    } catch (error) {
      this.sessionState.daemonConnected = false;
      this.log('Daemon not available, running in degraded mode');
      this.logError('Daemon connection error:', error);
    }
  }

  /**
   * Register the current project with the daemon
   */
  private async registerProject(): Promise<void> {
    if (!this.sessionState.projectPath || !this.sessionState.projectId) {
      return;
    }

    try {
      const response = await this.daemonClient.registerProject({
        path: this.sessionState.projectPath,
        project_id: this.sessionState.projectId,
        name: this.sessionState.projectPath.split('/').pop() ?? 'unknown',
      });

      this.log(
        `Project registered: ${response.created ? 'new' : 'existing'}, ` +
          `priority=${response.priority}, sessions=${response.active_sessions}`
      );
    } catch (error) {
      this.logError('Failed to register project:', error);
    }
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

    this.log(`Heartbeat started (interval: ${HEARTBEAT_INTERVAL_MS / 1000 / 60} minutes)`);
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
        this.log('Heartbeat acknowledged');
      }
    } catch (error) {
      this.logError('Heartbeat failed:', error);
      // Mark daemon as disconnected on heartbeat failure
      this.sessionState.daemonConnected = false;
    }
  }

  /**
   * Stop the MCP server gracefully
   */
  async stop(): Promise<void> {
    this.log('Stopping MCP server...');
    await this.cleanup();
    await this.server.close();
    this.log('MCP server stopped');
  }

  /**
   * Clean up resources on session end
   */
  private async cleanup(): Promise<void> {
    // Stop heartbeat
    if (this.sessionState.heartbeatInterval) {
      clearInterval(this.sessionState.heartbeatInterval);
      this.sessionState.heartbeatInterval = null;
      this.log('Heartbeat stopped');
    }

    // Deprioritize project with daemon
    if (this.sessionState.projectId && this.sessionState.daemonConnected) {
      try {
        const response = await this.daemonClient.deprioritizeProject({
          project_id: this.sessionState.projectId,
        });
        this.log(
          `Project deprioritized: remaining_sessions=${response.remaining_sessions}, ` +
            `new_priority=${response.new_priority}`
        );
      } catch (error) {
        this.logError('Failed to deprioritize project:', error);
      }
    }

    // Close daemon client
    this.daemonClient.close();

    // Close state manager
    this.stateManager.close();

    this.log(`Session ended: ${this.sessionState.sessionId}`);
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
   * Log a message (stderr in stdio mode to avoid protocol contamination)
   */
  private log(message: string): void {
    if (!this.isStdioMode) {
      console.log(`[${SERVER_NAME}] ${message}`);
    } else {
      console.error(`[${SERVER_NAME}] ${message}`);
    }
  }

  /**
   * Log an error (always to stderr)
   */
  private logError(message: string, error: unknown): void {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`[${SERVER_NAME}] ${message} ${errorMessage}`);
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
