/**
 * Tests for WorkspaceQdrantMcpServer session lifecycle
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer, type SessionState } from '../src/server.js';
import type { ServerConfig } from '../src/types/index.js';

// Mock the DaemonClient to avoid actual gRPC connections
vi.mock('../src/clients/daemon-client.js', () => ({
  DaemonClient: vi.fn().mockImplementation(() => ({
    connect: vi.fn().mockRejectedValue(new Error('Mock: daemon not available')),
    close: vi.fn(),
    isConnected: vi.fn().mockReturnValue(false),
    registerProject: vi.fn().mockResolvedValue({
      created: true,
      project_id: 'mock_project_id',
      priority: 'high',
      active_sessions: 1,
    }),
    deprioritizeProject: vi.fn().mockResolvedValue({
      success: true,
      remaining_sessions: 0,
      new_priority: 'normal',
    }),
    heartbeat: vi.fn().mockResolvedValue({
      acknowledged: true,
    }),
  })),
}));

// Mock the MCP SDK Server to avoid actual protocol handling
vi.mock('@modelcontextprotocol/sdk/server/index.js', () => ({
  Server: vi.fn().mockImplementation(() => ({
    connect: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
    onerror: null,
    onclose: null,
  })),
}));

vi.mock('@modelcontextprotocol/sdk/server/stdio.js', () => ({
  StdioServerTransport: vi.fn().mockImplementation(() => ({})),
}));

// Create test schema (minimal version matching daemon)
const TEST_SCHEMA = `
CREATE TABLE IF NOT EXISTS registered_projects (
    project_id TEXT PRIMARY KEY,
    project_path TEXT NOT NULL UNIQUE,
    git_remote_url TEXT,
    remote_hash TEXT,
    disambiguation_path TEXT,
    container_folder TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    last_seen_at TEXT,
    last_activity_at TEXT
);
`;

function createTestConfig(tempDir: string): ServerConfig {
  return {
    database: {
      path: join(tempDir, 'state.db'),
    },
    qdrant: {
      url: 'http://localhost:6333',
      timeout: 5000,
    },
    daemon: {
      grpcPort: 50051,
      queuePollIntervalMs: 1000,
      queueBatchSize: 10,
    },
    watching: {
      patterns: ['*.ts'],
      ignorePatterns: ['node_modules/*'],
    },
    collections: {
      memoryCollectionName: 'memory',
    },
    environment: {},
  };
}

describe('WorkspaceQdrantMcpServer', () => {
  let tempDir: string;
  let dbPath: string;
  let config: ServerConfig;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-server-test-'));
    dbPath = join(tempDir, 'state.db');

    // Create database with test schema
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();

    config = createTestConfig(tempDir);
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create server instance with config', () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });

      expect(server).toBeDefined();
      expect(server.isReady()).toBe(false);
    });

    it('should default to stdio mode', () => {
      const server = new WorkspaceQdrantMcpServer({ config });

      // Can't directly test the mode, but the server should be created
      expect(server).toBeDefined();
    });
  });

  describe('getSessionState', () => {
    it('should return initial session state', () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      const state = server.getSessionState();

      expect(state.sessionId).toBe('');
      expect(state.projectId).toBeNull();
      expect(state.projectPath).toBeNull();
      expect(state.heartbeatInterval).toBeNull();
      expect(state.daemonConnected).toBe(false);
    });
  });

  describe('start', () => {
    it('should initialize session with unique session ID', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });

      await server.start();

      const state = server.getSessionState();
      expect(state.sessionId).toBeDefined();
      expect(state.sessionId.length).toBe(36); // UUID format
      expect(server.isReady()).toBe(true);

      await server.stop();
    });

    it('should detect project from cwd', async () => {
      // Create a project directory with .git
      const projectPath = join(tempDir, 'my-project');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));

      // Resolve real path (handles macOS /private symlink)
      const realProjectPath = realpathSync(projectPath);

      // Change cwd to project
      const originalCwd = process.cwd();
      process.chdir(projectPath);

      try {
        const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        const state = server.getSessionState();
        // Use realpath for comparison since path.resolve may return symlinked path
        expect(state.projectPath).toBe(realProjectPath);

        await server.stop();
      } finally {
        process.chdir(originalCwd);
      }
    });

    it('should handle missing project gracefully', async () => {
      // Use a temp directory without project markers
      const originalCwd = process.cwd();
      process.chdir(tempDir);

      try {
        const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        const state = server.getSessionState();
        expect(state.projectPath).toBeNull();

        await server.stop();
      } finally {
        process.chdir(originalCwd);
      }
    });

    it('should handle daemon unavailable gracefully', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });

      await server.start();

      // Server should start even if daemon is not available
      expect(server.isReady()).toBe(true);
      expect(server.isDaemonConnected()).toBe(false);

      await server.stop();
    });
  });

  describe('stop', () => {
    it('should clean up resources on stop', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      await server.stop();

      // After stop, the server should be cleaned up
      expect(server.isReady()).toBe(true); // isReady doesn't change on stop
    });

    it('should clear heartbeat interval on stop', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const stateBefore = server.getSessionState();
      // Heartbeat won't be running since daemon is not connected

      await server.stop();

      const stateAfter = server.getSessionState();
      expect(stateAfter.heartbeatInterval).toBeNull();
    });
  });

  describe('accessor methods', () => {
    it('should provide access to daemon client', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const client = server.getDaemonClient();
      expect(client).toBeDefined();

      await server.stop();
    });

    it('should provide access to state manager', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const manager = server.getStateManager();
      expect(manager).toBeDefined();

      await server.stop();
    });

    it('should provide access to project detector', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const detector = server.getProjectDetector();
      expect(detector).toBeDefined();

      await server.stop();
    });

    it('should provide access to MCP server', async () => {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const mcpServer = server.getMcpServer();
      expect(mcpServer).toBeDefined();

      await server.stop();
    });
  });
});

describe('Session lifecycle with connected daemon', () => {
  let tempDir: string;
  let realTempDir: string;
  let config: ServerConfig;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-daemon-test-'));
    realTempDir = realpathSync(tempDir);
    const dbPath = join(tempDir, 'state.db');

    // Create database
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();

    config = createTestConfig(tempDir);

    // Reset mocks
    vi.clearAllMocks();
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('should register project with daemon when project detected', async () => {
    // Mock successful daemon connection
    const { DaemonClient } = await import('../src/clients/daemon-client.js');
    const mockInstance = {
      connect: vi.fn().mockResolvedValue(undefined),
      close: vi.fn(),
      isConnected: vi.fn().mockReturnValue(true),
      registerProject: vi.fn().mockResolvedValue({
        created: true,
        project_id: 'abc123456789',
        priority: 'high',
        active_sessions: 1,
      }),
      deprioritizeProject: vi.fn().mockResolvedValue({
        success: true,
        remaining_sessions: 0,
        new_priority: 'normal',
      }),
      heartbeat: vi.fn().mockResolvedValue({
        acknowledged: true,
      }),
    };
    vi.mocked(DaemonClient).mockImplementation(() => mockInstance as unknown as InstanceType<typeof DaemonClient>);

    // Create project with database entry
    const projectPath = join(tempDir, 'test-project');
    mkdirSync(projectPath);
    mkdirSync(join(projectPath, '.git'));

    // Use realpath for the database entry since the server will resolve paths
    const realProjectPath = realpathSync(projectPath);

    const dbPath = join(tempDir, 'state.db');
    const db = new Database(dbPath);
    db.prepare(`
      INSERT INTO registered_projects
      (project_id, project_path, container_folder, is_active, created_at)
      VALUES ('abc123456789', ?, 'test-project', 1, datetime('now'))
    `).run(realProjectPath);
    db.close();

    const originalCwd = process.cwd();
    process.chdir(projectPath);

    try {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const state = server.getSessionState();
      expect(state.daemonConnected).toBe(true);
      expect(state.projectId).toBe('abc123456789');

      // Verify registerProject was called with realpath
      expect(mockInstance.registerProject).toHaveBeenCalledWith({
        path: realProjectPath,
        project_id: 'abc123456789',
        name: 'test-project',
      });

      await server.stop();

      // Verify deprioritizeProject was called on stop
      expect(mockInstance.deprioritizeProject).toHaveBeenCalledWith({
        project_id: 'abc123456789',
      });
    } finally {
      process.chdir(originalCwd);
    }
  });
});
