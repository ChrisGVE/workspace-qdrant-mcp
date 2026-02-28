/**
 * Tests for WorkspaceQdrantMcpServer core functionality and session state
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../src/server.js';
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
      is_active: true,
      newly_registered: true,
    }),
    deprioritizeProject: vi.fn().mockResolvedValue({
      success: true,
      is_active: false,
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
    setRequestHandler: vi.fn(),
    onerror: null,
    onclose: null,
  })),
}));

vi.mock('@modelcontextprotocol/sdk/server/stdio.js', () => ({
  StdioServerTransport: vi.fn().mockImplementation(() => ({})),
}));

// Create test schema (minimal version matching daemon's watch_folders table)
const TEST_SCHEMA = `
CREATE TABLE IF NOT EXISTS watch_folders (
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    parent_watch_id TEXT,
    submodule_path TEXT,
    git_remote_url TEXT,
    remote_hash TEXT,
    disambiguation_path TEXT,
    is_active INTEGER DEFAULT 0,
    last_activity_at TEXT,
    library_mode TEXT,
    follow_symlinks INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    cleanup_on_disable INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_scan TEXT
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
      rulesCollectionName: 'rules',
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
      // Create a project directory
      const projectPath = join(tempDir, 'my-project');
      mkdirSync(projectPath);

      // Resolve real path (handles macOS /private symlink)
      const realProjectPath = realpathSync(projectPath);

      // Register project in database (initializeSession now queries DB, not filesystem)
      const db = new Database(dbPath);
      db.prepare(
        `INSERT INTO watch_folders
         (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
         VALUES ('watch-test', ?, 'projects', 'test_tenant_id', 1, datetime('now'), datetime('now'))`
      ).run(realProjectPath);
      db.close();

      // Change cwd to project
      const originalCwd = process.cwd();
      process.chdir(projectPath);

      try {
        const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        const state = server.getSessionState();
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
