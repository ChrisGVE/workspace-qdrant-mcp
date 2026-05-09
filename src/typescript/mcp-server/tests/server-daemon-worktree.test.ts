/**
 * Tests for WorkspaceQdrantMcpServer worktree and watch_path lifecycle scenarios
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../src/server.js';
import type { ServerConfig } from '../src/types/index.js';

// Mock the DaemonClient — default to unavailable; individual tests override as needed
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

describe('Session lifecycle — worktree and watch_path scenarios', () => {
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

  // T-14: cleanup sends watch_path in DeprioritizeProject call
  it('should send watch_path in deprioritizeProject call on cleanup', async () => {
    const { DaemonClient } = await import('../src/clients/daemon-client.js');
    const mockInstance = {
      connect: vi.fn().mockResolvedValue(undefined),
      close: vi.fn(),
      isConnected: vi.fn().mockReturnValue(true),
      registerProject: vi.fn(), // configured below after realProjectPath is known
      deprioritizeProject: vi.fn().mockResolvedValue({
        success: true,
        is_active: false,
        new_priority: 'normal',
      }),
      heartbeat: vi.fn().mockResolvedValue({ acknowledged: true }),
    };
    vi.mocked(DaemonClient).mockImplementation(
      () => mockInstance as unknown as InstanceType<typeof DaemonClient>
    );

    const projectPath = join(tempDir, 'path-project');
    mkdirSync(projectPath);
    mkdirSync(join(projectPath, '.git'));
    const realProjectPath = realpathSync(projectPath);

    // Configure registerProject mock now that realProjectPath is known
    mockInstance.registerProject.mockResolvedValue({
      created: false,
      project_id: 'proj-with-path',
      priority: 'high',
      is_active: true,
      newly_registered: false,
      is_worktree: true,
      watch_path: realProjectPath,
    });

    const db = new Database(join(tempDir, 'state.db'));
    db.prepare(
      `
      INSERT INTO watch_folders
      (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
      VALUES ('watch-path', ?, 'projects', 'proj-with-path', 1, datetime('now'), datetime('now'))
    `
    ).run(realProjectPath);
    db.close();

    const originalCwd = process.cwd();
    process.chdir(projectPath);

    try {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const state = server.getSessionState();
      expect(state.projectPath).toBe(realProjectPath);

      await server.stop();

      // watch_path must be included in the deprioritizeProject call
      expect(mockInstance.deprioritizeProject).toHaveBeenCalledWith({
        project_id: 'proj-with-path',
        watch_path: realProjectPath,
      });
    } finally {
      process.chdir(originalCwd);
    }
  });

  // T-15: session initialization in a worktree correctly sets projectPath
  it('should set projectPath to the worktree directory on initialization', async () => {
    const { DaemonClient } = await import('../src/clients/daemon-client.js');
    const mockInstance = {
      connect: vi.fn().mockResolvedValue(undefined),
      close: vi.fn(),
      isConnected: vi.fn().mockReturnValue(true),
      registerProject: vi.fn().mockResolvedValue({
        created: false,
        project_id: 'worktree-proj-id',
        priority: 'high',
        is_active: true,
        newly_registered: false,
        is_worktree: true,
        watch_path: realTempDir,
      }),
      deprioritizeProject: vi.fn().mockResolvedValue({
        success: true,
        is_active: false,
        new_priority: 'normal',
      }),
      heartbeat: vi.fn().mockResolvedValue({ acknowledged: true }),
    };
    vi.mocked(DaemonClient).mockImplementation(
      () => mockInstance as unknown as InstanceType<typeof DaemonClient>
    );

    // Set up a worktree: .git is a FILE (pointer), not a directory
    const worktreePath = join(tempDir, 'feature-worktree');
    mkdirSync(worktreePath);
    const { writeFileSync } = await import('node:fs');
    writeFileSync(
      join(worktreePath, '.git'),
      `gitdir: ${join(realTempDir, '.git', 'worktrees', 'feature-worktree')}\n`
    );
    const realWorktreePath = realpathSync(worktreePath);

    // Register the worktree path in the database as if the daemon did it
    const db = new Database(join(tempDir, 'state.db'));
    db.exec('ALTER TABLE watch_folders ADD COLUMN is_worktree INTEGER DEFAULT 0');
    db.prepare(
      `
      INSERT INTO watch_folders
      (watch_id, path, collection, tenant_id, is_active, is_worktree, created_at, updated_at)
      VALUES ('watch-wt', ?, 'projects', 'worktree-proj-id', 1, 1, datetime('now'), datetime('now'))
    `
    ).run(realWorktreePath);
    db.close();

    const originalCwd = process.cwd();
    process.chdir(worktreePath);

    try {
      const server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const state = server.getSessionState();
      // The session must record the worktree directory, not the main repo
      expect(state.projectPath).toBe(realWorktreePath);
      expect(state.projectId).toBe('worktree-proj-id');

      await server.stop();
    } finally {
      process.chdir(originalCwd);
    }
  });
});
