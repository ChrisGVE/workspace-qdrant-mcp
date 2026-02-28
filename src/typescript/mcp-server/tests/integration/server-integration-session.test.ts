/**
 * Integration tests for session lifecycle, error handling, graceful degradation,
 * and health monitor.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../../src/server.js';
import type { ServerConfig } from '../../src/types/index.js';
import { mockDaemonClient, mockQdrantClient, TEST_SCHEMA, createTestConfig } from './shared-setup.js';

vi.mock('../../src/clients/daemon-client.js', () => ({
  DaemonClient: vi.fn().mockImplementation(() => mockDaemonClient),
}));

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

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => mockQdrantClient),
}));

describe('Server Integration Tests', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-integration-test-'));
    const dbPath = join(tempDir, 'state.db');
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();
    config = createTestConfig(tempDir);
    vi.clearAllMocks();
  });

  afterEach(async () => {
    if (server) {
      await server.stop();
    }
    rmSync(tempDir, { recursive: true, force: true });
  });

  describe('Tool Error Handling', () => {
    beforeEach(async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();
    });

    it('should handle unknown tool name', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'unknown_tool',
          arguments: {},
        },
      });

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain('Unknown tool');
    });

    it('should handle tool execution errors gracefully', async () => {
      // Make the daemon client throw an error
      mockDaemonClient.embedText.mockRejectedValueOnce(new Error('Daemon unavailable'));

      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'search',
          arguments: { query: 'test' },
        },
      });

      // Should return error response, not throw
      expect(result.content).toBeDefined();
    });
  });

  describe('Session Lifecycle', () => {
    it('should initialize session with unique ID', async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const state = server.getSessionState();
      expect(state.sessionId).toBeDefined();
      expect(state.sessionId.length).toBe(36); // UUID format
    });

    it('should connect to daemon on start', async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      expect(mockDaemonClient.connect).toHaveBeenCalled();
      expect(server.isDaemonConnected()).toBe(true);
    });

    it('should register project with daemon when detected', async () => {
      const projectPath = join(tempDir, 'test-project');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));
      const realProjectPath = realpathSync(projectPath);

      const db = new Database(join(tempDir, 'state.db'));
      db.prepare(`
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-test', ?, 'projects', 'test-proj-id', 1, datetime('now'), datetime('now'))
      `).run(realProjectPath);
      db.close();

      const originalCwd = process.cwd();
      process.chdir(projectPath);

      try {
        server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        const state = server.getSessionState();
        expect(state.projectPath).toBe(realProjectPath);
        expect(state.projectId).toBe('test-proj-id');
        expect(mockDaemonClient.registerProject).toHaveBeenCalled();
      } finally {
        process.chdir(originalCwd);
      }
    });

    it('should send heartbeat after connecting', async () => {
      const projectPath = join(tempDir, 'test-project-hb');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));
      const realProjectPath = realpathSync(projectPath);

      const db = new Database(join(tempDir, 'state.db'));
      db.prepare(`
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-hb', ?, 'projects', 'hb-proj-id', 1, datetime('now'), datetime('now'))
      `).run(realProjectPath);
      db.close();

      const originalCwd = process.cwd();
      process.chdir(projectPath);

      try {
        server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        expect(mockDaemonClient.heartbeat).toHaveBeenCalled();
      } finally {
        process.chdir(originalCwd);
      }
    });

    it('should deprioritize project on stop', async () => {
      const projectPath = join(tempDir, 'test-project-stop');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));
      const realProjectPath = realpathSync(projectPath);

      const db = new Database(join(tempDir, 'state.db'));
      db.prepare(`
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-stop', ?, 'projects', 'stop-proj-id', 1, datetime('now'), datetime('now'))
      `).run(realProjectPath);
      db.close();

      const originalCwd = process.cwd();
      process.chdir(projectPath);

      try {
        server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();
        await server.stop();

        expect(mockDaemonClient.deprioritizeProject).toHaveBeenCalledWith({
          project_id: 'stop-proj-id',
        });
      } finally {
        process.chdir(originalCwd);
      }
    });

    it('should clear heartbeat interval on stop', async () => {
      const projectPath = join(tempDir, 'test-project-clear');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));
      const realProjectPath = realpathSync(projectPath);

      const db = new Database(join(tempDir, 'state.db'));
      db.prepare(`
        INSERT INTO watch_folders
        (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES ('watch-clear', ?, 'projects', 'clear-proj-id', 1, datetime('now'), datetime('now'))
      `).run(realProjectPath);
      db.close();

      const originalCwd = process.cwd();
      process.chdir(projectPath);

      try {
        server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        const stateBefore = server.getSessionState();
        expect(stateBefore.heartbeatInterval).not.toBeNull();

        await server.stop();

        const stateAfter = server.getSessionState();
        expect(stateAfter.heartbeatInterval).toBeNull();
      } finally {
        process.chdir(originalCwd);
      }
    });
  });

  describe('Graceful Degradation', () => {
    it('should start when daemon unavailable', async () => {
      mockDaemonClient.connect.mockRejectedValueOnce(new Error('Connection refused'));

      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      expect(server.isReady()).toBe(true);
      expect(server.isDaemonConnected()).toBe(false);
    });

    it('should work without project detected', async () => {
      const originalCwd = process.cwd();
      process.chdir(tempDir);

      try {
        server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        const state = server.getSessionState();
        expect(state.projectPath).toBeNull();
        expect(state.projectId).toBeNull();
        expect(server.isReady()).toBe(true);
      } finally {
        process.chdir(originalCwd);
      }
    });
  });

  describe('Health Monitor Integration', () => {
    it('should start health monitor on server start', async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const healthMonitor = server.getHealthMonitor();
      expect(healthMonitor).toBeDefined();
      expect(healthMonitor.isHealthy()).toBe(true);
    });

    it('should stop health monitor on server stop', async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const healthMonitor = server.getHealthMonitor();
      const stopSpy = vi.spyOn(healthMonitor, 'stop');

      await server.stop();

      expect(stopSpy).toHaveBeenCalled();
    });
  });
});
