/**
 * Integration tests for WorkspaceQdrantMcpServer
 *
 * Tests all 4 MCP tools and session lifecycle through the server's
 * request handlers, verifying end-to-end behavior.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../../src/server.js';
import type { ServerConfig } from '../../src/types/index.js';
import { ServiceStatus } from '../../src/clients/grpc-types.js';

// Mock the DaemonClient
const mockDaemonClient = {
  connect: vi.fn().mockResolvedValue(undefined),
  close: vi.fn(),
  isConnected: vi.fn().mockReturnValue(true),
  registerProject: vi.fn().mockResolvedValue({
    created: true,
    project_id: 'test-project-id',
    priority: 'high',
    is_active: true,
  }),
  deprioritizeProject: vi.fn().mockResolvedValue({
    success: true,
    is_active: false,
    new_priority: 'normal',
  }),
  heartbeat: vi.fn().mockResolvedValue({ acknowledged: true }),
  healthCheck: vi.fn().mockResolvedValue({
    status: ServiceStatus.SERVICE_STATUS_HEALTHY,
    components: [],
  }),
  ingestText: vi.fn().mockResolvedValue({
    documentIds: ['doc-123'],
  }),
  embedText: vi.fn().mockResolvedValue({
    denseEmbedding: new Array(384).fill(0.1),
    sparseIndices: [1, 2, 3],
    sparseValues: [0.5, 0.3, 0.2],
  }),
};

vi.mock('../../src/clients/daemon-client.js', () => ({
  DaemonClient: vi.fn().mockImplementation(() => mockDaemonClient),
}));

// Mock the MCP SDK Server
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

// Mock Qdrant client for tools
const mockQdrantClient = {
  search: vi.fn().mockResolvedValue([
    {
      id: 'result-1',
      score: 0.95,
      payload: { content: 'Test content', title: 'Test Doc' },
    },
  ]),
  retrieve: vi.fn().mockResolvedValue([
    {
      id: 'doc-1',
      payload: { content: 'Retrieved content', title: 'Doc Title' },
    },
  ]),
  scroll: vi.fn().mockResolvedValue({
    points: [
      {
        id: 'scrolled-1',
        payload: { content: 'Scrolled content' },
      },
    ],
    next_page_offset: null,
  }),
  upsert: vi.fn().mockResolvedValue({ status: 'completed' }),
  delete: vi.fn().mockResolvedValue({ status: 'completed' }),
  getCollections: vi.fn().mockResolvedValue({ collections: [{ name: 'projects' }] }),
};

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => mockQdrantClient),
}));

// Test database schema (minimal version matching daemon)
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

CREATE TABLE IF NOT EXISTS unified_queue (
    queue_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    item_type TEXT NOT NULL,
    op TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    branch TEXT,
    payload_json TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_error TEXT,
    leased_by TEXT,
    lease_expires_at TEXT
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

describe('Server Integration Tests', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-integration-test-'));
    const dbPath = join(tempDir, 'state.db');

    // Create database with test schema
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

  describe('Tool Registration', () => {
    it('should register all 4 tools with MCP server', async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();

      const mcpServer = server.getMcpServer();
      expect(mcpServer.setRequestHandler).toHaveBeenCalledTimes(2);

      // First call should be ListToolsRequestSchema
      const firstCall = vi.mocked(mcpServer.setRequestHandler).mock.calls[0];
      expect(firstCall).toBeDefined();

      // Second call should be CallToolRequestSchema
      const secondCall = vi.mocked(mcpServer.setRequestHandler).mock.calls[1];
      expect(secondCall).toBeDefined();
    });
  });

  describe('Search Tool Integration', () => {
    beforeEach(async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();
    });

    it('should handle search with default parameters', async () => {
      // Get the tool call handler from setRequestHandler mock
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'search',
          arguments: { query: 'test query' },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.content[0].type).toBe('text');
      expect(result.isError).toBeUndefined();
    });

    it('should handle search with all parameters', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'search',
          arguments: {
            query: 'test query',
            collection: 'projects',
            mode: 'hybrid',
            scope: 'global',
            limit: 5,
            projectId: 'proj-123',
            branch: 'main',
            fileType: 'ts',
            includeLibraries: true,
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should include health status in search results when system healthy', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'search',
          arguments: { query: 'test' },
        },
      });

      const resultData = JSON.parse(result.content[0].text);
      // When healthy, health field should not be present
      expect(resultData.health).toBeUndefined();
    });
  });

  describe('Retrieve Tool Integration', () => {
    beforeEach(async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();
    });

    it('should retrieve document by ID', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'retrieve',
          arguments: {
            documentId: 'doc-123',
            collection: 'projects',
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should retrieve documents with filter', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'retrieve',
          arguments: {
            collection: 'projects',
            filter: { fileType: 'ts', branch: 'main' },
            limit: 5,
            offset: 0,
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should handle retrieve from libraries collection', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'retrieve',
          arguments: {
            collection: 'libraries',
            libraryName: 'lodash',
            limit: 10,
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should handle retrieve from memory collection', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'retrieve',
          arguments: {
            collection: 'memory',
            projectId: 'proj-123',
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });
  });

  describe('Memory Tool Integration', () => {
    beforeEach(async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();
    });

    it('should add global memory rule', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'memory',
          arguments: {
            action: 'add',
            content: 'Always use TypeScript strict mode',
            scope: 'global',
            title: 'Strict Mode Rule',
            tags: ['typescript', 'best-practices'],
            priority: 10,
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should add project-scoped memory rule', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'memory',
          arguments: {
            action: 'add',
            content: 'Use Vitest for testing',
            scope: 'project',
            projectId: 'proj-123',
            title: 'Testing Framework',
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should list memory rules', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'memory',
          arguments: {
            action: 'list',
            limit: 20,
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should update memory rule', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'memory',
          arguments: {
            action: 'update',
            ruleId: 'rule-123',
            content: 'Updated rule content',
          },
        },
      });

      expect(result.content).toBeDefined();
    });

    it('should remove memory rule', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'memory',
          arguments: {
            action: 'remove',
            ruleId: 'rule-123',
          },
        },
      });

      expect(result.content).toBeDefined();
    });

    it('should reject invalid memory action', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'memory',
          arguments: {
            action: 'invalid',
          },
        },
      });

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain('Invalid memory action');
    });
  });

  describe('Store Tool Integration', () => {
    beforeEach(async () => {
      server = new WorkspaceQdrantMcpServer({ config, stdio: false });
      await server.start();
    });

    it('should reject store to projects collection (not supported per spec)', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      // Per spec: store tool only supports libraries collection
      // Projects collection content is handled by daemon file watching
      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'store',
          arguments: {
            content: 'Test content to store',
            title: 'Test Document',
            sourceType: 'user_input',
            // Missing libraryName - required parameter
          },
        },
      });

      // Should error because libraryName is required
      expect(result.content).toBeDefined();
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain('libraryName is required');
    });

    it('should store content to libraries collection', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'store',
          arguments: {
            content: 'Library documentation content',
            collection: 'libraries',
            libraryName: 'lodash',
            title: 'Lodash API Reference',
            sourceType: 'web',
            url: 'https://lodash.com/docs',
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should store content with all metadata fields', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      // Per spec: store tool only supports libraries collection
      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'store',
          arguments: {
            content: 'Full metadata test content',
            libraryName: 'test-library',
            title: 'Test Document',
            filePath: '/docs/test.md',
            sourceType: 'file',
            url: 'https://example.com/docs',
            metadata: { author: 'test', version: '1.0' },
          },
        },
      });

      expect(result.content).toBeDefined();
      expect(result.isError).toBeUndefined();
    });

    it('should reject store without content', async () => {
      const mcpServer = server.getMcpServer();
      const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

      // Per spec: libraryName is required but missing content should error
      const result = await callHandler({
        method: 'tools/call',
        params: {
          name: 'store',
          arguments: {
            libraryName: 'test-library',
          },
        },
      });

      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain('Content is required');
    });
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
      // Create a project directory with .git and database entry
      const projectPath = join(tempDir, 'test-project');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));
      const realProjectPath = realpathSync(projectPath);

      // Add project to database
      const db = new Database(join(tempDir, 'state.db'));
      db.prepare(`
        INSERT INTO registered_projects
        (project_id, project_path, container_folder, is_active, created_at)
        VALUES ('test-proj-id', ?, 'test-project', 1, datetime('now'))
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
      // Create project to trigger heartbeat
      const projectPath = join(tempDir, 'test-project-hb');
      mkdirSync(projectPath);
      mkdirSync(join(projectPath, '.git'));
      const realProjectPath = realpathSync(projectPath);

      const db = new Database(join(tempDir, 'state.db'));
      db.prepare(`
        INSERT INTO registered_projects
        (project_id, project_path, container_folder, is_active, created_at)
        VALUES ('hb-proj-id', ?, 'test-project-hb', 1, datetime('now'))
      `).run(realProjectPath);
      db.close();

      const originalCwd = process.cwd();
      process.chdir(projectPath);

      try {
        server = new WorkspaceQdrantMcpServer({ config, stdio: false });
        await server.start();

        // Heartbeat is sent immediately on start
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
        INSERT INTO registered_projects
        (project_id, project_path, container_folder, is_active, created_at)
        VALUES ('stop-proj-id', ?, 'test-project-stop', 1, datetime('now'))
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
        INSERT INTO registered_projects
        (project_id, project_path, container_folder, is_active, created_at)
        VALUES ('clear-proj-id', ?, 'test-project-clear', 1, datetime('now'))
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
      // Start in temp dir without project markers
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

describe('Queue Fallback Integration', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-queue-test-'));
    const dbPath = join(tempDir, 'state.db');

    // Create database with test schema
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();

    config = createTestConfig(tempDir);

    // Reset mock to daemon unavailable state
    mockDaemonClient.connect.mockRejectedValue(new Error('Daemon unavailable'));
    mockDaemonClient.isConnected.mockReturnValue(false);
    mockDaemonClient.ingestText.mockRejectedValue(new Error('Daemon unavailable'));
  });

  afterEach(async () => {
    if (server) {
      await server.stop();
    }
    rmSync(tempDir, { recursive: true, force: true });

    // Reset mocks
    mockDaemonClient.connect.mockResolvedValue(undefined);
    mockDaemonClient.isConnected.mockReturnValue(true);
    mockDaemonClient.ingestText.mockResolvedValue({ documentIds: ['doc-123'] });
  });

  it('should fall back to queue when daemon unavailable for store', async () => {
    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    // Per spec: store tool only supports libraries collection
    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'store',
        arguments: {
          content: 'Content to queue',
          libraryName: 'test-library',
        },
      },
    });

    expect(result.content).toBeDefined();
    const resultData = JSON.parse(result.content[0].text);

    // Should have queue fallback indication
    expect(resultData.status === 'queued' || resultData.fallback_mode === 'unified_queue').toBe(true);
  });

  it('should include queue_id in fallback response', async () => {
    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    // Per spec: store tool only supports libraries collection
    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'store',
        arguments: {
          content: 'Another queued content',
          libraryName: 'test-library',
        },
      },
    });

    const resultData = JSON.parse(result.content[0].text);

    // Should have documentId and queue_id in fallback mode
    expect(resultData.documentId).toBeDefined();
    expect(resultData.queue_id).toBeDefined();
  });
});

describe('Uncertain Health Status Integration', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-health-test-'));
    const dbPath = join(tempDir, 'state.db');

    // Create database with test schema
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();

    config = createTestConfig(tempDir);
  });

  afterEach(async () => {
    if (server) {
      await server.stop();
    }
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('should include uncertain status when daemon unavailable', async () => {
    // Make daemon appear unhealthy
    mockDaemonClient.isConnected.mockReturnValue(false);
    mockDaemonClient.healthCheck.mockResolvedValue({
      status: ServiceStatus.SERVICE_STATUS_UNHEALTHY,
      components: [],
    });

    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();

    // Force health check to update state
    const healthMonitor = server.getHealthMonitor();
    await healthMonitor.forceCheck();

    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'search',
        arguments: { query: 'test' },
      },
    });

    const resultData = JSON.parse(result.content[0].text);

    // Should have health status indicating uncertainty
    expect(resultData.health).toBeDefined();
    expect(resultData.health.status).toBe('uncertain');
    expect(resultData.health.reason).toBe('daemon_unavailable');

    // Reset mock
    mockDaemonClient.isConnected.mockReturnValue(true);
    mockDaemonClient.healthCheck.mockResolvedValue({
      status: ServiceStatus.SERVICE_STATUS_HEALTHY,
      components: [],
    });
  });
});
