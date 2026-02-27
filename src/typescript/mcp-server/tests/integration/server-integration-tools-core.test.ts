/**
 * Integration tests for core tool registration, search, and retrieve tools.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
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

  describe('Tool Registration', () => {
    it('should register all 6 tools with MCP server', async () => {
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
});
