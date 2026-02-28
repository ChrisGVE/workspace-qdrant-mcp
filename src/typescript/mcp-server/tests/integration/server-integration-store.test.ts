/**
 * Integration tests for the store tool.
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

describe('Store Tool Integration', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;

  beforeEach(async () => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-integration-test-'));
    const dbPath = join(tempDir, 'state.db');
    const db = new Database(dbPath);
    db.exec(TEST_SCHEMA);
    db.close();
    config = createTestConfig(tempDir);
    vi.clearAllMocks();
    server = new WorkspaceQdrantMcpServer({ config, stdio: false });
    await server.start();
  });

  afterEach(async () => {
    if (server) {
      await server.stop();
    }
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('should reject library store without libraryName', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

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

  it('should reject store project without path', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'store',
        arguments: {
          type: 'project',
        },
      },
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('path is required');
  });

  it('should handle store project registration via daemon', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'store',
        arguments: {
          type: 'project',
          path: '/tmp/test-project',
        },
      },
    });

    expect(result.content).toBeDefined();
    expect(result.content[0].text).toBeDefined();
  });
});
