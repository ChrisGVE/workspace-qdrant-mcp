/**
 * Integration tests for the rules tool.
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

describe('Rules Tool Integration', () => {
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

  it('should add global rule', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'rules',
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

  it('should add project-scoped rule', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'rules',
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

  it('should list rules', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'rules',
        arguments: {
          action: 'list',
          limit: 20,
        },
      },
    });

    expect(result.content).toBeDefined();
    expect(result.isError).toBeUndefined();
  });

  it('should update rule', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'rules',
        arguments: {
          action: 'update',
          ruleId: 'rule-123',
          content: 'Updated rule content',
        },
      },
    });

    expect(result.content).toBeDefined();
  });

  it('should remove rule', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'rules',
        arguments: {
          action: 'remove',
          ruleId: 'rule-123',
        },
      },
    });

    expect(result.content).toBeDefined();
  });

  it('should reject invalid rules action', async () => {
    const mcpServer = server.getMcpServer();
    const callHandler = vi.mocked(mcpServer.setRequestHandler).mock.calls[1][1];

    const result = await callHandler({
      method: 'tools/call',
      params: {
        name: 'rules',
        arguments: {
          action: 'invalid',
        },
      },
    });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('Invalid rules action');
  });
});
