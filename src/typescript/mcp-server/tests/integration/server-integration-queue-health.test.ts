/**
 * Integration tests for queue fallback and uncertain health status.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../../src/server.js';
import type { ServerConfig } from '../../src/types/index.js';
import { ServiceStatus } from '../../src/clients/grpc-types.js';
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

describe('Queue Fallback Integration', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-queue-test-'));
    const dbPath = join(tempDir, 'state.db');
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
