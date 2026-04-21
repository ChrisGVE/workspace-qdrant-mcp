/**
 * Smoke tests for the HTTP MCP transport.
 *
 * Spins up a real WorkspaceQdrantMcpServer in `http` mode on an ephemeral
 * loopback port, issues a JSON-RPC `initialize` request, and verifies the
 * server returns a valid response with an `Mcp-Session-Id` header. These
 * tests exercise the wiring in `mcp-http-server.ts` end-to-end; they do not
 * mock the SDK transport. A separate mock-based test covers mode resolution.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AddressInfo } from 'node:net';
import Database from 'better-sqlite3';

import { WorkspaceQdrantMcpServer } from '../src/server.js';
import type { ServerConfig } from '../src/types/index.js';

vi.mock('../src/clients/daemon-client.js', () => ({
  DaemonClient: vi.fn().mockImplementation(() => ({
    connect: vi.fn().mockRejectedValue(new Error('Mock: daemon not available')),
    close: vi.fn(),
    isConnected: vi.fn().mockReturnValue(false),
    registerProject: vi.fn().mockResolvedValue({
      created: true,
      project_id: 'mock',
      priority: 'high',
      is_active: true,
      newly_registered: true,
    }),
    deprioritizeProject: vi.fn().mockResolvedValue({
      success: true,
      is_active: false,
      new_priority: 'normal',
    }),
    heartbeat: vi.fn().mockResolvedValue({ acknowledged: true }),
  })),
}));

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
    database: { path: join(tempDir, 'state.db') },
    qdrant: { url: 'http://localhost:6333', timeout: 5000 },
    daemon: { grpcPort: 50051, queuePollIntervalMs: 1000, queueBatchSize: 10 },
    watching: { patterns: ['*.ts'], ignorePatterns: ['node_modules/*'] },
    collections: { rulesCollectionName: 'rules' },
    environment: {},
  };
}

describe('MCP HTTP transport', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer | null = null;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-http-'));
    const db = new Database(join(tempDir, 'state.db'));
    db.exec(TEST_SCHEMA);
    db.close();
    config = createTestConfig(tempDir);
  });

  afterEach(async () => {
    if (server) {
      await server.stop();
      server = null;
    }
    rmSync(tempDir, { recursive: true, force: true });
    vi.clearAllMocks();
  });

  it('should start, respond to initialize, and stop cleanly', async () => {
    server = new WorkspaceQdrantMcpServer({
      config,
      mode: 'http',
      http: { host: '127.0.0.1', port: 0, path: '/mcp' },
    });
    await server.start();
    expect(server.getMode()).toBe('http');

    // Grab the ephemeral port assigned by the kernel.
    const httpServer = (
      server as unknown as {
        httpHandle: { httpServer: { address(): AddressInfo | string | null } };
      }
    ).httpHandle.httpServer;
    const addr = httpServer.address() as AddressInfo;
    expect(addr).toBeTruthy();
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    // healthz reachable
    const healthRes = await fetch(`${baseUrl}/healthz`);
    expect(healthRes.status).toBe(200);
    expect(await healthRes.text()).toBe('ok');

    // JSON-RPC initialize on /mcp
    const initRes = await fetch(`${baseUrl}/mcp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json, text/event-stream',
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {
          protocolVersion: '2025-06-18',
          capabilities: {},
          clientInfo: { name: 'test-client', version: '0.0.0' },
        },
      }),
    });

    expect(initRes.status).toBe(200);
    expect(initRes.headers.get('mcp-session-id')).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
    );
  });

  it('should return 404 for unknown paths', async () => {
    server = new WorkspaceQdrantMcpServer({
      config,
      mode: 'http',
      http: { host: '127.0.0.1', port: 0, path: '/mcp' },
    });
    await server.start();

    const httpServer = (
      server as unknown as {
        httpHandle: { httpServer: { address(): AddressInfo | string | null } };
      }
    ).httpHandle.httpServer;
    const addr = httpServer.address() as AddressInfo;

    const res = await fetch(`http://127.0.0.1:${addr.port}/nonexistent`);
    expect(res.status).toBe(404);
  });

  it('should honour a custom path', async () => {
    server = new WorkspaceQdrantMcpServer({
      config,
      mode: 'http',
      http: { host: '127.0.0.1', port: 0, path: '/rpc/mcp' },
    });
    await server.start();

    const httpServer = (
      server as unknown as {
        httpHandle: { httpServer: { address(): AddressInfo | string | null } };
      }
    ).httpHandle.httpServer;
    const addr = httpServer.address() as AddressInfo;

    // Default /mcp path must 404 when custom path is configured.
    const wrongRes = await fetch(`http://127.0.0.1:${addr.port}/mcp`);
    expect(wrongRes.status).toBe(404);
  });
});
