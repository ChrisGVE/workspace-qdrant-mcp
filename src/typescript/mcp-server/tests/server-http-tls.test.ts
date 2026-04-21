/**
 * Native TLS tests for the MCP HTTP transport.
 *
 * Generates a self-signed cert via `openssl` into a per-suite temp dir,
 * boots the server with TLS enabled, and verifies an HTTPS client can hit
 * `/healthz`. A separate case verifies startup fails fast when cert/key
 * paths are unreadable.
 */

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import type { AddressInfo } from 'node:net';
import { request } from 'node:https';
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

const TEST_TOKEN = 'a'.repeat(32);

let certDir: string;
let certPath: string;
let keyPath: string;

beforeAll(() => {
  certDir = mkdtempSync(join(tmpdir(), 'mcp-tls-cert-'));
  keyPath = join(certDir, 'key.pem');
  certPath = join(certDir, 'cert.pem');
  execFileSync(
    'openssl',
    [
      'req',
      '-x509',
      '-newkey',
      'rsa:2048',
      '-nodes',
      '-days',
      '1',
      '-keyout',
      keyPath,
      '-out',
      certPath,
      '-subj',
      '/CN=localhost',
      '-addext',
      'subjectAltName=DNS:localhost,IP:127.0.0.1',
    ],
    { stdio: 'ignore' }
  );
});

afterAll(() => {
  rmSync(certDir, { recursive: true, force: true });
});

function httpsGet(port: number, path: string): Promise<{ statusCode: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = request(
      {
        host: '127.0.0.1',
        port,
        method: 'GET',
        path,
        rejectUnauthorized: false,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer) => chunks.push(chunk));
        res.on('end', () =>
          resolve({
            statusCode: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString('utf8'),
          })
        );
      }
    );
    req.on('error', reject);
    req.end();
  });
}

describe('MCP HTTP transport — native TLS', () => {
  let tempDir: string;
  let config: ServerConfig;
  let server: WorkspaceQdrantMcpServer | null = null;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'mcp-tls-'));
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

  it('should terminate TLS when cert+key are provided', async () => {
    server = new WorkspaceQdrantMcpServer({
      config,
      mode: 'http',
      http: {
        host: '127.0.0.1',
        port: 0,
        path: '/mcp',
        tls: { certPath, keyPath },
      },
      auth: { token: TEST_TOKEN, rateLimitPerMin: 1000, corsOrigins: [] },
    });
    await server.start();

    const handle = (
      server as unknown as {
        httpHandle: {
          tlsEnabled: boolean;
          httpServer: { address(): AddressInfo | string | null };
        };
      }
    ).httpHandle;
    expect(handle.tlsEnabled).toBe(true);

    const addr = handle.httpServer.address() as AddressInfo;

    // Self-signed cert → skip CA verification on the client side only.
    const { statusCode, body } = await httpsGet(addr.port, '/healthz');
    expect(statusCode).toBe(200);
    expect(body).toBe('ok');
  });

  it('should run plain HTTP when TLS options are absent', async () => {
    server = new WorkspaceQdrantMcpServer({
      config,
      mode: 'http',
      http: { host: '127.0.0.1', port: 0, path: '/mcp' },
      auth: { token: TEST_TOKEN, rateLimitPerMin: 1000, corsOrigins: [] },
    });
    await server.start();

    const handle = (
      server as unknown as {
        httpHandle: {
          tlsEnabled: boolean;
          httpServer: { address(): AddressInfo | string | null };
        };
      }
    ).httpHandle;
    expect(handle.tlsEnabled).toBe(false);
  });

  it('should fail fast when the cert path is unreadable', async () => {
    server = new WorkspaceQdrantMcpServer({
      config,
      mode: 'http',
      http: {
        host: '127.0.0.1',
        port: 0,
        path: '/mcp',
        tls: { certPath: '/nonexistent/cert.pem', keyPath },
      },
      auth: { token: TEST_TOKEN, rateLimitPerMin: 1000, corsOrigins: [] },
    });
    await expect(server.start()).rejects.toThrow(/MCP_HTTP_TLS_CERT/);
    server = null;
  });

  it('should fail fast when the key path is unreadable', async () => {
    server = new WorkspaceQdrantMcpServer({
      config,
      mode: 'http',
      http: {
        host: '127.0.0.1',
        port: 0,
        path: '/mcp',
        tls: { certPath, keyPath: '/nonexistent/key.pem' },
      },
      auth: { token: TEST_TOKEN, rateLimitPerMin: 1000, corsOrigins: [] },
    });
    await expect(server.start()).rejects.toThrow(/MCP_HTTP_TLS_KEY/);
    server = null;
  });
});
