/**
 * Tests for telemetry/http-server.ts
 *
 * Verifies:
 *  - GET /metrics responds 200 with Prometheus text content-type
 *  - Response body contains Prometheus metric lines
 *  - Non-/metrics paths return 404
 *  - Non-GET methods return 404
 */

import { describe, it, expect } from 'vitest';
import { request as httpRequest, type IncomingMessage } from 'node:http';
import type { Server } from 'node:http';
import { startMetricsServer } from '../../src/telemetry/http-server.js';

// Each test gets its own port so there are no TIME_WAIT / ECONNRESET races
// when the OS releases the port between afterEach and the next test start.
// Ports start at 19100 and increment by 1 per test in this file.
let portCounter = 19100;

function nextPort(): number {
  return portCounter++;
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((err) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

function makeRequest(
  port: number,
  path: string,
  method = 'GET'
): Promise<{ statusCode: number; contentType: string; body: string }> {
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      { hostname: '127.0.0.1', port, path, method },
      (res: IncomingMessage) => {
        let body = '';
        res.on('data', (chunk: Buffer) => {
          body += chunk.toString();
        });
        res.on('end', () => {
          resolve({
            statusCode: res.statusCode ?? 0,
            contentType: (res.headers['content-type'] as string) ?? '',
            body,
          });
        });
      }
    );
    req.on('error', reject);
    req.end();
  });
}

function startAndWait(port: number): Promise<Server> {
  return new Promise((resolve, reject) => {
    const server = startMetricsServer(port);
    server.once('listening', () => resolve(server));
    server.once('error', reject);
  });
}

describe('startMetricsServer', () => {
  it('GET /metrics returns 200', async () => {
    const port = nextPort();
    const server = await startAndWait(port);
    try {
      const { statusCode } = await makeRequest(port, '/metrics');
      expect(statusCode).toBe(200);
    } finally {
      await closeServer(server);
    }
  });

  it('GET /metrics returns Prometheus text content-type', async () => {
    const port = nextPort();
    const server = await startAndWait(port);
    try {
      const { contentType } = await makeRequest(port, '/metrics');
      expect(contentType).toContain('text/plain');
    } finally {
      await closeServer(server);
    }
  });

  it('GET /metrics body contains Prometheus comment lines', async () => {
    const port = nextPort();
    const server = await startAndWait(port);
    try {
      const { body } = await makeRequest(port, '/metrics');
      // Prometheus text format always has # HELP or # TYPE lines
      expect(body).toMatch(/^#\s+(HELP|TYPE)\s+/m);
    } finally {
      await closeServer(server);
    }
  });

  it('GET /metrics body contains wqm_mcp metric names', async () => {
    const port = nextPort();
    const server = await startAndWait(port);
    try {
      const { body } = await makeRequest(port, '/metrics');
      expect(body).toContain('wqm_mcp_tool_invocations_total');
      expect(body).toContain('wqm_mcp_session_count');
    } finally {
      await closeServer(server);
    }
  });

  it('GET /unknown returns 404', async () => {
    const port = nextPort();
    const server = await startAndWait(port);
    try {
      const { statusCode } = await makeRequest(port, '/unknown');
      expect(statusCode).toBe(404);
    } finally {
      await closeServer(server);
    }
  });

  it('POST /metrics returns 404', async () => {
    const port = nextPort();
    const server = await startAndWait(port);
    try {
      const { statusCode } = await makeRequest(port, '/metrics', 'POST');
      expect(statusCode).toBe(404);
    } finally {
      await closeServer(server);
    }
  });
});
