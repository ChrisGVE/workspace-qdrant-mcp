/**
 * HTTP server for exposing Prometheus metrics in HTTP mode.
 *
 * Activated when MCP_SERVER_MODE=http (or any truthy value).
 * Listens on port MCP_METRICS_PORT (default 9092).
 * Serves GET /metrics with Prometheus text format.
 *
 * Uses Node built-in `http` module — no Express dependency.
 *
 * In stdio mode this module is not used; metrics are flushed via
 * pushMetricsOnExit() in metrics.ts (placeholder for task-6 OTLP).
 */

import {
  createServer as createHttpServer,
  type IncomingMessage,
  type ServerResponse,
} from 'node:http';
import type { Server } from 'node:http';
import { register } from './metrics.js';

const DEFAULT_METRICS_PORT = 9092;

/**
 * Start the /metrics HTTP server.
 *
 * @param port  Port to listen on (defaults to MCP_METRICS_PORT env var or 9092).
 * @returns     The Node http.Server instance (already listening).
 */
export function startMetricsServer(port?: number): Server {
  const listenPort =
    port ??
    (process.env['MCP_METRICS_PORT'] !== undefined
      ? parseInt(process.env['MCP_METRICS_PORT'], 10)
      : DEFAULT_METRICS_PORT);

  const server = createHttpServer((req: IncomingMessage, res: ServerResponse): void => {
    void handleRequest(req, res);
  });

  server.listen(listenPort, '0.0.0.0', () => {
    process.stderr.write(`[wqm-metrics] HTTP metrics server listening on :${listenPort}/metrics\n`);
  });

  return server;
}

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method !== 'GET' || req.url !== '/metrics') {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not Found');
    return;
  }

  try {
    const metrics = await register.metrics();
    res.writeHead(200, { 'Content-Type': register.contentType });
    res.end(metrics);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end(`Internal Server Error: ${msg}`);
  }
}
