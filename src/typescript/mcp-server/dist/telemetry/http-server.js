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
 *
 * ## Bind address (MCP_METRICS_HOST)
 *
 * Defaults to `127.0.0.1` (loopback-only, safe for local scraping).
 * Set `MCP_METRICS_HOST=0.0.0.0` to bind on all interfaces (e.g. inside
 * a container). When host is non-loopback, bearer-token auth is enforced:
 * - If `MCP_METRICS_TOKEN` is set: every request must carry
 *   `Authorization: Bearer <token>`. Missing or wrong token → 401.
 * - If `MCP_METRICS_TOKEN` is NOT set but host is non-loopback: all
 *   requests are rejected with 401 (fail-closed). A startup warning is
 *   also emitted so operators know the misconfiguration immediately.
 *
 * ## Port validation (MCP_METRICS_PORT)
 *
 * Must be an integer in [1, 65535]. Invalid value → startup Error thrown
 * before `server.listen()` is called. EADDRINUSE / EACCES errors after
 * `listen()` are logged but do not crash the process.
 */
import { createServer as createHttpServer, } from 'node:http';
import { register } from './metrics.js';
const DEFAULT_METRICS_PORT = 9092;
const DEFAULT_METRICS_HOST = '127.0.0.1';
// ── Helpers ────────────────────────────────────────────────────────────────────
/**
 * Returns true when `host` resolves to the local loopback interface.
 * Covers IPv4 `127.0.0.1`, IPv6 `::1`, and the hostname `localhost`
 * (case-insensitive).
 */
export function isLoopbackAddress(host) {
    const lower = host.toLowerCase();
    return lower === '127.0.0.1' || lower === '::1' || lower === 'localhost';
}
/**
 * Extract the bearer token from an HTTP `Authorization` header.
 *
 * Parses `Authorization: Bearer <token>` case-insensitively.
 * Returns the token string, or `null` if the header is absent or
 * does not follow the `Bearer <token>` format.
 */
export function extractBearerToken(req) {
    const header = req.headers['authorization'];
    if (!header)
        return null;
    const match = /^bearer\s+(\S+)$/i.exec(header);
    return match?.[1] ?? null;
}
/**
 * Produce a human-readable error message for common `server.listen()` failures.
 *
 * @param err   The error emitted by the `'error'` event on the http.Server.
 * @param port  The port that was attempted.
 */
export function formatListenError(err, port) {
    if (err.code === 'EADDRINUSE') {
        return `[wqm-metrics] Port ${port} is already in use (EADDRINUSE). Choose a different MCP_METRICS_PORT.`;
    }
    if (err.code === 'EACCES') {
        return `[wqm-metrics] Permission denied binding to port ${port} (EACCES). Use a port > 1024 or run with appropriate privileges.`;
    }
    return `[wqm-metrics] Failed to start metrics server on port ${port}: ${err.message}`;
}
// ── Server ─────────────────────────────────────────────────────────────────────
/**
 * Start the /metrics HTTP server.
 *
 * Reads `MCP_METRICS_HOST` (default `127.0.0.1`) and `MCP_METRICS_PORT`
 * (default 9092). When host is non-loopback, bearer-token auth via
 * `MCP_METRICS_TOKEN` is enforced (see module-level doc for rules).
 *
 * @param port  Port override (for tests). When omitted, reads MCP_METRICS_PORT.
 * @returns     The Node http.Server instance (already listening).
 * @throws      Error if the resolved port is outside [1, 65535].
 */
export function startMetricsServer(port) {
    const rawPort = port ??
        (process.env['MCP_METRICS_PORT'] !== undefined
            ? parseInt(process.env['MCP_METRICS_PORT'], 10)
            : DEFAULT_METRICS_PORT);
    // Validate port before attempting to bind.
    if (!Number.isInteger(rawPort) || rawPort < 1 || rawPort > 65535) {
        throw new Error(`[wqm-metrics] Invalid MCP_METRICS_PORT value "${String(process.env['MCP_METRICS_PORT'] ?? rawPort)}". Must be an integer in [1, 65535].`);
    }
    const listenPort = rawPort;
    const host = process.env['MCP_METRICS_HOST'] ?? DEFAULT_METRICS_HOST;
    const token = process.env['MCP_METRICS_TOKEN'];
    const loopback = isLoopbackAddress(host);
    if (!loopback && !token) {
        process.stderr.write(`[wqm-metrics] WARNING: metrics server bound to non-loopback host "${host}" without MCP_METRICS_TOKEN set. All /metrics requests will be rejected (401). Set MCP_METRICS_TOKEN to allow scraping.\n`);
    }
    const server = createHttpServer((req, res) => {
        void handleRequest(req, res, loopback, token ?? null);
    });
    server.on('error', (err) => {
        process.stderr.write(formatListenError(err, listenPort) + '\n');
    });
    server.listen(listenPort, host, () => {
        process.stderr.write(`[wqm-metrics] HTTP metrics server listening on ${host}:${listenPort}/metrics\n`);
    });
    return server;
}
async function handleRequest(req, res, loopback, token) {
    if (req.method !== 'GET' || req.url !== '/metrics') {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
        return;
    }
    // Enforce bearer-token auth for non-loopback binds.
    if (!loopback) {
        const provided = extractBearerToken(req);
        if (!token) {
            // No token configured: fail-closed.
            res.writeHead(401, { 'Content-Type': 'text/plain', 'WWW-Authenticate': 'Bearer' });
            res.end('Unauthorized: MCP_METRICS_TOKEN not configured');
            return;
        }
        if (provided !== token) {
            res.writeHead(401, { 'Content-Type': 'text/plain', 'WWW-Authenticate': 'Bearer' });
            res.end('Unauthorized');
            return;
        }
    }
    try {
        const metrics = await register.metrics();
        res.writeHead(200, { 'Content-Type': register.contentType });
        res.end(metrics);
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end(`Internal Server Error: ${msg}`);
    }
}
//# sourceMappingURL=http-server.js.map