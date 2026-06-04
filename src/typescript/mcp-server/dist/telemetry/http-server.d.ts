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
import { type IncomingMessage } from 'node:http';
import type { Server } from 'node:http';
/**
 * Returns true when `host` resolves to the local loopback interface.
 * Covers IPv4 `127.0.0.1`, IPv6 `::1`, and the hostname `localhost`
 * (case-insensitive).
 */
export declare function isLoopbackAddress(host: string): boolean;
/**
 * Extract the bearer token from an HTTP `Authorization` header.
 *
 * Parses `Authorization: Bearer <token>` case-insensitively.
 * Returns the token string, or `null` if the header is absent or
 * does not follow the `Bearer <token>` format.
 */
export declare function extractBearerToken(req: IncomingMessage): string | null;
/**
 * Produce a human-readable error message for common `server.listen()` failures.
 *
 * @param err   The error emitted by the `'error'` event on the http.Server.
 * @param port  The port that was attempted.
 */
export declare function formatListenError(err: Error & {
    code?: string;
}, port: number): string;
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
export declare function startMetricsServer(port?: number): Server;
//# sourceMappingURL=http-server.d.ts.map