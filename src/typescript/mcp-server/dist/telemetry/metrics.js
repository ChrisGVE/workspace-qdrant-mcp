/**
 * Prometheus metrics for the workspace-qdrant MCP server.
 *
 * Defines 9 metric families:
 *   wqm_mcp_tool_invocations_total     - Counter, labels [tool, status]
 *   wqm_mcp_tool_duration_seconds      - Histogram, label [tool], buckets [0.01…5]
 *   wqm_mcp_session_count              - Gauge
 *   wqm_mcp_daemon_fallback_total      - Counter, labels [tool, reason]
 *   wqm_mcp_cache_hits_total           - Counter, label [cache]
 *   wqm_mcp_cache_misses_total         - Counter, label [cache]
 *   wqm_mcp_http_requests_total        - Counter, labels [path, status_class]
 *   wqm_mcp_http_auth_failures_total   - Counter, label [reason]
 *   wqm_mcp_http_rate_limited_total    - Counter (no labels; single signal)
 *
 * Cache hit/miss counters are defined but unwired: no cache layer exists in
 * the MCP server at v0.1.3. They are ready for future use.
 *
 * Daemon-fallback counter is wired to gRPC error paths in DaemonClient callers.
 * For operations that have no current fallback path (e.g. store queue), the
 * counter is still defined so dashboards can reference it.
 */
import { Counter, Histogram, Gauge, Registry } from 'prom-client';
import { pushMetricsToOTLP } from './otlp.js';
/** Shared Prometheus registry for this MCP server process. */
export const register = new Registry();
// ── Metric definitions ──────────────────────────────────────────────────────
export const toolInvocations = new Counter({
    name: 'wqm_mcp_tool_invocations_total',
    help: 'Total MCP tool invocations by tool name and completion status',
    labelNames: ['tool', 'status'],
    registers: [register],
});
export const toolDuration = new Histogram({
    name: 'wqm_mcp_tool_duration_seconds',
    help: 'MCP tool execution duration in seconds',
    labelNames: ['tool'],
    buckets: [0.01, 0.05, 0.1, 0.5, 1, 5],
    registers: [register],
});
export const sessionCount = new Gauge({
    name: 'wqm_mcp_session_count',
    help: 'Number of active MCP sessions',
    registers: [register],
});
export const daemonFallback = new Counter({
    name: 'wqm_mcp_daemon_fallback_total',
    help: 'Number of times the daemon was unreachable and a fallback was triggered',
    labelNames: ['tool', 'reason'],
    registers: [register],
});
/**
 * Cache hits by cache type.
 * Defined for future use — no in-process cache exists at v0.1.3.
 */
export const cacheHits = new Counter({
    name: 'wqm_mcp_cache_hits_total',
    help: 'Cache hits by cache name (defined for future use; no cache layer at v0.1.3)',
    labelNames: ['cache'],
    registers: [register],
});
/**
 * Cache misses by cache type.
 * Defined for future use — no in-process cache exists at v0.1.3.
 */
export const cacheMisses = new Counter({
    name: 'wqm_mcp_cache_misses_total',
    help: 'Cache misses by cache name (defined for future use; no cache layer at v0.1.3)',
    labelNames: ['cache'],
    registers: [register],
});
// ── HTTP transport metrics (MCP_SERVER_MODE=http) ──────────────────────────
/**
 * Total HTTP requests handled by the MCP listener, bucketed by logical path
 * (`/mcp`, `/healthz`, `other`) and status class (`2xx`, `4xx`, `5xx`). These
 * are only incremented in HTTP mode; stdio deployments leave them at zero.
 */
export const httpRequests = new Counter({
    name: 'wqm_mcp_http_requests_total',
    help: 'MCP HTTP transport requests by path and status class',
    labelNames: ['path', 'status_class'],
    registers: [register],
});
/**
 * HTTP bearer-auth failures. Labelled by failure reason (`missing_header`,
 * `invalid_token`, `not_configured`) so dashboards can distinguish
 * mis-configured clients from probable attacks.
 */
export const httpAuthFailures = new Counter({
    name: 'wqm_mcp_http_auth_failures_total',
    help: 'MCP HTTP auth failures by failure reason',
    labelNames: ['reason'],
    registers: [register],
});
/**
 * HTTP rate-limit hits. Single counter — alerting on sustained non-zero rate
 * catches runaway clients or brute-force attempts.
 */
export const httpRateLimited = new Counter({
    name: 'wqm_mcp_http_rate_limited_total',
    help: 'MCP HTTP requests rejected by the per-IP sliding-window rate limiter',
    registers: [register],
});
// ── Tool wrapper ─────────────────────────────────────────────────────────────
/**
 * Wraps a tool handler function with Prometheus instrumentation.
 *
 * Records:
 *  - wqm_mcp_tool_invocations_total{tool, status="success"|"error"}
 *  - wqm_mcp_tool_duration_seconds{tool}
 *
 * The original return value is preserved on success; the original error is
 * re-thrown unchanged on failure so callers see unmodified error behaviour.
 */
export async function withToolMetrics(toolName, fn) {
    const end = toolDuration.startTimer({ tool: toolName });
    try {
        const result = await fn();
        end();
        toolInvocations.labels({ tool: toolName, status: 'success' }).inc();
        return result;
    }
    catch (err) {
        end();
        toolInvocations.labels({ tool: toolName, status: 'error' }).inc();
        throw err;
    }
}
// ── Session helpers ───────────────────────────────────────────────────────────
/** Call once when a new MCP session starts. */
export function recordSessionStart() {
    sessionCount.inc();
}
/** Call once when the current MCP session ends. */
export function recordSessionEnd() {
    sessionCount.dec();
}
// ── Daemon-fallback helper ────────────────────────────────────────────────────
/**
 * Increment the daemon-fallback counter.
 *
 * @param tool   Name of the tool that triggered the fallback (or 'session' for
 *               non-tool paths such as heartbeat).
 * @param reason Short reason string, e.g. 'connection_failed', 'timeout'.
 */
export function recordDaemonFallback(tool, reason) {
    daemonFallback.labels({ tool, reason }).inc();
}
// ── HTTP helpers ────────────────────────────────────────────────────────────
/**
 * The configured MCP HTTP path, cached at module load from `MCP_HTTP_PATH`
 * (same env var used by `resolveHttpOptions()` in `src/index.ts`).
 * Defaults to `/mcp` when the variable is unset (matching DEFAULT_HTTP_PATH
 * in server-types.ts).
 *
 * Cached so every call to `httpPathLabel()` does not re-read process.env.
 * Tests that need a different value must call `vi.resetModules()` and
 * re-import this module after setting the env var.
 */
const _mcpHttpPath = process.env['MCP_HTTP_PATH'] ?? '/mcp';
/** Logical path label for HTTP counters — collapses ad-hoc URLs into buckets.
 *
 * Uses `MCP_HTTP_PATH` (cached at module load) to recognise the MCP route.
 * Any path that matches the configured MCP path (or starts with it followed
 * by `/`) is labelled `mcp`. `/healthz` is labelled `/healthz`. Everything
 * else falls back to `other`.
 */
export function httpPathLabel(rawPath) {
    if (rawPath === undefined)
        return 'other';
    const noQuery = rawPath.split('?', 1)[0] ?? '';
    if (noQuery === '/healthz')
        return '/healthz';
    if (noQuery === _mcpHttpPath || noQuery.startsWith(_mcpHttpPath + '/'))
        return 'mcp';
    return 'other';
}
/** Status-class label for HTTP counters (`2xx`, `4xx`, `5xx`, or `other`). */
export function httpStatusClass(statusCode) {
    if (statusCode >= 200 && statusCode < 300)
        return '2xx';
    if (statusCode >= 400 && statusCode < 500)
        return '4xx';
    if (statusCode >= 500 && statusCode < 600)
        return '5xx';
    return 'other';
}
/** Record a completed HTTP request. Safe to call from middleware or handler. */
export function recordHttpRequest(rawPath, statusCode) {
    httpRequests
        .labels({ path: httpPathLabel(rawPath), status_class: httpStatusClass(statusCode) })
        .inc();
}
/** Record a bearer-auth failure. `reason` is free-form but low-cardinality. */
export function recordHttpAuthFailure(reason) {
    httpAuthFailures.labels({ reason }).inc();
}
/** Record a rate-limit rejection. */
export function recordHttpRateLimited() {
    httpRateLimited.inc();
}
// ── Stdio-mode exit hook ──────────────────────────────────────────────────────
/**
 * Push accumulated metrics to the OTLP collector on process exit (stdio mode).
 *
 * In HTTP mode metrics are already scraped by Prometheus, so OTLP push is
 * redundant and intentionally skipped — callers are expected to guard on
 * MCP_SERVER_MODE before invoking this function (see src/index.ts).
 *
 * Delegates to pushMetricsToOTLP() which is fire-and-forget with a 1s timeout
 * and never throws, ensuring process exit is not blocked.
 */
export async function pushMetricsOnExit() {
    await pushMetricsToOTLP();
}
//# sourceMappingURL=metrics.js.map