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
/** Shared Prometheus registry for this MCP server process. */
export declare const register: Registry<"text/plain; version=0.0.4; charset=utf-8">;
export declare const toolInvocations: Counter<"status" | "tool">;
export declare const toolDuration: Histogram<"tool">;
export declare const sessionCount: Gauge<string>;
export declare const daemonFallback: Counter<"reason" | "tool">;
/**
 * Cache hits by cache type.
 * Defined for future use — no in-process cache exists at v0.1.3.
 */
export declare const cacheHits: Counter<"cache">;
/**
 * Cache misses by cache type.
 * Defined for future use — no in-process cache exists at v0.1.3.
 */
export declare const cacheMisses: Counter<"cache">;
/**
 * Total HTTP requests handled by the MCP listener, bucketed by logical path
 * (`/mcp`, `/healthz`, `other`) and status class (`2xx`, `4xx`, `5xx`). These
 * are only incremented in HTTP mode; stdio deployments leave them at zero.
 */
export declare const httpRequests: Counter<"path" | "status_class">;
/**
 * HTTP bearer-auth failures. Labelled by failure reason (`missing_header`,
 * `invalid_token`, `not_configured`) so dashboards can distinguish
 * mis-configured clients from probable attacks.
 */
export declare const httpAuthFailures: Counter<"reason">;
/**
 * HTTP rate-limit hits. Single counter — alerting on sustained non-zero rate
 * catches runaway clients or brute-force attempts.
 */
export declare const httpRateLimited: Counter<string>;
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
export declare function withToolMetrics<T>(toolName: string, fn: () => Promise<T>): Promise<T>;
/** Call once when a new MCP session starts. */
export declare function recordSessionStart(): void;
/** Call once when the current MCP session ends. */
export declare function recordSessionEnd(): void;
/**
 * Increment the daemon-fallback counter.
 *
 * @param tool   Name of the tool that triggered the fallback (or 'session' for
 *               non-tool paths such as heartbeat).
 * @param reason Short reason string, e.g. 'connection_failed', 'timeout'.
 */
export declare function recordDaemonFallback(tool: string, reason: string): void;
/** Logical path label for HTTP counters — collapses ad-hoc URLs into buckets.
 *
 * Uses `MCP_HTTP_PATH` (cached at module load) to recognise the MCP route.
 * Any path that matches the configured MCP path (or starts with it followed
 * by `/`) is labelled `mcp`. `/healthz` is labelled `/healthz`. Everything
 * else falls back to `other`.
 */
export declare function httpPathLabel(rawPath: string | undefined): string;
/** Status-class label for HTTP counters (`2xx`, `4xx`, `5xx`, or `other`). */
export declare function httpStatusClass(statusCode: number): string;
/** Record a completed HTTP request. Safe to call from middleware or handler. */
export declare function recordHttpRequest(rawPath: string | undefined, statusCode: number): void;
/** Record a bearer-auth failure. `reason` is free-form but low-cardinality. */
export declare function recordHttpAuthFailure(reason: string): void;
/** Record a rate-limit rejection. */
export declare function recordHttpRateLimited(): void;
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
export declare function pushMetricsOnExit(): Promise<void>;
//# sourceMappingURL=metrics.d.ts.map