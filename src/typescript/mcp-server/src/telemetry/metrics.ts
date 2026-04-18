/**
 * Prometheus metrics for the workspace-qdrant MCP server.
 *
 * Defines 6 metric families:
 *   wqm_mcp_tool_invocations_total   - Counter, labels [tool, status]
 *   wqm_mcp_tool_duration_seconds    - Histogram, label [tool], buckets [0.01…5]
 *   wqm_mcp_session_count            - Gauge
 *   wqm_mcp_daemon_fallback_total    - Counter, labels [tool, reason]
 *   wqm_mcp_cache_hits_total         - Counter, label [cache]
 *   wqm_mcp_cache_misses_total       - Counter, label [cache]
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
  labelNames: ['tool', 'status'] as const,
  registers: [register],
});

export const toolDuration = new Histogram({
  name: 'wqm_mcp_tool_duration_seconds',
  help: 'MCP tool execution duration in seconds',
  labelNames: ['tool'] as const,
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
  labelNames: ['tool', 'reason'] as const,
  registers: [register],
});

/**
 * Cache hits by cache type.
 * Defined for future use — no in-process cache exists at v0.1.3.
 */
export const cacheHits = new Counter({
  name: 'wqm_mcp_cache_hits_total',
  help: 'Cache hits by cache name (defined for future use; no cache layer at v0.1.3)',
  labelNames: ['cache'] as const,
  registers: [register],
});

/**
 * Cache misses by cache type.
 * Defined for future use — no in-process cache exists at v0.1.3.
 */
export const cacheMisses = new Counter({
  name: 'wqm_mcp_cache_misses_total',
  help: 'Cache misses by cache name (defined for future use; no cache layer at v0.1.3)',
  labelNames: ['cache'] as const,
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
export async function withToolMetrics<T>(toolName: string, fn: () => Promise<T>): Promise<T> {
  const end = toolDuration.startTimer({ tool: toolName });
  try {
    const result = await fn();
    end();
    toolInvocations.labels({ tool: toolName, status: 'success' }).inc();
    return result;
  } catch (err: unknown) {
    end();
    toolInvocations.labels({ tool: toolName, status: 'error' }).inc();
    throw err;
  }
}

// ── Session helpers ───────────────────────────────────────────────────────────

/** Call once when a new MCP session starts. */
export function recordSessionStart(): void {
  sessionCount.inc();
}

/** Call once when the current MCP session ends. */
export function recordSessionEnd(): void {
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
export function recordDaemonFallback(tool: string, reason: string): void {
  daemonFallback.labels({ tool, reason }).inc();
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
export async function pushMetricsOnExit(): Promise<void> {
  await pushMetricsToOTLP();
}
