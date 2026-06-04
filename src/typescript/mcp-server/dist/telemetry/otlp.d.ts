/**
 * OTLP metrics push for stdio mode.
 *
 * On session exit, converts the prom-client registry snapshot to OTLP JSON
 * (protobuf-JSON encoding) and POSTs it to the configured collector.
 *
 * Design: direct fetch bridge rather than the OTel SDK stack.
 * Rationale:
 *   - Zero additional production dependencies (Node 20+ fetch is built-in).
 *   - One-shot fire-and-forget; no periodic reader, no SDK lifecycle.
 *   - Total added code < 150 lines; easy to audit and replace.
 *
 * Environment:
 *   OTEL_EXPORTER_OTLP_ENDPOINT  — base URL (default http://localhost:4318)
 *   OTEL_EXPORTER_OTLP_HEADERS   — comma-separated "key=value" pairs
 */
/**
 * Collect the current prom-client metrics snapshot, convert to OTLP JSON,
 * and POST to the configured collector endpoint.
 *
 * Always resolves — never throws.  All errors are swallowed after a
 * console.debug log so that the stdio-mode exit path is never blocked.
 *
 * Uses a 1-second AbortSignal timeout so a slow or unreachable collector
 * does not hold process exit for more than ~1s.
 */
export declare function pushMetricsToOTLP(): Promise<void>;
//# sourceMappingURL=otlp.d.ts.map