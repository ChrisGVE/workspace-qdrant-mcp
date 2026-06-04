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
import { register } from './metrics.js';
// ── Config helpers ────────────────────────────────────────────────────────────
/** Read the OTLP endpoint from env; strip trailing slash. */
function resolveEndpoint() {
    const raw = process.env['OTEL_EXPORTER_OTLP_ENDPOINT'] ?? 'http://localhost:4318';
    return raw.replace(/\/$/, '');
}
/** Parse OTEL_EXPORTER_OTLP_HEADERS into a plain object. */
function resolveHeaders() {
    const raw = process.env['OTEL_EXPORTER_OTLP_HEADERS'];
    if (!raw)
        return {};
    const out = {};
    for (const pair of raw.split(',')) {
        const eq = pair.indexOf('=');
        if (eq < 1)
            continue;
        const k = pair.slice(0, eq).trim();
        const v = pair.slice(eq + 1).trim();
        if (k)
            out[k] = v;
    }
    return out;
}
// ── prom-client → OTLP conversion ────────────────────────────────────────────
/** Convert prom-client label map to OTLP attributes array. */
function labelsToAttributes(labels) {
    return Object.entries(labels).map(([key, value]) => ({
        key,
        value: { stringValue: value },
    }));
}
/** Convert epoch ms to OTLP nanosecond string. */
function msToNs(ms) {
    return String(ms * 1_000_000);
}
/**
 * Build OTLP Sum (monotonic) from a prom-client counter family.
 * Groups all values into data points; skips the _total suffix copy
 * that prom-client adds for Prometheus 2.x compatibility.
 */
function counterToOtlp(family) {
    const nowNs = msToNs(Date.now());
    const dataPoints = family.values
        .filter((v) => v.metricName === undefined || v.metricName === family.name)
        .map((v) => ({
        attributes: labelsToAttributes(v.labels),
        startTimeUnixNano: '0',
        timeUnixNano: nowNs,
        asDouble: v.value,
    }));
    return {
        name: family.name,
        description: family.help,
        sum: {
            dataPoints,
            aggregationTemporality: 2, // AGGREGATION_TEMPORALITY_CUMULATIVE
            isMonotonic: true,
        },
    };
}
/**
 * Build OTLP Gauge from a prom-client gauge family.
 * Each value becomes one data point.
 */
function gaugeToOtlp(family) {
    const nowNs = msToNs(Date.now());
    const dataPoints = family.values.map((v) => ({
        attributes: labelsToAttributes(v.labels),
        timeUnixNano: nowNs,
        asDouble: v.value,
    }));
    return {
        name: family.name,
        description: family.help,
        gauge: { dataPoints },
    };
}
/** Group histogram metric values by non-le label fingerprint. */
function groupHistogramValues(values, familyName) {
    const groups = new Map();
    for (const v of values) {
        const baseLabels = {};
        for (const [k, val] of Object.entries(v.labels)) {
            if (k !== 'le')
                baseLabels[k] = val;
        }
        const fingerprint = JSON.stringify(baseLabels);
        if (!groups.has(fingerprint))
            groups.set(fingerprint, { labels: baseLabels, buckets: [], sum: 0, count: 0 });
        const g = groups.get(fingerprint);
        const metricName = v.metricName ?? familyName;
        if (metricName.endsWith('_sum'))
            g.sum = v.value;
        else if (metricName.endsWith('_count'))
            g.count = v.value;
        else if (v.labels['le'] !== undefined) {
            const le = v.labels['le'] === '+Inf' ? Infinity : parseFloat(v.labels['le'] ?? '0');
            g.buckets.push({ le, count: v.value });
        }
    }
    return groups;
}
/** Convert a single BucketGroup to an OTLP histogram data point. */
function bucketGroupToDataPoint(g, nowNs) {
    const sorted = g.buckets.filter((b) => isFinite(b.le)).sort((a, b) => a.le - b.le);
    const infBucket = g.buckets.find((b) => !isFinite(b.le));
    return {
        attributes: labelsToAttributes(g.labels),
        startTimeUnixNano: '0',
        timeUnixNano: nowNs,
        count: String(g.count),
        sum: g.sum,
        explicitBounds: sorted.map((b) => b.le),
        bucketCounts: [...sorted.map((b) => b.count), infBucket?.count ?? g.count].map(String),
    };
}
/**
 * Build OTLP Histogram from a prom-client histogram family.
 *
 * prom-client exposes histogram data as individual _bucket / _sum / _count
 * values sharing a label set.  We group by the non-le labels, then
 * reconstruct explicit bounds + bucket counts.
 */
function histogramToOtlp(family) {
    const nowNs = msToNs(Date.now());
    const groups = groupHistogramValues(family.values, family.name);
    const dataPoints = Array.from(groups.values()).map((g) => bucketGroupToDataPoint(g, nowNs));
    return {
        name: family.name,
        description: family.help,
        histogram: { dataPoints, aggregationTemporality: 2 },
    };
}
/** Convert a single prom-client metric family to an OTLP Metric object. */
function familyToOtlp(family) {
    switch (family.type) {
        case 'counter':
            return counterToOtlp(family);
        case 'gauge':
            return gaugeToOtlp(family);
        case 'histogram':
            return histogramToOtlp(family);
        default:
            // summary and untyped are rare; skip rather than emit malformed data
            return null;
    }
}
/** Build the top-level OTLP ExportMetricsServiceRequest body. */
function buildOtlpPayload(families) {
    const metrics = families
        .map(familyToOtlp)
        .filter((m) => m !== null);
    return {
        resourceMetrics: [
            {
                resource: {
                    attributes: [
                        { key: 'service.name', value: { stringValue: 'workspace-qdrant-mcp' } },
                        { key: 'telemetry.sdk.language', value: { stringValue: 'nodejs' } },
                    ],
                },
                scopeMetrics: [
                    {
                        scope: { name: 'workspace-qdrant-mcp', version: '0.1.3' },
                        metrics,
                    },
                ],
            },
        ],
    };
}
// ── Public API ────────────────────────────────────────────────────────────────
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
export async function pushMetricsToOTLP() {
    try {
        const endpoint = resolveEndpoint();
        const extraHeaders = resolveHeaders();
        const url = `${endpoint}/v1/metrics`;
        const families = (await register.getMetricsAsJSON());
        const body = JSON.stringify(buildOtlpPayload(families));
        await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...extraHeaders,
            },
            body,
            signal: AbortSignal.timeout(1000),
        });
    }
    catch (err) {
        // Fire-and-forget: log at debug level and continue.
        // In stdio mode console.debug is suppressed so this is silent in production.
        console.debug('[wqm-otlp] push failed (ignored):', err);
    }
}
//# sourceMappingURL=otlp.js.map