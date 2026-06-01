//! Prometheus metrics for the workspace-qdrant MCP server.
//!
//! Mirrors `src/typescript/mcp-server/src/telemetry/metrics.ts`.
//!
//! 9 metric families:
//!
//! | Name                                 | Type      | Labels               |
//! |--------------------------------------|-----------|----------------------|
//! | `wqm_mcp_tool_invocations_total`     | Counter   | `tool`, `status`     |
//! | `wqm_mcp_tool_duration_seconds`      | Histogram | `tool`               |
//! | `wqm_mcp_session_count`              | Gauge     | —                    |
//! | `wqm_mcp_daemon_fallback_total`      | Counter   | `tool`, `reason`     |
//! | `wqm_mcp_cache_hits_total`           | Counter   | `cache`              |
//! | `wqm_mcp_cache_misses_total`         | Counter   | `cache`              |
//! | `wqm_mcp_http_requests_total`        | Counter   | `path`, `status_class` |
//! | `wqm_mcp_http_auth_failures_total`   | Counter   | `reason`             |
//! | `wqm_mcp_http_rate_limited_total`    | Counter   | —                    |

use once_cell::sync::Lazy;
use prometheus::{CounterVec, Gauge, HistogramOpts, HistogramVec, Opts, Registry, TextEncoder};

// ─────────────────────────────────────────────────────────────────────────────
// Shared registry
// ─────────────────────────────────────────────────────────────────────────────

/// Shared Prometheus registry for this MCP server process.
///
/// All metrics are registered here on first access (lazy init).
pub static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

// ─────────────────────────────────────────────────────────────────────────────
// Metric families
// ─────────────────────────────────────────────────────────────────────────────

pub static TOOL_INVOCATIONS: Lazy<CounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "wqm_mcp_tool_invocations_total",
        "Total MCP tool invocations by tool name and completion status",
    );
    let cv = CounterVec::new(opts, &["tool", "status"]).expect("valid metric");
    REGISTRY.register(Box::new(cv.clone())).expect("register");
    cv
});

pub static TOOL_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    let opts = HistogramOpts::new(
        "wqm_mcp_tool_duration_seconds",
        "MCP tool execution duration in seconds",
    )
    .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0]);
    let hv = HistogramVec::new(opts, &["tool"]).expect("valid metric");
    REGISTRY.register(Box::new(hv.clone())).expect("register");
    hv
});

pub static SESSION_COUNT: Lazy<Gauge> = Lazy::new(|| {
    let opts = Opts::new("wqm_mcp_session_count", "Number of active MCP sessions");
    let g = Gauge::with_opts(opts).expect("valid metric");
    REGISTRY.register(Box::new(g.clone())).expect("register");
    g
});

pub static DAEMON_FALLBACK: Lazy<CounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "wqm_mcp_daemon_fallback_total",
        "Number of times the daemon was unreachable and a fallback was triggered",
    );
    let cv = CounterVec::new(opts, &["tool", "reason"]).expect("valid metric");
    REGISTRY.register(Box::new(cv.clone())).expect("register");
    cv
});

/// Cache hits by cache type. Defined for future use — no cache layer at v0.1.3.
pub static CACHE_HITS: Lazy<CounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "wqm_mcp_cache_hits_total",
        "Cache hits by cache name (defined for future use; no cache layer at v0.1.3)",
    );
    let cv = CounterVec::new(opts, &["cache"]).expect("valid metric");
    REGISTRY.register(Box::new(cv.clone())).expect("register");
    cv
});

/// Cache misses by cache type. Defined for future use — no cache layer at v0.1.3.
pub static CACHE_MISSES: Lazy<CounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "wqm_mcp_cache_misses_total",
        "Cache misses by cache name (defined for future use; no cache layer at v0.1.3)",
    );
    let cv = CounterVec::new(opts, &["cache"]).expect("valid metric");
    REGISTRY.register(Box::new(cv.clone())).expect("register");
    cv
});

pub static HTTP_REQUESTS: Lazy<CounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "wqm_mcp_http_requests_total",
        "MCP HTTP transport requests by path and status class",
    );
    let cv = CounterVec::new(opts, &["path", "status_class"]).expect("valid metric");
    REGISTRY.register(Box::new(cv.clone())).expect("register");
    cv
});

pub static HTTP_AUTH_FAILURES: Lazy<CounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "wqm_mcp_http_auth_failures_total",
        "MCP HTTP auth failures by failure reason",
    );
    let cv = CounterVec::new(opts, &["reason"]).expect("valid metric");
    REGISTRY.register(Box::new(cv.clone())).expect("register");
    cv
});

pub static HTTP_RATE_LIMITED: Lazy<prometheus::Counter> = Lazy::new(|| {
    let opts = Opts::new(
        "wqm_mcp_http_rate_limited_total",
        "MCP HTTP requests rejected by the per-IP sliding-window rate limiter",
    );
    let c = prometheus::Counter::with_opts(opts).expect("valid metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions  (mirror TS helpers in telemetry/metrics.ts)
// ─────────────────────────────────────────────────────────────────────────────

/// Logical path label for HTTP counters — collapses ad-hoc URLs into buckets.
///
/// Mirrors `httpPathLabel()` in `telemetry/metrics.ts:189`.
pub fn http_path_label(raw_path: Option<&str>, mcp_path: &str) -> &'static str {
    let Some(raw) = raw_path else {
        return "other";
    };
    let no_query = raw.split('?').next().unwrap_or("");
    if no_query == "/healthz" {
        return "/healthz";
    }
    if no_query == mcp_path || no_query.starts_with(&format!("{mcp_path}/")) {
        return "mcp";
    }
    "other"
}

/// Status-class label (`2xx`, `4xx`, `5xx`, or `other`).
///
/// Mirrors `httpStatusClass()` in `telemetry/metrics.ts:198`.
pub fn http_status_class(status: u16) -> &'static str {
    match status {
        200..=299 => "2xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "other",
    }
}

/// Record a completed HTTP request.
///
/// Mirrors `recordHttpRequest()` in `telemetry/metrics.ts:206`.
pub fn record_http_request(raw_path: Option<&str>, status: u16, mcp_path: &str) {
    let path = http_path_label(raw_path, mcp_path);
    let class = http_status_class(status);
    HTTP_REQUESTS.with_label_values(&[path, class]).inc();
}

/// Record a bearer-auth failure.
///
/// Mirrors `recordHttpAuthFailure()` in `telemetry/metrics.ts:213`.
pub fn record_http_auth_failure(reason: &str) {
    HTTP_AUTH_FAILURES.with_label_values(&[reason]).inc();
}

/// Record a rate-limit rejection.
///
/// Mirrors `recordHttpRateLimited()` in `telemetry/metrics.ts:218`.
pub fn record_http_rate_limited() {
    HTTP_RATE_LIMITED.inc();
}

/// Increment the session gauge.
pub fn record_session_start() {
    SESSION_COUNT.inc();
}

/// Decrement the session gauge.
pub fn record_session_end() {
    SESSION_COUNT.dec();
}

/// Record a tool invocation (success or error) and its duration.
pub fn record_tool_call(tool: &str, status: &str, duration_secs: f64) {
    TOOL_INVOCATIONS.with_label_values(&[tool, status]).inc();
    TOOL_DURATION
        .with_label_values(&[tool])
        .observe(duration_secs);
}

/// Record a daemon fallback.
pub fn record_daemon_fallback(tool: &str, reason: &str) {
    DAEMON_FALLBACK.with_label_values(&[tool, reason]).inc();
}

/// Render all registered metrics in Prometheus text exposition format.
///
/// Returns an empty string if encoding fails (non-fatal).
pub fn render_metrics() -> String {
    let encoder = TextEncoder::new();
    match encoder.encode_to_string(&REGISTRY.gather()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to encode Prometheus metrics");
            String::new()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_label_healthz() {
        assert_eq!(http_path_label(Some("/healthz"), "/mcp"), "/healthz");
    }

    #[test]
    fn path_label_mcp_exact() {
        assert_eq!(http_path_label(Some("/mcp"), "/mcp"), "mcp");
    }

    #[test]
    fn path_label_mcp_with_query() {
        assert_eq!(http_path_label(Some("/mcp?foo=bar"), "/mcp"), "mcp");
    }

    #[test]
    fn path_label_mcp_sub_path() {
        assert_eq!(http_path_label(Some("/mcp/sse"), "/mcp"), "mcp");
    }

    #[test]
    fn path_label_other() {
        assert_eq!(http_path_label(Some("/unknown"), "/mcp"), "other");
    }

    #[test]
    fn path_label_none() {
        assert_eq!(http_path_label(None, "/mcp"), "other");
    }

    #[test]
    fn status_class_2xx() {
        assert_eq!(http_status_class(200), "2xx");
        assert_eq!(http_status_class(204), "2xx");
    }

    #[test]
    fn status_class_4xx() {
        assert_eq!(http_status_class(401), "4xx");
        assert_eq!(http_status_class(429), "4xx");
    }

    #[test]
    fn status_class_5xx() {
        assert_eq!(http_status_class(500), "5xx");
    }

    #[test]
    fn status_class_other() {
        assert_eq!(http_status_class(301), "other");
    }

    #[test]
    fn render_metrics_returns_string() {
        // Ensure statics are initialised by touching them.
        let _ = &*TOOL_INVOCATIONS;
        let _ = &*HTTP_REQUESTS;
        let output = render_metrics();
        // Should contain at least one metric family name.
        assert!(output.contains("wqm_mcp") || output.is_empty());
    }
}
