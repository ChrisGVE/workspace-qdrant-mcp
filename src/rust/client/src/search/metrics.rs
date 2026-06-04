//! Metrics hook for the search pipeline (WI-d4, #82).
//!
//! The shared pipeline records a "daemon fallback" event when embedding fails
//! and it degrades to the scroll-based fallback search. The concrete metric
//! backend (Prometheus in the MCP server) lives in the consuming crate, so the
//! pipeline depends only on this trait — never on a metrics registry.

/// Records search-pipeline fallback events.
///
/// Implemented by the consumer (e.g. the MCP server's Prometheus counter). The
/// no-op `()` implementation is provided for tests and call sites that do not
/// track metrics.
pub trait FallbackMetrics: Send + Sync {
    /// Record that `tool` fell back to degraded behaviour for `reason`.
    fn record_daemon_fallback(&self, tool: &str, reason: &str);
}

/// No-op implementation — pass `&()` when metrics are not needed.
impl FallbackMetrics for () {
    fn record_daemon_fallback(&self, _tool: &str, _reason: &str) {}
}
