//! Prometheus metrics for the embedding-provider subsystem.
//!
//! Three families per PRD §6 (task 12):
//!
//! | Metric                                       | Type      | Labels                          |
//! |----------------------------------------------|-----------|---------------------------------|
//! | `wqm_memexd_embedding_provider_requests_total`   | Counter   | provider, model, status_class   |
//! | `wqm_memexd_embedding_provider_rate_limit_waits_total` | Counter | provider                  |
//! | `wqm_memexd_embedding_provider_latency_seconds`  | Histogram | provider, model                 |
//!
//! The `provider` label is bounded-cardinality: it is the value returned by
//! [`DenseProvider::metrics_label`] (one of `fastembed`, `openai`,
//! `azure_openai`, `lmstudio`, `llama_cpp`, `openai_compatible_other`). The
//! `status_class` label takes one of `2xx`, `4xx`, `5xx`, `error` (transport-
//! level failure, no HTTP response).
//!
//! Metrics are owned by a process-global `EmbeddingProviderMetrics` registered
//! against `prometheus::default_registry()` on first access. The metrics
//! server's snapshotter scrapes the same registry, so no additional wiring is
//! required.

use once_cell::sync::Lazy;
use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry};

/// Latency histogram buckets in seconds — covers fast-local providers
/// (fastembed/llama_cpp) up to long remote tail latencies (10s+).
const LATENCY_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];

/// Process-global handle to the three embedding-provider metric families.
pub struct EmbeddingProviderMetrics {
    pub requests_total: IntCounterVec,
    pub rate_limit_waits_total: IntCounterVec,
    pub latency_seconds: HistogramVec,
}

impl EmbeddingProviderMetrics {
    fn build_and_register(registry: &Registry) -> Self {
        let requests_total = IntCounterVec::new(
            Opts::new(
                "wqm_memexd_embedding_provider_requests_total",
                "Total embedding provider requests by provider, model, and HTTP status class",
            ),
            &["provider", "model", "status_class"],
        )
        .expect("metric can be created");
        let rate_limit_waits_total = IntCounterVec::new(
            Opts::new(
                "wqm_memexd_embedding_provider_rate_limit_waits_total",
                "Total times the embedding provider waited on a rate-limit window",
            ),
            &["provider"],
        )
        .expect("metric can be created");
        let latency_seconds = HistogramVec::new(
            HistogramOpts::new(
                "wqm_memexd_embedding_provider_latency_seconds",
                "Latency of embedding provider requests in seconds",
            )
            .buckets(LATENCY_BUCKETS.to_vec()),
            &["provider", "model"],
        )
        .expect("metric can be created");

        // Register; `AlreadyReg` is treated as a no-op so test harnesses that
        // import this module from multiple binaries don't panic.
        for collector in [
            Box::new(requests_total.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(rate_limit_waits_total.clone()),
            Box::new(latency_seconds.clone()),
        ] {
            match registry.register(collector) {
                Ok(()) => {}
                Err(prometheus::Error::AlreadyReg) => {}
                Err(e) => panic!("failed to register embedding_provider metric: {e}"),
            }
        }

        Self {
            requests_total,
            rate_limit_waits_total,
            latency_seconds,
        }
    }
}

static METRICS: Lazy<EmbeddingProviderMetrics> =
    Lazy::new(|| EmbeddingProviderMetrics::build_and_register(prometheus::default_registry()));

/// Access the process-global embedding-provider metrics.
///
/// First call lazily registers the three metric families against
/// `prometheus::default_registry()`. Subsequent calls return the same handle.
pub fn embedding_provider_metrics() -> &'static EmbeddingProviderMetrics {
    &METRICS
}

/// Map an HTTP status code or transport error to the bounded `status_class`
/// label value used by `requests_total`.
pub fn status_class(status: Option<u16>) -> &'static str {
    match status {
        Some(s) if (200..300).contains(&s) => "2xx",
        Some(s) if (400..500).contains(&s) => "4xx",
        Some(s) if (500..600).contains(&s) => "5xx",
        _ => "error",
    }
}

/// Record a completed provider request.
pub fn record_request(provider: &str, model: &str, status: Option<u16>, latency_secs: f64) {
    let m = embedding_provider_metrics();
    m.requests_total
        .with_label_values(&[provider, model, status_class(status)])
        .inc();
    m.latency_seconds
        .with_label_values(&[provider, model])
        .observe(latency_secs);
}

/// Record a rate-limit wait observed by the provider's rate-limit adapter.
pub fn record_rate_limit_wait(provider: &str) {
    embedding_provider_metrics()
        .rate_limit_waits_total
        .with_label_values(&[provider])
        .inc();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §7.10 test 1 — metrics are observable without panicking; double-call
    /// must not double-register.
    #[test]
    fn test_embedding_provider_metrics_registered() {
        // First access registers; second must not panic on AlreadyReg.
        let _ = embedding_provider_metrics();
        record_request("openai", "text-embedding-3-small", Some(200), 0.123);
        record_request("openai", "text-embedding-3-small", Some(429), 0.05);
        record_request("openai", "text-embedding-3-small", None, 1.5);
        record_rate_limit_wait("openai");
        let _again = embedding_provider_metrics();
    }

    /// §7.10 test 2 — provider-label values are drawn from a fixed enum so
    /// the `provider` Prometheus label has bounded cardinality. We exercise
    /// `classify_provider`, which is the single source the openai provider
    /// uses to compute its `metrics_label()`.
    #[test]
    fn test_metrics_label_is_bounded_cardinality() {
        use crate::embedding::provider::openai::classify_provider;

        let cases: &[(&str, &str)] = &[
            ("https://api.openai.com", "openai"),
            ("https://example.openai.azure.com", "azure_openai"),
            ("http://localhost:1234", "lmstudio"),
            ("http://127.0.0.1:8080", "llama_cpp"),
            ("https://example.com/v1", "openai_compatible_other"),
        ];
        for (url, expected) in cases {
            assert_eq!(classify_provider(url), *expected, "url={url}");
        }
        // FastEmbed label is owned by FastEmbedProvider and asserted in its
        // module's own test (`fastembed.rs`).
    }

    #[test]
    fn test_status_class_mapping() {
        assert_eq!(status_class(Some(200)), "2xx");
        assert_eq!(status_class(Some(204)), "2xx");
        assert_eq!(status_class(Some(401)), "4xx");
        assert_eq!(status_class(Some(429)), "4xx");
        assert_eq!(status_class(Some(503)), "5xx");
        assert_eq!(status_class(None), "error");
        assert_eq!(status_class(Some(100)), "error");
    }
}
