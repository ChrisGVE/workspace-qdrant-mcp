//! HTTP metrics endpoint and snapshot capture
//!
//! Provides the Prometheus /metrics endpoint via axum and a MetricsSnapshot
//! for CLI/API consumption.

use std::collections::HashMap;

use prometheus::core::Collector;

use super::metrics_core::METRICS;

/// HTTP metrics endpoint server
///
/// Serves Prometheus metrics at /metrics endpoint
pub struct MetricsServer {
    /// Port to listen on
    port: u16,
    /// Shutdown signal sender
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl MetricsServer {
    /// Create a new metrics server on the given port
    pub fn new(port: u16) -> Self {
        Self {
            port,
            shutdown_tx: None,
        }
    }

    /// Start the metrics HTTP server
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use axum::{routing::get, Router};
        use std::net::SocketAddr;

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(tx);

        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler));

        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        tracing::info!("Metrics server listening on http://{}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await?;

        Ok(())
    }

    /// Shutdown the metrics server
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }

    /// Get the port
    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Handler for /metrics endpoint
async fn metrics_handler() -> impl axum::response::IntoResponse {
    match METRICS.encode() {
        Ok(metrics) => (
            axum::http::StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            metrics,
        ),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "text/plain")],
            format!("Error encoding metrics: {}", e),
        ),
    }
}

/// Handler for /health endpoint
async fn health_handler() -> impl axum::response::IntoResponse {
    (axum::http::StatusCode::OK, "OK")
}

/// Sum all gauge values across labels for an `IntGaugeVec`
fn sum_int_gauge(metric: &impl Collector) -> i64 {
    metric
        .collect()
        .iter()
        .flat_map(|m| m.get_metric())
        .map(|m| m.get_gauge().get_value() as i64)
        .sum()
}

/// Sum all counter values across labels for an `IntCounterVec`
fn sum_int_counter(metric: &impl Collector) -> u64 {
    metric
        .collect()
        .iter()
        .flat_map(|m| m.get_metric())
        .map(|m| m.get_counter().get_value() as u64)
        .sum()
}

/// Collect a labeled gauge into a map keyed by the first label value
fn labeled_gauge_map(metric: &impl Collector) -> HashMap<String, i64> {
    metric
        .collect()
        .iter()
        .flat_map(|m| m.get_metric())
        .map(|m| {
            let key = first_label(m);
            (key, m.get_gauge().get_value() as i64)
        })
        .collect()
}

/// Collect a labeled counter into a map keyed by the first label value
fn labeled_counter_map(metric: &impl Collector) -> HashMap<String, u64> {
    metric
        .collect()
        .iter()
        .flat_map(|m| m.get_metric())
        .map(|m| {
            let key = first_label(m);
            (key, m.get_counter().get_value() as u64)
        })
        .collect()
}

/// Extract the first label value from a metric, or an empty string if none
fn first_label(m: &prometheus::proto::Metric) -> String {
    m.get_label()
        .first()
        .map_or_else(String::new, |l| l.get_value().to_string())
}

/// Metrics snapshot for CLI/API consumption
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
    /// Daemon uptime in seconds
    pub uptime_seconds: f64,
    /// Active session count
    pub active_sessions: i64,
    /// Total sessions lifetime
    pub total_sessions_lifetime: u64,
    /// Queue depths by priority
    pub queue_depths: HashMap<String, i64>,
    /// Total items processed
    pub total_items_processed: u64,
    /// Error counts by type
    pub error_counts: HashMap<String, u64>,
    /// Per-tenant document counts
    pub tenant_documents: HashMap<String, i64>,
}

impl MetricsSnapshot {
    /// Create a snapshot from current metrics
    pub fn capture() -> Self {
        let metrics = &*METRICS;

        let uptime_seconds = metrics
            .uptime_seconds
            .collect()
            .first()
            .and_then(|m| m.get_metric().first())
            .map(|m| m.get_gauge().get_value())
            .unwrap_or(0.0);

        Self {
            uptime_seconds,
            active_sessions: sum_int_gauge(&metrics.active_sessions),
            total_sessions_lifetime: sum_int_counter(&metrics.total_sessions),
            queue_depths: labeled_gauge_map(&metrics.queue_depth),
            total_items_processed: sum_int_counter(&metrics.queue_items_processed_total),
            error_counts: labeled_counter_map(&metrics.ingestion_errors_total),
            tenant_documents: labeled_gauge_map(&metrics.tenant_documents_total),
        }
    }
}
