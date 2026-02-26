//! HTTP metrics endpoint and snapshot capture
//!
//! Provides the Prometheus /metrics endpoint via axum and a MetricsSnapshot
//! for CLI/API consumption.

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
    pub queue_depths: std::collections::HashMap<String, i64>,
    /// Total items processed
    pub total_items_processed: u64,
    /// Error counts by type
    pub error_counts: std::collections::HashMap<String, u64>,
    /// Per-tenant document counts
    pub tenant_documents: std::collections::HashMap<String, i64>,
}

impl MetricsSnapshot {
    /// Create a snapshot from current metrics
    pub fn capture() -> Self {
        let metrics = &*METRICS;

        // Sum active sessions across all labels
        let active_sessions = metrics
            .active_sessions
            .collect()
            .iter()
            .flat_map(|m| m.get_metric())
            .map(|m| m.get_gauge().get_value() as i64)
            .sum();

        // Sum total sessions across all labels
        let total_sessions_lifetime = metrics
            .total_sessions
            .collect()
            .iter()
            .flat_map(|m| m.get_metric())
            .map(|m| m.get_counter().get_value() as u64)
            .sum();

        // Get queue depths by priority
        let queue_depths: std::collections::HashMap<String, i64> = metrics
            .queue_depth
            .collect()
            .iter()
            .flat_map(|m| m.get_metric())
            .map(|m| {
                let labels: Vec<_> = m.get_label().iter().map(|l| l.get_value()).collect();
                let key = labels.first().cloned().unwrap_or_default().to_string();
                (key, m.get_gauge().get_value() as i64)
            })
            .collect();

        // Sum total items processed
        let total_items_processed = metrics
            .queue_items_processed_total
            .collect()
            .iter()
            .flat_map(|m| m.get_metric())
            .map(|m| m.get_counter().get_value() as u64)
            .sum();

        // Get error counts by type
        let error_counts: std::collections::HashMap<String, u64> = metrics
            .ingestion_errors_total
            .collect()
            .iter()
            .flat_map(|m| m.get_metric())
            .map(|m| {
                let labels: Vec<_> = m.get_label().iter().map(|l| l.get_value()).collect();
                let key = labels.first().cloned().unwrap_or_default().to_string();
                (key, m.get_counter().get_value() as u64)
            })
            .collect();

        // Get tenant document counts
        let tenant_documents: std::collections::HashMap<String, i64> = metrics
            .tenant_documents_total
            .collect()
            .iter()
            .flat_map(|m| m.get_metric())
            .map(|m| {
                let labels: Vec<_> = m.get_label().iter().map(|l| l.get_value()).collect();
                let key = labels.first().cloned().unwrap_or_default().to_string();
                (key, m.get_gauge().get_value() as i64)
            })
            .collect();

        // Get uptime
        let uptime_seconds = metrics
            .uptime_seconds
            .collect()
            .first()
            .and_then(|m| m.get_metric().first())
            .map(|m| m.get_gauge().get_value())
            .unwrap_or(0.0);

        Self {
            uptime_seconds,
            active_sessions,
            total_sessions_lifetime,
            queue_depths,
            total_items_processed,
            error_counts,
            tenant_documents,
        }
    }
}
