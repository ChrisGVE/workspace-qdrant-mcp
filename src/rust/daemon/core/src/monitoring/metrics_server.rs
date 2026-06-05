//! HTTP metrics endpoint and snapshot capture
//!
//! Provides the Prometheus /metrics endpoint via axum and a MetricsSnapshot
//! for CLI/API consumption.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};

use prometheus::core::Collector;

use super::metrics_core::METRICS;
use crate::config::PrometheusExportConfig;

/// HTTP metrics endpoint server
///
/// Serves Prometheus metrics at /metrics endpoint
pub struct MetricsServer {
    /// Resolved listen address
    addr: SocketAddr,
    /// Configured bind spec (for diagnostics)
    bind: String,
    /// Shutdown signal sender
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl MetricsServer {
    /// Create a metrics server bound to `port` on 0.0.0.0 (legacy helper).
    pub fn new(port: u16) -> Self {
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        Self {
            addr,
            bind: "0.0.0.0".to_string(),
            shutdown_tx: None,
        }
    }

    /// Create a metrics server from a PrometheusExportConfig.
    ///
    /// Returns Err if `config.bind` is not a parsable IP address.
    pub fn from_config(config: &PrometheusExportConfig) -> Result<Self, String> {
        let ip: IpAddr = config.bind.parse().map_err(|e| {
            format!(
                "telemetry.prometheus.bind '{}' is not a valid IP address: {e}",
                config.bind
            )
        })?;
        Ok(Self {
            addr: SocketAddr::new(ip, config.port),
            bind: config.bind.clone(),
            shutdown_tx: None,
        })
    }

    /// Start the metrics HTTP server
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use axum::{routing::get, Router};

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(tx);

        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler));

        tracing::info!("Metrics server listening on http://{}", self.addr);

        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        let local_addr = listener.local_addr()?;
        self.addr = local_addr;
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

    /// Get the configured port
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// Get the bound address. If constructed from a config using port 0,
    /// this reflects the actual OS-assigned port after `start()` runs.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Get the configured bind spec string.
    pub fn bind(&self) -> &str {
        &self.bind
    }
}

/// Handler for /metrics endpoint
async fn metrics_handler() -> impl axum::response::IntoResponse {
    match METRICS.encode() {
        Ok(metrics) => (
            axum::http::StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4",
            )],
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
        .map(|m| m.get_gauge().value() as i64)
        .sum()
}

/// Sum all counter values across labels for an `IntCounterVec`
fn sum_int_counter(metric: &impl Collector) -> u64 {
    metric
        .collect()
        .iter()
        .flat_map(|m| m.get_metric())
        .map(|m| m.get_counter().value() as u64)
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
            (key, m.get_gauge().value() as i64)
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
            (key, m.get_counter().value() as u64)
        })
        .collect()
}

/// Extract the first label value from a metric, or an empty string if none
fn first_label(m: &prometheus::proto::Metric) -> String {
    m.get_label()
        .first()
        .map_or_else(String::new, |l| l.value().to_string())
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
            .map(|m| m.get_gauge().value())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn from_config_parses_bind_address() {
        let cfg = PrometheusExportConfig {
            enabled: true,
            port: 9464,
            bind: "127.0.0.1".to_string(),
        };
        let server = MetricsServer::from_config(&cfg).expect("valid config");
        assert_eq!(server.port(), 9464);
        assert_eq!(server.bind(), "127.0.0.1");
        assert_eq!(server.addr().to_string(), "127.0.0.1:9464");
    }

    #[test]
    fn from_config_rejects_invalid_bind() {
        let cfg = PrometheusExportConfig {
            enabled: true,
            port: 9464,
            bind: "not-an-ip".to_string(),
        };
        assert!(MetricsServer::from_config(&cfg).is_err());
    }

    #[tokio::test]
    async fn server_serves_metrics_and_health_on_ephemeral_port() {
        let cfg = PrometheusExportConfig {
            enabled: true,
            port: 0,
            bind: "127.0.0.1".to_string(),
        };
        // Bind ephemeral port first so we know the address before spawning.
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind ephemeral");
        let addr = listener.local_addr().expect("local_addr");
        drop(listener);

        // Record one of the new telemetry metrics so it shows up in /metrics.
        METRICS.record_watcher_event("create");
        METRICS.record_grpc_call(
            "SystemService",
            "HealthCheck",
            true,
            Duration::from_millis(2),
        );

        let bound_cfg = PrometheusExportConfig {
            port: addr.port(),
            ..cfg
        };
        let mut server = MetricsServer::from_config(&bound_cfg).expect("valid cfg");
        let server_task = tokio::spawn(async move {
            // Ignore shutdown-related errors after signal.
            let _ = server.start().await;
        });

        // Give the server a moment to bind.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let client = reqwest::Client::new();
        let metrics_url = format!("http://{}/metrics", addr);
        let metrics_body = client
            .get(&metrics_url)
            .send()
            .await
            .expect("metrics request succeeds")
            .text()
            .await
            .expect("body utf8");
        assert!(metrics_body.contains("wqm_memexd_watcher_events_total"));
        assert!(metrics_body.contains("wqm_memexd_grpc_requests_total"));

        let health_url = format!("http://{}/health", addr);
        let health = client
            .get(&health_url)
            .send()
            .await
            .expect("health request succeeds");
        assert_eq!(health.status(), reqwest::StatusCode::OK);

        server_task.abort();
    }
}
