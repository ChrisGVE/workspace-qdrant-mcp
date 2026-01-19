//! Prometheus metrics for the workspace-qdrant-mcp daemon
//!
//! This module provides comprehensive metrics collection for:
//! - Session tracking (active sessions, duration, lifecycle events)
//! - Queue metrics (depth, processing time, items processed)
//! - Per-tenant metrics (documents, search requests, storage)
//!
//! Task 412: Add monitoring and observability for multi-tenant operations

use once_cell::sync::Lazy;
use prometheus::{
    self, core::Collector, Encoder, GaugeVec, HistogramVec, IntCounterVec, IntGaugeVec,
    Opts, Registry, TextEncoder,
};

/// Global metrics registry
pub static METRICS: Lazy<DaemonMetrics> = Lazy::new(DaemonMetrics::new);

/// Daemon metrics collector
///
/// Provides Prometheus-format metrics for monitoring the daemon's performance
/// and health across all multi-tenant operations.
pub struct DaemonMetrics {
    /// Custom registry for daemon metrics
    pub registry: Registry,

    // Session tracking metrics
    /// Number of active sessions by project_id and priority
    /// Labels: project_id, priority
    pub active_sessions: IntGaugeVec,

    /// Total number of sessions created (lifetime) by project_id
    /// Labels: project_id
    pub total_sessions: IntCounterVec,

    /// Session duration in seconds
    /// Labels: project_id
    pub session_duration_seconds: HistogramVec,

    // Queue metrics
    /// Current queue depth by priority and collection
    /// Labels: priority, collection
    pub queue_depth: IntGaugeVec,

    /// Queue processing time in seconds by priority
    /// Labels: priority
    pub queue_processing_time_seconds: HistogramVec,

    /// Total items processed by priority and status
    /// Labels: priority, status (success, failure, skipped)
    pub queue_items_processed_total: IntCounterVec,

    // Per-tenant metrics
    /// Total documents per tenant and collection
    /// Labels: tenant_id, collection
    pub tenant_documents_total: IntGaugeVec,

    /// Total search requests per tenant
    /// Labels: tenant_id
    pub tenant_search_requests_total: IntCounterVec,

    /// Storage bytes per tenant (estimated)
    /// Labels: tenant_id
    pub tenant_storage_bytes: GaugeVec,

    // System metrics
    /// Daemon uptime in seconds
    pub uptime_seconds: GaugeVec,

    /// Total ingestion errors
    /// Labels: error_type
    pub ingestion_errors_total: IntCounterVec,

    /// Heartbeat latency in seconds
    /// Labels: project_id
    pub heartbeat_latency_seconds: HistogramVec,
}

impl DaemonMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let registry = Registry::new();

        // Session tracking metrics
        let active_sessions = IntGaugeVec::new(
            Opts::new(
                "memexd_active_sessions",
                "Number of active sessions by project and priority",
            )
            .namespace("memexd"),
            &["project_id", "priority"],
        )
        .expect("metric can be created");

        let total_sessions = IntCounterVec::new(
            Opts::new(
                "memexd_total_sessions",
                "Total number of sessions created (lifetime)",
            )
            .namespace("memexd"),
            &["project_id"],
        )
        .expect("metric can be created");

        let session_duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "memexd_session_duration_seconds",
                "Session duration in seconds",
            )
            .namespace("memexd")
            .buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]),
            &["project_id"],
        )
        .expect("metric can be created");

        // Queue metrics
        let queue_depth = IntGaugeVec::new(
            Opts::new(
                "memexd_queue_depth",
                "Current queue depth by priority and collection",
            )
            .namespace("memexd"),
            &["priority", "collection"],
        )
        .expect("metric can be created");

        let queue_processing_time_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "memexd_queue_processing_time_seconds",
                "Queue item processing time in seconds",
            )
            .namespace("memexd")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]),
            &["priority"],
        )
        .expect("metric can be created");

        let queue_items_processed_total = IntCounterVec::new(
            Opts::new(
                "memexd_queue_items_processed_total",
                "Total items processed by priority and status",
            )
            .namespace("memexd"),
            &["priority", "status"],
        )
        .expect("metric can be created");

        // Per-tenant metrics
        let tenant_documents_total = IntGaugeVec::new(
            Opts::new(
                "memexd_tenant_documents_total",
                "Total documents per tenant and collection",
            )
            .namespace("memexd"),
            &["tenant_id", "collection"],
        )
        .expect("metric can be created");

        let tenant_search_requests_total = IntCounterVec::new(
            Opts::new(
                "memexd_tenant_search_requests_total",
                "Total search requests per tenant",
            )
            .namespace("memexd"),
            &["tenant_id"],
        )
        .expect("metric can be created");

        let tenant_storage_bytes = GaugeVec::new(
            Opts::new(
                "memexd_tenant_storage_bytes",
                "Estimated storage bytes per tenant",
            )
            .namespace("memexd"),
            &["tenant_id"],
        )
        .expect("metric can be created");

        // System metrics
        let uptime_seconds = GaugeVec::new(
            Opts::new("memexd_uptime_seconds", "Daemon uptime in seconds").namespace("memexd"),
            &[],
        )
        .expect("metric can be created");

        let ingestion_errors_total = IntCounterVec::new(
            Opts::new(
                "memexd_ingestion_errors_total",
                "Total ingestion errors by error type",
            )
            .namespace("memexd"),
            &["error_type"],
        )
        .expect("metric can be created");

        let heartbeat_latency_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "memexd_heartbeat_latency_seconds",
                "Heartbeat processing latency in seconds",
            )
            .namespace("memexd")
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["project_id"],
        )
        .expect("metric can be created");

        // Register all metrics
        registry
            .register(Box::new(active_sessions.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(total_sessions.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(session_duration_seconds.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(queue_depth.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(queue_processing_time_seconds.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(queue_items_processed_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(tenant_documents_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(tenant_search_requests_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(tenant_storage_bytes.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(uptime_seconds.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(ingestion_errors_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(heartbeat_latency_seconds.clone()))
            .expect("metric can be registered");

        Self {
            registry,
            active_sessions,
            total_sessions,
            session_duration_seconds,
            queue_depth,
            queue_processing_time_seconds,
            queue_items_processed_total,
            tenant_documents_total,
            tenant_search_requests_total,
            tenant_storage_bytes,
            uptime_seconds,
            ingestion_errors_total,
            heartbeat_latency_seconds,
        }
    }

    /// Encode all metrics in Prometheus text format
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }

    // Session tracking helpers

    /// Record a new session start
    pub fn session_started(&self, project_id: &str, priority: &str) {
        self.active_sessions
            .with_label_values(&[project_id, priority])
            .inc();
        self.total_sessions.with_label_values(&[project_id]).inc();
    }

    /// Record a session end with duration
    pub fn session_ended(&self, project_id: &str, priority: &str, duration_secs: f64) {
        self.active_sessions
            .with_label_values(&[project_id, priority])
            .dec();
        self.session_duration_seconds
            .with_label_values(&[project_id])
            .observe(duration_secs);
    }

    /// Update session priority
    pub fn session_priority_changed(&self, project_id: &str, old_priority: &str, new_priority: &str) {
        self.active_sessions
            .with_label_values(&[project_id, old_priority])
            .dec();
        self.active_sessions
            .with_label_values(&[project_id, new_priority])
            .inc();
    }

    // Queue tracking helpers

    /// Update queue depth
    pub fn set_queue_depth(&self, priority: &str, collection: &str, depth: i64) {
        self.queue_depth
            .with_label_values(&[priority, collection])
            .set(depth);
    }

    /// Record a queue item processed
    pub fn queue_item_processed(
        &self,
        priority: &str,
        status: &str,
        processing_time_secs: f64,
    ) {
        self.queue_items_processed_total
            .with_label_values(&[priority, status])
            .inc();
        self.queue_processing_time_seconds
            .with_label_values(&[priority])
            .observe(processing_time_secs);
    }

    // Tenant tracking helpers

    /// Set tenant document count
    pub fn set_tenant_documents(&self, tenant_id: &str, collection: &str, count: i64) {
        self.tenant_documents_total
            .with_label_values(&[tenant_id, collection])
            .set(count);
    }

    /// Record a tenant search request
    pub fn tenant_search(&self, tenant_id: &str) {
        self.tenant_search_requests_total
            .with_label_values(&[tenant_id])
            .inc();
    }

    /// Set tenant storage bytes
    pub fn set_tenant_storage(&self, tenant_id: &str, bytes: f64) {
        self.tenant_storage_bytes
            .with_label_values(&[tenant_id])
            .set(bytes);
    }

    // System helpers

    /// Update daemon uptime
    pub fn set_uptime(&self, seconds: f64) {
        self.uptime_seconds.with_label_values(&[]).set(seconds);
    }

    /// Record an ingestion error
    pub fn ingestion_error(&self, error_type: &str) {
        self.ingestion_errors_total
            .with_label_values(&[error_type])
            .inc();
    }

    /// Record heartbeat latency
    pub fn heartbeat_processed(&self, project_id: &str, latency_secs: f64) {
        self.heartbeat_latency_seconds
            .with_label_values(&[project_id])
            .observe(latency_secs);
    }
}

impl Default for DaemonMetrics {
    fn default() -> Self {
        Self::new()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = DaemonMetrics::new();
        // Should be able to create metrics without panic
        assert!(metrics.encode().is_ok());
    }

    #[test]
    fn test_session_metrics() {
        let metrics = DaemonMetrics::new();

        // Start a session
        metrics.session_started("project-1", "high");

        // Verify active session count
        let value = metrics
            .active_sessions
            .with_label_values(&["project-1", "high"])
            .get();
        assert_eq!(value, 1);

        // End the session
        metrics.session_ended("project-1", "high", 60.0);
        let value = metrics
            .active_sessions
            .with_label_values(&["project-1", "high"])
            .get();
        assert_eq!(value, 0);
    }

    #[test]
    fn test_queue_metrics() {
        let metrics = DaemonMetrics::new();

        // Set queue depth
        metrics.set_queue_depth("high", "_projects", 100);
        let value = metrics
            .queue_depth
            .with_label_values(&["high", "_projects"])
            .get();
        assert_eq!(value, 100);

        // Process an item
        metrics.queue_item_processed("high", "success", 0.5);
        let value = metrics
            .queue_items_processed_total
            .with_label_values(&["high", "success"])
            .get();
        assert_eq!(value, 1);
    }

    #[test]
    fn test_tenant_metrics() {
        let metrics = DaemonMetrics::new();

        // Set tenant documents
        metrics.set_tenant_documents("tenant-123", "_projects", 500);
        let value = metrics
            .tenant_documents_total
            .with_label_values(&["tenant-123", "_projects"])
            .get();
        assert_eq!(value, 500);

        // Record a search
        metrics.tenant_search("tenant-123");
        let value = metrics
            .tenant_search_requests_total
            .with_label_values(&["tenant-123"])
            .get();
        assert_eq!(value, 1);
    }

    #[test]
    fn test_encode_prometheus_format() {
        let metrics = DaemonMetrics::new();

        // Add some metrics
        metrics.session_started("test-project", "normal");
        metrics.set_queue_depth("normal", "_projects", 50);

        // Encode
        let output = metrics.encode().expect("encoding should succeed");

        // Verify output contains expected metrics
        assert!(output.contains("memexd_active_sessions"));
        assert!(output.contains("memexd_queue_depth"));
    }

    #[test]
    fn test_metrics_snapshot() {
        // Use global METRICS for snapshot test
        METRICS.session_started("snapshot-test", "low");
        METRICS.set_queue_depth("low", "_memory", 25);

        let snapshot = MetricsSnapshot::capture();
        assert!(snapshot.active_sessions >= 1);
    }
}
