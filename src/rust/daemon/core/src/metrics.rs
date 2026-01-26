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

    // Watch error metrics (Task 461.12)
    /// Total watch errors by watch_id
    /// Labels: watch_id
    pub watch_errors_total: IntCounterVec,

    /// Current consecutive errors by watch_id
    /// Labels: watch_id
    pub watch_consecutive_errors: IntGaugeVec,

    /// Watch health status (1 = in this state, 0 = not)
    /// Labels: watch_id, health_status (healthy, degraded, backoff, disabled)
    pub watch_health_status: IntGaugeVec,

    /// Number of watches currently in backoff
    pub watches_in_backoff: IntGaugeVec,

    /// Watch error recovery time in seconds (from first error to recovery)
    /// Labels: watch_id
    pub watch_recovery_time_seconds: HistogramVec,

    /// Events throttled due to queue depth
    /// Labels: watch_id, load_level (high, critical)
    pub watch_events_throttled_total: IntCounterVec,

    // Unified Queue metrics (Task 37.35)
    /// Current unified queue depth by item_type and status
    /// Labels: item_type, status (pending, in_progress, done, failed)
    pub unified_queue_depth: IntGaugeVec,

    /// Unified queue processing time in seconds by item_type
    /// Labels: item_type
    pub unified_queue_processing_time_seconds: HistogramVec,

    /// Total unified queue items by item_type, op, and result
    /// Labels: item_type, op, result (success, failure, skipped)
    pub unified_queue_items_total: IntCounterVec,

    /// Total unified queue enqueues by source
    /// Labels: source (mcp_store, mcp_manage, cli_ingest, cli_memory, daemon)
    pub unified_queue_enqueues_total: IntCounterVec,

    /// Total unified queue dequeues by item_type
    /// Labels: item_type
    pub unified_queue_dequeues_total: IntCounterVec,

    /// Stale lease items in unified queue (expired but not recovered)
    pub unified_queue_stale_items: IntGaugeVec,

    /// Unified queue retry count by item_type
    /// Labels: item_type
    pub unified_queue_retries_total: IntCounterVec,
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

        // Watch error metrics (Task 461.12)
        let watch_errors_total = IntCounterVec::new(
            Opts::new(
                "memexd_watch_errors_total",
                "Total watch errors by watch_id",
            )
            .namespace("memexd"),
            &["watch_id"],
        )
        .expect("metric can be created");

        let watch_consecutive_errors = IntGaugeVec::new(
            Opts::new(
                "memexd_watch_consecutive_errors",
                "Current consecutive errors by watch_id",
            )
            .namespace("memexd"),
            &["watch_id"],
        )
        .expect("metric can be created");

        let watch_health_status = IntGaugeVec::new(
            Opts::new(
                "memexd_watch_health_status",
                "Watch health status (1 = in this state)",
            )
            .namespace("memexd"),
            &["watch_id", "health_status"],
        )
        .expect("metric can be created");

        let watches_in_backoff = IntGaugeVec::new(
            Opts::new(
                "memexd_watches_in_backoff",
                "Number of watches currently in backoff",
            )
            .namespace("memexd"),
            &[],
        )
        .expect("metric can be created");

        let watch_recovery_time_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "memexd_watch_recovery_time_seconds",
                "Watch error recovery time in seconds",
            )
            .namespace("memexd")
            .buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]),
            &["watch_id"],
        )
        .expect("metric can be created");

        let watch_events_throttled_total = IntCounterVec::new(
            Opts::new(
                "memexd_watch_events_throttled_total",
                "Events throttled due to queue depth",
            )
            .namespace("memexd"),
            &["watch_id", "load_level"],
        )
        .expect("metric can be created");

        // Unified Queue metrics (Task 37.35)
        let unified_queue_depth = IntGaugeVec::new(
            Opts::new(
                "memexd_unified_queue_depth",
                "Current unified queue depth by item_type and status",
            )
            .namespace("memexd"),
            &["item_type", "status"],
        )
        .expect("metric can be created");

        let unified_queue_processing_time_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "memexd_unified_queue_processing_time_seconds",
                "Unified queue item processing time in seconds",
            )
            .namespace("memexd")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]),
            &["item_type"],
        )
        .expect("metric can be created");

        let unified_queue_items_total = IntCounterVec::new(
            Opts::new(
                "memexd_unified_queue_items_total",
                "Total unified queue items processed by item_type, op, and result",
            )
            .namespace("memexd"),
            &["item_type", "op", "result"],
        )
        .expect("metric can be created");

        let unified_queue_enqueues_total = IntCounterVec::new(
            Opts::new(
                "memexd_unified_queue_enqueues_total",
                "Total unified queue enqueues by source",
            )
            .namespace("memexd"),
            &["source"],
        )
        .expect("metric can be created");

        let unified_queue_dequeues_total = IntCounterVec::new(
            Opts::new(
                "memexd_unified_queue_dequeues_total",
                "Total unified queue dequeues by item_type",
            )
            .namespace("memexd"),
            &["item_type"],
        )
        .expect("metric can be created");

        let unified_queue_stale_items = IntGaugeVec::new(
            Opts::new(
                "memexd_unified_queue_stale_items",
                "Stale lease items in unified queue",
            )
            .namespace("memexd"),
            &[],
        )
        .expect("metric can be created");

        let unified_queue_retries_total = IntCounterVec::new(
            Opts::new(
                "memexd_unified_queue_retries_total",
                "Unified queue retry count by item_type",
            )
            .namespace("memexd"),
            &["item_type"],
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
        registry
            .register(Box::new(watch_errors_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(watch_consecutive_errors.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(watch_health_status.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(watches_in_backoff.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(watch_recovery_time_seconds.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(watch_events_throttled_total.clone()))
            .expect("metric can be registered");
        // Unified queue metrics registration (Task 37.35)
        registry
            .register(Box::new(unified_queue_depth.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(unified_queue_processing_time_seconds.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(unified_queue_items_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(unified_queue_enqueues_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(unified_queue_dequeues_total.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(unified_queue_stale_items.clone()))
            .expect("metric can be registered");
        registry
            .register(Box::new(unified_queue_retries_total.clone()))
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
            watch_errors_total,
            watch_consecutive_errors,
            watch_health_status,
            watches_in_backoff,
            watch_recovery_time_seconds,
            watch_events_throttled_total,
            // Unified queue metrics
            unified_queue_depth,
            unified_queue_processing_time_seconds,
            unified_queue_items_total,
            unified_queue_enqueues_total,
            unified_queue_dequeues_total,
            unified_queue_stale_items,
            unified_queue_retries_total,
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

    // Watch error tracking helpers (Task 461.12)

    /// Record a watch error
    pub fn watch_error(&self, watch_id: &str) {
        self.watch_errors_total
            .with_label_values(&[watch_id])
            .inc();
    }

    /// Set the current consecutive error count for a watch
    pub fn set_watch_consecutive_errors(&self, watch_id: &str, count: i64) {
        self.watch_consecutive_errors
            .with_label_values(&[watch_id])
            .set(count);
    }

    /// Update watch health status
    ///
    /// Sets the new status to 1 and clears the old status to 0.
    /// Valid statuses: "healthy", "degraded", "backoff", "disabled"
    pub fn set_watch_health_status(&self, watch_id: &str, old_status: Option<&str>, new_status: &str) {
        // Clear the old status if provided
        if let Some(old) = old_status {
            self.watch_health_status
                .with_label_values(&[watch_id, old])
                .set(0);
        }
        // Set the new status
        self.watch_health_status
            .with_label_values(&[watch_id, new_status])
            .set(1);
    }

    /// Set the total number of watches currently in backoff
    pub fn set_watches_in_backoff(&self, count: i64) {
        self.watches_in_backoff
            .with_label_values(&[])
            .set(count);
    }

    /// Increment watches in backoff count
    pub fn inc_watches_in_backoff(&self) {
        self.watches_in_backoff
            .with_label_values(&[])
            .inc();
    }

    /// Decrement watches in backoff count
    pub fn dec_watches_in_backoff(&self) {
        self.watches_in_backoff
            .with_label_values(&[])
            .dec();
    }

    /// Record watch recovery time (from first error to recovery)
    pub fn watch_recovered(&self, watch_id: &str, recovery_time_secs: f64) {
        self.watch_recovery_time_seconds
            .with_label_values(&[watch_id])
            .observe(recovery_time_secs);
        // Clear consecutive errors when recovered
        self.set_watch_consecutive_errors(watch_id, 0);
    }

    /// Record a throttled event due to queue depth
    pub fn watch_event_throttled(&self, watch_id: &str, load_level: &str) {
        self.watch_events_throttled_total
            .with_label_values(&[watch_id, load_level])
            .inc();
    }

    // Unified Queue helpers (Task 37.35)

    /// Set unified queue depth for item_type and status
    pub fn set_unified_queue_depth(&self, item_type: &str, status: &str, depth: i64) {
        self.unified_queue_depth
            .with_label_values(&[item_type, status])
            .set(depth);
    }

    /// Record unified queue item processed
    pub fn unified_queue_item_processed(
        &self,
        item_type: &str,
        op: &str,
        result: &str,
        processing_time_secs: f64,
    ) {
        self.unified_queue_items_total
            .with_label_values(&[item_type, op, result])
            .inc();
        self.unified_queue_processing_time_seconds
            .with_label_values(&[item_type])
            .observe(processing_time_secs);
    }

    /// Record unified queue enqueue
    pub fn unified_queue_enqueued(&self, source: &str) {
        self.unified_queue_enqueues_total
            .with_label_values(&[source])
            .inc();
    }

    /// Record unified queue dequeue
    pub fn unified_queue_dequeued(&self, item_type: &str) {
        self.unified_queue_dequeues_total
            .with_label_values(&[item_type])
            .inc();
    }

    /// Set count of stale lease items
    pub fn set_unified_queue_stale_items(&self, count: i64) {
        self.unified_queue_stale_items
            .with_label_values(&[])
            .set(count);
    }

    /// Record a retry for unified queue item
    pub fn unified_queue_retry(&self, item_type: &str) {
        self.unified_queue_retries_total
            .with_label_values(&[item_type])
            .inc();
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

// =============================================================================
// Alerting Logic (Task 412.15-18)
// =============================================================================

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

/// Alert type enumeration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AlertType {
    /// Queue depth exceeds threshold (Task 412.15)
    HighQueueDepth { depth: i64, threshold: i64 },
    /// Orphaned session detected (Task 412.16)
    OrphanedSession { project_id: String, last_heartbeat_secs: f64 },
    /// Error rate exceeds threshold (Task 412.17)
    HighErrorRate { error_rate_percent: f64, threshold_percent: f64 },
    /// Search latency exceeds threshold (Task 412.18)
    SlowSearches { p95_latency_ms: f64, threshold_ms: f64 },
}

/// Alert structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Alert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Alert configuration thresholds
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlertConfig {
    /// Queue depth threshold for alerts (default: 1000)
    pub queue_depth_threshold: i64,
    /// Orphaned session timeout in seconds (default: 600 = 10 minutes)
    pub orphaned_session_timeout_secs: f64,
    /// Error rate threshold percentage (default: 5.0)
    pub error_rate_threshold_percent: f64,
    /// Slow search threshold in milliseconds (default: 500)
    pub slow_search_threshold_ms: f64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            queue_depth_threshold: 1000,
            orphaned_session_timeout_secs: 600.0,  // 10 minutes
            error_rate_threshold_percent: 5.0,
            slow_search_threshold_ms: 500.0,
        }
    }
}

/// Alert checker for monitoring system health
pub struct AlertChecker {
    config: AlertConfig,
}

impl AlertChecker {
    /// Create a new alert checker with default configuration
    pub fn new() -> Self {
        Self {
            config: AlertConfig::default(),
        }
    }

    /// Create a new alert checker with custom configuration
    pub fn with_config(config: AlertConfig) -> Self {
        Self { config }
    }

    /// Check for high queue depth alert (Task 412.15)
    pub fn check_queue_depth(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let total_depth: i64 = snapshot.queue_depths.values().sum();

        if total_depth > self.config.queue_depth_threshold {
            let severity = if total_depth > self.config.queue_depth_threshold * 2 {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            };

            Some(Alert {
                alert_type: AlertType::HighQueueDepth {
                    depth: total_depth,
                    threshold: self.config.queue_depth_threshold,
                },
                severity,
                message: format!(
                    "Queue depth ({}) exceeds threshold ({})",
                    total_depth, self.config.queue_depth_threshold
                ),
                timestamp: chrono::Utc::now(),
            })
        } else {
            None
        }
    }

    /// Check for high error rate alert (Task 412.17)
    pub fn check_error_rate(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let total_errors: u64 = snapshot.error_counts.values().sum();
        let total_processed = snapshot.total_items_processed;

        if total_processed == 0 {
            return None;
        }

        let error_rate = (total_errors as f64 / total_processed as f64) * 100.0;

        if error_rate > self.config.error_rate_threshold_percent {
            let severity = if error_rate > self.config.error_rate_threshold_percent * 2.0 {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            };

            Some(Alert {
                alert_type: AlertType::HighErrorRate {
                    error_rate_percent: error_rate,
                    threshold_percent: self.config.error_rate_threshold_percent,
                },
                severity,
                message: format!(
                    "Error rate ({:.2}%) exceeds threshold ({:.1}%)",
                    error_rate, self.config.error_rate_threshold_percent
                ),
                timestamp: chrono::Utc::now(),
            })
        } else {
            None
        }
    }

    /// Check all alerts and return active alerts
    pub fn check_all(&self, snapshot: &MetricsSnapshot) -> Vec<Alert> {
        let mut alerts = Vec::new();

        if let Some(alert) = self.check_queue_depth(snapshot) {
            alerts.push(alert);
        }

        if let Some(alert) = self.check_error_rate(snapshot) {
            alerts.push(alert);
        }

        // Note: Orphaned session and slow search alerts require additional
        // data sources (session heartbeat timestamps, search latency histograms)
        // that are not available in MetricsSnapshot alone.
        // These should be checked by the session monitor and search handler.

        alerts
    }
}

impl Default for AlertChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Check for orphaned sessions (Task 412.16)
///
/// This function should be called by the session monitor which has access
/// to project heartbeat timestamps.
pub fn create_orphaned_session_alert(
    project_id: &str,
    last_heartbeat_secs: f64,
    timeout_threshold_secs: f64,
) -> Option<Alert> {
    if last_heartbeat_secs > timeout_threshold_secs {
        let severity = if last_heartbeat_secs > timeout_threshold_secs * 2.0 {
            AlertSeverity::Critical
        } else {
            AlertSeverity::Warning
        };

        Some(Alert {
            alert_type: AlertType::OrphanedSession {
                project_id: project_id.to_string(),
                last_heartbeat_secs,
            },
            severity,
            message: format!(
                "Orphaned session for project '{}': no heartbeat for {:.0}s (threshold: {:.0}s)",
                project_id, last_heartbeat_secs, timeout_threshold_secs
            ),
            timestamp: chrono::Utc::now(),
        })
    } else {
        None
    }
}

/// Check for slow searches (Task 412.18)
///
/// This function should be called by the search handler with p95 latency data.
pub fn create_slow_search_alert(
    p95_latency_ms: f64,
    threshold_ms: f64,
) -> Option<Alert> {
    if p95_latency_ms > threshold_ms {
        let severity = if p95_latency_ms > threshold_ms * 2.0 {
            AlertSeverity::Critical
        } else {
            AlertSeverity::Warning
        };

        Some(Alert {
            alert_type: AlertType::SlowSearches {
                p95_latency_ms,
                threshold_ms,
            },
            severity,
            message: format!(
                "Search p95 latency ({:.0}ms) exceeds threshold ({:.0}ms)",
                p95_latency_ms, threshold_ms
            ),
            timestamp: chrono::Utc::now(),
        })
    } else {
        None
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

        // Set queue depth (using canonical collection name)
        metrics.set_queue_depth("high", "projects", 100);
        let value = metrics
            .queue_depth
            .with_label_values(&["high", "projects"])
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

        // Set tenant documents (using canonical collection name)
        metrics.set_tenant_documents("tenant-123", "projects", 500);
        let value = metrics
            .tenant_documents_total
            .with_label_values(&["tenant-123", "projects"])
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

        // Add some metrics (using canonical collection name)
        metrics.session_started("test-project", "normal");
        metrics.set_queue_depth("normal", "projects", 50);

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
        METRICS.set_queue_depth("low", "memory", 25);

        let snapshot = MetricsSnapshot::capture();
        assert!(snapshot.active_sessions >= 1);
    }

    // Alerting tests (Task 412.15-18)

    #[test]
    fn test_high_queue_depth_alert() {
        let checker = AlertChecker::new();

        // Create snapshot with high queue depth
        let mut snapshot = MetricsSnapshot {
            uptime_seconds: 100.0,
            active_sessions: 5,
            total_sessions_lifetime: 10,
            queue_depths: std::collections::HashMap::new(),
            total_items_processed: 100,
            error_counts: std::collections::HashMap::new(),
            tenant_documents: std::collections::HashMap::new(),
        };

        // Below threshold - no alert
        snapshot.queue_depths.insert("high".to_string(), 500);
        assert!(checker.check_queue_depth(&snapshot).is_none());

        // Above threshold - warning
        snapshot.queue_depths.insert("high".to_string(), 1500);
        let alert = checker.check_queue_depth(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        // Way above threshold - critical
        snapshot.queue_depths.insert("high".to_string(), 3000);
        let alert = checker.check_queue_depth(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_high_error_rate_alert() {
        let checker = AlertChecker::new();

        let mut snapshot = MetricsSnapshot {
            uptime_seconds: 100.0,
            active_sessions: 5,
            total_sessions_lifetime: 10,
            queue_depths: std::collections::HashMap::new(),
            total_items_processed: 100,
            error_counts: std::collections::HashMap::new(),
            tenant_documents: std::collections::HashMap::new(),
        };

        // Below threshold - no alert
        snapshot.error_counts.insert("parse_error".to_string(), 2);
        assert!(checker.check_error_rate(&snapshot).is_none());

        // Above threshold - warning
        snapshot.error_counts.insert("parse_error".to_string(), 10);
        let alert = checker.check_error_rate(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        // Way above threshold - critical
        snapshot.error_counts.insert("parse_error".to_string(), 20);
        let alert = checker.check_error_rate(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_orphaned_session_alert() {
        // Below threshold - no alert
        assert!(create_orphaned_session_alert("project-1", 300.0, 600.0).is_none());

        // Above threshold - warning
        let alert = create_orphaned_session_alert("project-1", 700.0, 600.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        // Way above threshold - critical
        let alert = create_orphaned_session_alert("project-1", 1500.0, 600.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_slow_search_alert() {
        // Below threshold - no alert
        assert!(create_slow_search_alert(300.0, 500.0).is_none());

        // Above threshold - warning
        let alert = create_slow_search_alert(600.0, 500.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        // Way above threshold - critical
        let alert = create_slow_search_alert(1200.0, 500.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_checker_all() {
        let checker = AlertChecker::new();

        let snapshot = MetricsSnapshot {
            uptime_seconds: 100.0,
            active_sessions: 5,
            total_sessions_lifetime: 10,
            queue_depths: vec![("high".to_string(), 1500)].into_iter().collect(),
            total_items_processed: 100,
            error_counts: vec![("error".to_string(), 10)].into_iter().collect(),
            tenant_documents: std::collections::HashMap::new(),
        };

        let alerts = checker.check_all(&snapshot);
        assert_eq!(alerts.len(), 2);  // Queue depth and error rate alerts
    }

    // Unified Queue metrics tests (Task 37.35)

    #[test]
    fn test_unified_queue_metrics() {
        let metrics = DaemonMetrics::new();

        // Test queue depth
        metrics.set_unified_queue_depth("content", "pending", 50);
        let value = metrics
            .unified_queue_depth
            .with_label_values(&["content", "pending"])
            .get();
        assert_eq!(value, 50);

        // Test item processed
        metrics.unified_queue_item_processed("content", "ingest", "success", 0.5);
        let value = metrics
            .unified_queue_items_total
            .with_label_values(&["content", "ingest", "success"])
            .get();
        assert_eq!(value, 1);

        // Test enqueue
        metrics.unified_queue_enqueued("mcp_store");
        let value = metrics
            .unified_queue_enqueues_total
            .with_label_values(&["mcp_store"])
            .get();
        assert_eq!(value, 1);

        // Test dequeue
        metrics.unified_queue_dequeued("file");
        let value = metrics
            .unified_queue_dequeues_total
            .with_label_values(&["file"])
            .get();
        assert_eq!(value, 1);

        // Test stale items
        metrics.set_unified_queue_stale_items(3);
        let value = metrics
            .unified_queue_stale_items
            .with_label_values(&[])
            .get();
        assert_eq!(value, 3);

        // Test retry
        metrics.unified_queue_retry("folder");
        let value = metrics
            .unified_queue_retries_total
            .with_label_values(&["folder"])
            .get();
        assert_eq!(value, 1);
    }

    #[test]
    fn test_unified_queue_metrics_in_prometheus_output() {
        let metrics = DaemonMetrics::new();

        // Add unified queue metrics
        metrics.set_unified_queue_depth("content", "pending", 100);
        metrics.unified_queue_enqueued("cli_ingest");

        // Encode and verify
        let output = metrics.encode().expect("encoding should succeed");
        assert!(output.contains("memexd_unified_queue_depth"));
        assert!(output.contains("memexd_unified_queue_enqueues_total"));
    }
}
