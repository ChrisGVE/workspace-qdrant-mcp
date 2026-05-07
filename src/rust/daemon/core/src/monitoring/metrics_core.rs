//! Prometheus metrics core: [`DaemonMetrics`] struct and the [`METRICS`]
//! global lazy-static registry.
//!
//! Per-subsystem collector factories live in [`super::metrics_factories`];
//! convenience tracking helpers live in [`super::metrics_helpers`]; and unit
//! tests for the per-record helpers live in `metrics_core_tests`.

use once_cell::sync::Lazy;
use prometheus::{
    self, Encoder, GaugeVec, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Registry,
    TextEncoder,
};

use super::metrics_factories::{
    create_dependency_metrics, create_queue_metrics, create_session_metrics, create_system_metrics,
    create_telemetry_extension_metrics, create_tenant_metrics, create_unified_queue_metrics,
    create_watch_metrics, register_all,
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

    /// Age in seconds of the oldest pending queue item (0 if none)
    pub queue_oldest_pending_age_seconds: IntGauge,

    // Telemetry extension metrics (issue-64 Task 2)
    /// Total filesystem watcher events by event_type
    /// Labels: event_type (create, modify, delete, rename)
    pub watcher_events_total: IntCounterVec,

    /// Total watcher events coalesced before enqueue
    /// Labels: reason (debounce, duplicate)
    pub watcher_coalesced_total: IntCounterVec,

    /// Total gRPC requests by service, method, and status
    /// Labels: service, method, status (ok, error)
    pub grpc_requests_total: IntCounterVec,

    /// gRPC request duration in seconds by service and method
    /// Labels: service, method
    pub grpc_request_duration_seconds: HistogramVec,

    // Dependency latency metrics (issue-64 Task 7)
    /// Embedding generation duration in seconds by model
    /// Labels: model
    pub embedding_duration_seconds: HistogramVec,

    /// Embedding batch size (items per call) by model
    /// Labels: model
    pub embedding_batch_size: HistogramVec,

    /// SQLite query duration in seconds by op
    /// Labels: op (read, write, transaction)
    pub sqlite_query_duration_seconds: HistogramVec,

    /// Qdrant request duration in seconds by op
    /// Labels: op (upsert, search, delete, collection_create, collection_info)
    pub qdrant_request_duration_seconds: HistogramVec,

    /// Total Qdrant request errors by op and error_type
    /// Labels: op, error_type
    pub qdrant_request_errors_total: IntCounterVec,
}

impl DaemonMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let registry = Registry::new();

        let (active_sessions, total_sessions, session_duration_seconds) = create_session_metrics();
        let (queue_depth, queue_processing_time_seconds, queue_items_processed_total) =
            create_queue_metrics();
        let (tenant_documents_total, tenant_search_requests_total, tenant_storage_bytes) =
            create_tenant_metrics();
        let (uptime_seconds, ingestion_errors_total, heartbeat_latency_seconds) =
            create_system_metrics();
        let (
            watch_errors_total,
            watch_consecutive_errors,
            watch_health_status,
            watches_in_backoff,
            watch_recovery_time_seconds,
            watch_events_throttled_total,
        ) = create_watch_metrics();
        let (
            unified_queue_depth,
            unified_queue_processing_time_seconds,
            unified_queue_items_total,
            unified_queue_enqueues_total,
            unified_queue_dequeues_total,
            unified_queue_stale_items,
            unified_queue_retries_total,
        ) = create_unified_queue_metrics();
        let (
            watcher_events_total,
            watcher_coalesced_total,
            grpc_requests_total,
            grpc_request_duration_seconds,
        ) = create_telemetry_extension_metrics();
        let (
            embedding_duration_seconds,
            embedding_batch_size,
            sqlite_query_duration_seconds,
            qdrant_request_duration_seconds,
            qdrant_request_errors_total,
        ) = create_dependency_metrics();

        // Scalar gauge: oldest pending queue item age in seconds
        let queue_oldest_pending_age_seconds = IntGauge::new(
            "wqm_queue_oldest_pending_age_seconds",
            "Age in seconds of the oldest pending queue item",
        )
        .expect("metric can be created");

        register_all(
            &registry,
            vec![
                Box::new(active_sessions.clone()),
                Box::new(total_sessions.clone()),
                Box::new(session_duration_seconds.clone()),
                Box::new(queue_depth.clone()),
                Box::new(queue_processing_time_seconds.clone()),
                Box::new(queue_items_processed_total.clone()),
                Box::new(tenant_documents_total.clone()),
                Box::new(tenant_search_requests_total.clone()),
                Box::new(tenant_storage_bytes.clone()),
                Box::new(uptime_seconds.clone()),
                Box::new(ingestion_errors_total.clone()),
                Box::new(heartbeat_latency_seconds.clone()),
                Box::new(watch_errors_total.clone()),
                Box::new(watch_consecutive_errors.clone()),
                Box::new(watch_health_status.clone()),
                Box::new(watches_in_backoff.clone()),
                Box::new(watch_recovery_time_seconds.clone()),
                Box::new(watch_events_throttled_total.clone()),
                Box::new(unified_queue_depth.clone()),
                Box::new(unified_queue_processing_time_seconds.clone()),
                Box::new(unified_queue_items_total.clone()),
                Box::new(unified_queue_enqueues_total.clone()),
                Box::new(unified_queue_dequeues_total.clone()),
                Box::new(unified_queue_stale_items.clone()),
                Box::new(unified_queue_retries_total.clone()),
                Box::new(queue_oldest_pending_age_seconds.clone()),
                Box::new(watcher_events_total.clone()),
                Box::new(watcher_coalesced_total.clone()),
                Box::new(grpc_requests_total.clone()),
                Box::new(grpc_request_duration_seconds.clone()),
                Box::new(embedding_duration_seconds.clone()),
                Box::new(embedding_batch_size.clone()),
                Box::new(sqlite_query_duration_seconds.clone()),
                Box::new(qdrant_request_duration_seconds.clone()),
                Box::new(qdrant_request_errors_total.clone()),
            ],
        );

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
            unified_queue_depth,
            unified_queue_processing_time_seconds,
            unified_queue_items_total,
            unified_queue_enqueues_total,
            unified_queue_dequeues_total,
            unified_queue_stale_items,
            unified_queue_retries_total,
            queue_oldest_pending_age_seconds,
            watcher_events_total,
            watcher_coalesced_total,
            grpc_requests_total,
            grpc_request_duration_seconds,
            embedding_duration_seconds,
            embedding_batch_size,
            sqlite_query_duration_seconds,
            qdrant_request_duration_seconds,
            qdrant_request_errors_total,
        }
    }

    /// Record an embedding batch: duration and batch size by model.
    pub fn record_embedding(&self, model: &str, batch_size: usize, duration: std::time::Duration) {
        self.embedding_duration_seconds
            .with_label_values(&[model])
            .observe(duration.as_secs_f64());
        self.embedding_batch_size
            .with_label_values(&[model])
            .observe(batch_size as f64);
    }

    /// Record a SQLite query by op (read, write, transaction).
    pub fn record_sqlite(&self, op: &str, duration: std::time::Duration) {
        self.sqlite_query_duration_seconds
            .with_label_values(&[op])
            .observe(duration.as_secs_f64());
    }

    /// Record a Qdrant request with op, duration, and optional error type.
    pub fn record_qdrant(&self, op: &str, duration: std::time::Duration, error: Option<&str>) {
        self.qdrant_request_duration_seconds
            .with_label_values(&[op])
            .observe(duration.as_secs_f64());
        if let Some(error_type) = error {
            self.qdrant_request_errors_total
                .with_label_values(&[op, error_type])
                .inc();
        }
    }

    /// Record a single filesystem watcher event.
    pub fn record_watcher_event(&self, event_type: &str) {
        self.watcher_events_total
            .with_label_values(&[event_type])
            .inc();
    }

    /// Record a coalesced watcher event (debounced or duplicate).
    pub fn record_watcher_coalesced(&self, reason: &str) {
        self.watcher_coalesced_total
            .with_label_values(&[reason])
            .inc();
    }

    /// Record a completed gRPC call with its duration and outcome.
    pub fn record_grpc_call(
        &self,
        service: &str,
        method: &str,
        ok: bool,
        duration: std::time::Duration,
    ) {
        let status = if ok { "ok" } else { "error" };
        self.grpc_requests_total
            .with_label_values(&[service, method, status])
            .inc();
        self.grpc_request_duration_seconds
            .with_label_values(&[service, method])
            .observe(duration.as_secs_f64());
    }

    /// Encode all metrics in Prometheus text format
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer).unwrap_or_default())
    }
}

impl Default for DaemonMetrics {
    fn default() -> Self {
        Self::new()
    }
}
