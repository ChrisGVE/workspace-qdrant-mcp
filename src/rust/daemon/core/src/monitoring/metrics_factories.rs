//! Factory functions for constructing the per-subsystem Prometheus metric
//! collectors used by [`DaemonMetrics`](super::metrics_core::DaemonMetrics).
//!
//! The factories are split out of `metrics_core.rs` to keep that file under
//! the project's 500-line ceiling and to give each subsystem an obvious
//! co-located definition. Each `create_*_metrics()` returns a tuple of
//! `prometheus` collectors which the constructor in `metrics_core` registers
//! and stores on the struct.

use prometheus::{
    self, core::Collector, GaugeVec, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
};

// ── Small typed builders to reduce verbosity in factories ────────────────

pub(super) fn int_gauge_vec(name: &str, help: &str, labels: &[&str]) -> IntGaugeVec {
    IntGaugeVec::new(Opts::new(name, help), labels)
        .expect("metric can be created")
}

pub(super) fn int_counter_vec(name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
    IntCounterVec::new(Opts::new(name, help), labels)
        .expect("metric can be created")
}

pub(super) fn gauge_vec(name: &str, help: &str, labels: &[&str]) -> GaugeVec {
    GaugeVec::new(Opts::new(name, help), labels).expect("metric can be created")
}

pub(super) fn histogram_vec(
    name: &str,
    help: &str,
    labels: &[&str],
    buckets: Vec<f64>,
) -> HistogramVec {
    HistogramVec::new(
        prometheus::HistogramOpts::new(name, help)
            .buckets(buckets),
        labels,
    )
    .expect("metric can be created")
}

/// Register a batch of collectors against the shared registry.
pub(super) fn register_all(registry: &Registry, collectors: Vec<Box<dyn Collector>>) {
    for c in collectors {
        registry.register(c).expect("metric can be registered");
    }
}

// ── Per-subsystem metric factories ───────────────────────────────────────

pub(super) fn create_session_metrics() -> (IntGaugeVec, IntCounterVec, HistogramVec) {
    let active_sessions = int_gauge_vec(
        "wqm_memexd_active_sessions",
        "Number of active sessions by project and priority",
        &["project_id", "priority"],
    );
    let total_sessions = int_counter_vec(
        "wqm_memexd_total_sessions",
        "Total number of sessions created (lifetime)",
        &["project_id"],
    );
    let session_duration_seconds = histogram_vec(
        "wqm_memexd_session_duration_seconds",
        "Session duration in seconds",
        &["project_id"],
        vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
    );
    (active_sessions, total_sessions, session_duration_seconds)
}

pub(super) fn create_queue_metrics() -> (IntGaugeVec, HistogramVec, IntCounterVec) {
    let queue_depth = int_gauge_vec(
        "wqm_memexd_queue_depth",
        "Current queue depth by priority and collection",
        &["priority", "collection"],
    );
    let queue_processing_time_seconds = histogram_vec(
        "wqm_memexd_queue_processing_time_seconds",
        "Queue item processing time in seconds",
        &["priority"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
    );
    let queue_items_processed_total = int_counter_vec(
        "wqm_memexd_queue_items_processed_total",
        "Total items processed by priority and status",
        &["priority", "status"],
    );
    (
        queue_depth,
        queue_processing_time_seconds,
        queue_items_processed_total,
    )
}

pub(super) fn create_tenant_metrics() -> (IntGaugeVec, IntCounterVec, GaugeVec) {
    let tenant_documents_total = int_gauge_vec(
        "wqm_memexd_tenant_documents_total",
        "Total documents per tenant and collection",
        &["tenant_id", "collection"],
    );
    let tenant_search_requests_total = int_counter_vec(
        "wqm_memexd_tenant_search_requests_total",
        "Total search requests per tenant",
        &["tenant_id"],
    );
    let tenant_storage_bytes = gauge_vec(
        "wqm_memexd_tenant_storage_bytes",
        "Estimated storage bytes per tenant",
        &["tenant_id"],
    );
    (
        tenant_documents_total,
        tenant_search_requests_total,
        tenant_storage_bytes,
    )
}

pub(super) fn create_system_metrics() -> (GaugeVec, IntCounterVec, HistogramVec) {
    let uptime_seconds = gauge_vec("wqm_memexd_uptime_seconds", "Daemon uptime in seconds", &[]);
    let ingestion_errors_total = int_counter_vec(
        "wqm_memexd_ingestion_errors_total",
        "Total ingestion errors by error type",
        &["error_type"],
    );
    let heartbeat_latency_seconds = histogram_vec(
        "wqm_memexd_heartbeat_latency_seconds",
        "Heartbeat processing latency in seconds",
        &["project_id"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    );
    (
        uptime_seconds,
        ingestion_errors_total,
        heartbeat_latency_seconds,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn create_watch_metrics() -> (
    IntCounterVec,
    IntGaugeVec,
    IntGaugeVec,
    IntGaugeVec,
    HistogramVec,
    IntCounterVec,
) {
    let watch_errors_total = int_counter_vec(
        "wqm_memexd_watch_errors_total",
        "Total watch errors by watch_id",
        &["watch_id"],
    );
    let watch_consecutive_errors = int_gauge_vec(
        "wqm_memexd_watch_consecutive_errors",
        "Current consecutive errors by watch_id",
        &["watch_id"],
    );
    let watch_health_status = int_gauge_vec(
        "wqm_memexd_watch_health_status",
        "Watch health status (1 = in this state)",
        &["watch_id", "health_status"],
    );
    let watches_in_backoff = int_gauge_vec(
        "wqm_memexd_watches_in_backoff",
        "Number of watches currently in backoff",
        &[],
    );
    let watch_recovery_time_seconds = histogram_vec(
        "wqm_memexd_watch_recovery_time_seconds",
        "Watch error recovery time in seconds",
        &["watch_id"],
        vec![
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0,
        ],
    );
    let watch_events_throttled_total = int_counter_vec(
        "wqm_memexd_watch_events_throttled_total",
        "Events throttled due to queue depth",
        &["watch_id", "load_level"],
    );
    (
        watch_errors_total,
        watch_consecutive_errors,
        watch_health_status,
        watches_in_backoff,
        watch_recovery_time_seconds,
        watch_events_throttled_total,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn create_dependency_metrics() -> (
    HistogramVec,
    HistogramVec,
    HistogramVec,
    HistogramVec,
    IntCounterVec,
) {
    let embedding_duration_seconds = histogram_vec(
        "wqm_memexd_embedding_duration_seconds",
        "Embedding generation duration in seconds by model",
        &["model"],
        vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    );
    let embedding_batch_size = histogram_vec(
        "wqm_memexd_embedding_batch_size",
        "Embedding batch size (items per call) by model",
        &["model"],
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0],
    );
    let sqlite_query_duration_seconds = histogram_vec(
        "wqm_memexd_sqlite_query_duration_seconds",
        "SQLite query duration in seconds by op",
        &["op"],
        vec![0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    );
    let qdrant_request_duration_seconds = histogram_vec(
        "wqm_memexd_qdrant_request_duration_seconds",
        "Qdrant request duration in seconds by op",
        &["op"],
        vec![0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    );
    let qdrant_request_errors_total = int_counter_vec(
        "wqm_memexd_qdrant_request_errors_total",
        "Total Qdrant request errors by op and error_type",
        &["op", "error_type"],
    );
    (
        embedding_duration_seconds,
        embedding_batch_size,
        sqlite_query_duration_seconds,
        qdrant_request_duration_seconds,
        qdrant_request_errors_total,
    )
}

pub(super) fn create_telemetry_extension_metrics(
) -> (IntCounterVec, IntCounterVec, IntCounterVec, HistogramVec) {
    let watcher_events_total = int_counter_vec(
        "wqm_memexd_watcher_events_total",
        "Total filesystem watcher events by event_type",
        &["event_type"],
    );
    let watcher_coalesced_total = int_counter_vec(
        "wqm_memexd_watcher_coalesced_total",
        "Total watcher events coalesced before enqueue",
        &["reason"],
    );
    let grpc_requests_total = int_counter_vec(
        "wqm_memexd_grpc_requests_total",
        "Total gRPC requests by service, method, and status",
        &["service", "method", "status"],
    );
    let grpc_request_duration_seconds = histogram_vec(
        "wqm_memexd_grpc_request_duration_seconds",
        "gRPC request duration in seconds by service and method",
        &["service", "method"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    );
    (
        watcher_events_total,
        watcher_coalesced_total,
        grpc_requests_total,
        grpc_request_duration_seconds,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn create_unified_queue_metrics() -> (
    IntGaugeVec,
    HistogramVec,
    IntCounterVec,
    IntCounterVec,
    IntCounterVec,
    IntGaugeVec,
    IntCounterVec,
) {
    let unified_queue_depth = int_gauge_vec(
        "wqm_memexd_unified_queue_depth",
        "Current unified queue depth by item_type and status",
        &["item_type", "status"],
    );
    let unified_queue_processing_time_seconds = histogram_vec(
        "wqm_memexd_unified_queue_processing_time_seconds",
        "Unified queue item processing time in seconds",
        &["item_type"],
        vec![
            0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0,
        ],
    );
    let unified_queue_items_total = int_counter_vec(
        "wqm_memexd_unified_queue_items_total",
        "Total unified queue items processed by item_type, op, and result",
        &["item_type", "op", "result"],
    );
    let unified_queue_enqueues_total = int_counter_vec(
        "wqm_memexd_unified_queue_enqueues_total",
        "Total unified queue enqueues by source",
        &["source"],
    );
    let unified_queue_dequeues_total = int_counter_vec(
        "wqm_memexd_unified_queue_dequeues_total",
        "Total unified queue dequeues by item_type",
        &["item_type"],
    );
    let unified_queue_stale_items = int_gauge_vec(
        "wqm_memexd_unified_queue_stale_items",
        "Stale lease items in unified queue",
        &[],
    );
    let unified_queue_retries_total = int_counter_vec(
        "wqm_memexd_unified_queue_retries_total",
        "Unified queue retry count by item_type",
        &["item_type"],
    );
    (
        unified_queue_depth,
        unified_queue_processing_time_seconds,
        unified_queue_items_total,
        unified_queue_enqueues_total,
        unified_queue_dequeues_total,
        unified_queue_stale_items,
        unified_queue_retries_total,
    )
}
