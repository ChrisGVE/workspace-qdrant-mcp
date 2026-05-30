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
    IntGaugeVec::new(Opts::new(name, help).namespace("memexd"), labels)
        .expect("metric can be created")
}

pub(super) fn int_counter_vec(name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
    IntCounterVec::new(Opts::new(name, help).namespace("memexd"), labels)
        .expect("metric can be created")
}

pub(super) fn gauge_vec(name: &str, help: &str, labels: &[&str]) -> GaugeVec {
    GaugeVec::new(Opts::new(name, help).namespace("memexd"), labels).expect("metric can be created")
}

pub(super) fn histogram_vec(
    name: &str,
    help: &str,
    labels: &[&str],
    buckets: Vec<f64>,
) -> HistogramVec {
    HistogramVec::new(
        prometheus::HistogramOpts::new(name, help)
            .namespace("memexd")
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
        "active_sessions",
        "Number of active sessions by project and priority",
        &["project_id", "priority"],
    );
    let total_sessions = int_counter_vec(
        "total_sessions",
        "Total number of sessions created (lifetime)",
        &["project_id"],
    );
    let session_duration_seconds = histogram_vec(
        "session_duration_seconds",
        "Session duration in seconds",
        &["project_id"],
        vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
    );
    (active_sessions, total_sessions, session_duration_seconds)
}

pub(super) fn create_queue_metrics() -> (IntGaugeVec, HistogramVec, IntCounterVec) {
    let queue_depth = int_gauge_vec(
        "queue_depth",
        "Current queue depth by priority and collection",
        &["priority", "collection"],
    );
    let queue_processing_time_seconds = histogram_vec(
        "queue_processing_time_seconds",
        "Queue item processing time in seconds",
        &["priority"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
    );
    let queue_items_processed_total = int_counter_vec(
        "queue_items_processed_total",
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
        "tenant_documents_total",
        "Total documents per tenant and collection",
        &["tenant_id", "collection"],
    );
    let tenant_search_requests_total = int_counter_vec(
        "tenant_search_requests_total",
        "Total search requests per tenant",
        &["tenant_id"],
    );
    let tenant_storage_bytes = gauge_vec(
        "tenant_storage_bytes",
        "Estimated storage bytes per tenant",
        &["tenant_id"],
    );
    (
        tenant_documents_total,
        tenant_search_requests_total,
        tenant_storage_bytes,
    )
}

pub(super) fn create_indexed_project_metrics(
) -> (IntGaugeVec, IntGaugeVec, IntGaugeVec, IntGaugeVec) {
    let indexed_project_tracked_files = int_gauge_vec(
        "indexed_project_tracked_files",
        "Tracked file count per indexed project with project inventory labels",
        &[
            "watch_id",
            "tenant_id",
            "path",
            "enabled",
            "is_active",
            "is_paused",
            "is_archived",
            "is_worktree",
            "is_git_tracked",
            "git_remote",
        ],
    );
    let indexed_project_points = int_gauge_vec(
        "indexed_project_points",
        "Qdrant point count per indexed project",
        &["watch_id"],
    );
    let indexed_project_last_scan_seconds = int_gauge_vec(
        "indexed_project_last_scan_seconds",
        "Unix epoch seconds of the most recent scan per indexed project",
        &["watch_id"],
    );
    let indexed_project_last_activity_seconds = int_gauge_vec(
        "indexed_project_last_activity_seconds",
        "Unix epoch seconds of the most recent activity per indexed project",
        &["watch_id"],
    );
    (
        indexed_project_tracked_files,
        indexed_project_points,
        indexed_project_last_scan_seconds,
        indexed_project_last_activity_seconds,
    )
}

pub(super) fn create_system_metrics() -> (GaugeVec, IntCounterVec, HistogramVec) {
    let uptime_seconds = gauge_vec("uptime_seconds", "Daemon uptime in seconds", &[]);
    let ingestion_errors_total = int_counter_vec(
        "ingestion_errors_total",
        "Total ingestion errors by error type",
        &["error_type"],
    );
    let heartbeat_latency_seconds = histogram_vec(
        "heartbeat_latency_seconds",
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
        "watch_errors_total",
        "Total watch errors by watch_id",
        &["watch_id"],
    );
    let watch_consecutive_errors = int_gauge_vec(
        "watch_consecutive_errors",
        "Current consecutive errors by watch_id",
        &["watch_id"],
    );
    let watch_health_status = int_gauge_vec(
        "watch_health_status",
        "Watch health status (1 = in this state)",
        &["watch_id", "health_status"],
    );
    let watches_in_backoff = int_gauge_vec(
        "watches_in_backoff",
        "Number of watches currently in backoff",
        &[],
    );
    let watch_recovery_time_seconds = histogram_vec(
        "watch_recovery_time_seconds",
        "Watch error recovery time in seconds",
        &["watch_id"],
        vec![
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0,
        ],
    );
    let watch_events_throttled_total = int_counter_vec(
        "watch_events_throttled_total",
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
        "embedding_duration_seconds",
        "Embedding generation duration in seconds by model",
        &["model"],
        vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    );
    let embedding_batch_size = histogram_vec(
        "embedding_batch_size",
        "Embedding batch size (items per call) by model",
        &["model"],
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0],
    );
    let sqlite_query_duration_seconds = histogram_vec(
        "sqlite_query_duration_seconds",
        "SQLite query duration in seconds by op",
        &["op"],
        vec![0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    );
    let qdrant_request_duration_seconds = histogram_vec(
        "qdrant_request_duration_seconds",
        "Qdrant request duration in seconds by op",
        &["op"],
        vec![0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    );
    let qdrant_request_errors_total = int_counter_vec(
        "qdrant_request_errors_total",
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
        "watcher_events_total",
        "Total filesystem watcher events by event_type",
        &["event_type"],
    );
    let watcher_coalesced_total = int_counter_vec(
        "watcher_coalesced_total",
        "Total watcher events coalesced before enqueue",
        &["reason"],
    );
    let grpc_requests_total = int_counter_vec(
        "grpc_requests_total",
        "Total gRPC requests by service, method, and status",
        &["service", "method", "status"],
    );
    let grpc_request_duration_seconds = histogram_vec(
        "grpc_request_duration_seconds",
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
        "unified_queue_depth",
        "Current unified queue depth by item_type and status",
        &["item_type", "status"],
    );
    let unified_queue_processing_time_seconds = histogram_vec(
        "unified_queue_processing_time_seconds",
        "Unified queue item processing time in seconds",
        &["item_type"],
        vec![
            0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0,
        ],
    );
    let unified_queue_items_total = int_counter_vec(
        "unified_queue_items_total",
        "Total unified queue items processed by item_type, op, and result",
        &["item_type", "op", "result"],
    );
    let unified_queue_enqueues_total = int_counter_vec(
        "unified_queue_enqueues_total",
        "Total unified queue enqueues by source",
        &["source"],
    );
    let unified_queue_dequeues_total = int_counter_vec(
        "unified_queue_dequeues_total",
        "Total unified queue dequeues by item_type",
        &["item_type"],
    );
    let unified_queue_stale_items = int_gauge_vec(
        "unified_queue_stale_items",
        "Stale lease items in unified queue",
        &[],
    );
    let unified_queue_retries_total = int_counter_vec(
        "unified_queue_retries_total",
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

/// Per-tenant indexing-progress gauge: `unified_queue` rows grouped by
/// `(tenant_id, status)` (status in pending / in_progress / failed).
///
/// Kept separate from `unified_queue_depth` to avoid label-cardinality
/// blow-up on the global gauge. Refreshed every 10s by the exporter.
pub(super) fn create_per_tenant_indexing_metric() -> IntGaugeVec {
    int_gauge_vec(
        "unified_queue_depth_by_tenant",
        "Current unified queue depth per tenant_id and status (excludes done)",
        &["tenant_id", "status"],
    )
}

/// Per-tenant ETA (seconds) derived from the rate at which
/// `tracked_files.updated_at` advances over a 5-minute window. Set to
/// `-1` when the daemon can't estimate (cold-start, rate==0, queue
/// drained) — Prometheus has no native null, and a sentinel lets PromQL
/// filter via `>= 0`.
pub(super) fn create_per_tenant_eta_metric() -> IntGaugeVec {
    int_gauge_vec(
        "indexing_eta_seconds_by_tenant",
        "Estimated seconds to drain the queue per tenant; -1 = unknown / warming up",
        &["tenant_id"],
    )
}

/// Per-(tenant, branch) `file_metadata` observability — refreshed every
/// 30s by the search.db exporter in [`memexd::background`].
///
/// Cardinality: bounded by the number of registered (tenant_id, branch)
/// pairs. No `path` label — that would be unbounded. For per-file
/// inspection, route operators to the admin UI / sidecar SQL queries
/// (see [[daemon-db-sidecar-query]] memory note).
#[allow(clippy::type_complexity)]
pub(super) fn create_file_metadata_metrics(
) -> (IntGaugeVec, IntGaugeVec, IntGaugeVec, IntCounterVec) {
    let indexed_files_count = int_gauge_vec(
        "indexed_files_count",
        "Number of files in search.db file_metadata per tenant and branch",
        &["tenant_id", "branch"],
    );
    let indexed_files_total_bytes = int_gauge_vec(
        "indexed_files_total_bytes",
        "Sum of file_metadata.size_bytes per tenant and branch (NULL sizes counted as 0)",
        &["tenant_id", "branch"],
    );
    let fts5_skipped_files_count = int_gauge_vec(
        "fts5_skipped_files_count",
        "Current number of files with fts5_skipped=1 per tenant and branch \
         (hard-cap bypass — see WQM_FTS5_HARD_CAP)",
        &["tenant_id", "branch"],
    );
    let fts5_skipped_files_total = int_counter_vec(
        "fts5_skipped_files_total",
        "Cumulative count of times the FTS5 hard cap fired on ingestion per tenant and branch",
        &["tenant_id", "branch"],
    );
    (
        indexed_files_count,
        indexed_files_total_bytes,
        fts5_skipped_files_count,
        fts5_skipped_files_total,
    )
}

/// LSP subsystem metrics
///
/// Returns `(lsp_server_state, lsp_enrichments_total)`.
///
/// - `lsp_server_state{language}` — 1 if the server for that language is
///   running in at least one project, 0 otherwise. Updated by the LSP
///   metrics background task.
/// - `lsp_enrichments_total{status}` — cumulative count of LSP enrichment
///   attempts; status values mirror [`EnrichmentStatus`]: success, partial,
///   failed, skipped. Incremented inline by the chunk-embed pipeline.
pub(super) fn create_lsp_metrics() -> (IntGaugeVec, IntCounterVec) {
    let lsp_server_state = int_gauge_vec(
        "lsp_server_state",
        "LSP server running state per language (1=running, 0=stopped/absent)",
        &["language"],
    );
    let lsp_enrichments_total = int_counter_vec(
        "lsp_enrichments_total",
        "Cumulative LSP chunk enrichment attempts by outcome status",
        &["status"],
    );
    (lsp_server_state, lsp_enrichments_total)
}
