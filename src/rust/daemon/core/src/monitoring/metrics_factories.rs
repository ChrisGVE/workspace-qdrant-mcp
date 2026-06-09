//! Factory functions for constructing the per-subsystem Prometheus metric
//! collectors used by [`DaemonMetrics`](super::metrics_core::DaemonMetrics).
//!
//! The factories are split out of `metrics_core.rs` to keep that file under
//! the project's 500-line ceiling and to give each subsystem an obvious
//! co-located definition. Each `create_*_metrics()` returns a tuple of
//! `prometheus` collectors which the constructor in `metrics_core` registers
//! and stores on the struct.

use prometheus::{
    self, core::Collector, GaugeVec, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts, Registry,
};

// ── Frozen histogram bucket layouts (stable API — A5) ────────────────────
//
// These bucket boundary sets are a **stable API**: dashboards built on
// `histogram_quantile()` and heatmaps interpolate across these boundaries, so
// changing a layout silently breaks every panel that spans the changed edge.
// To change buckets, RENAME the metric instead of editing the layout. Both
// consts are documented in `docker/docs/telemetry.md`.

/// Bucket layout for `wqm_memexd_embedding_duration_seconds`.
///
/// Extends the original `[…5.0]` spread with `10.0` and `30.0` upper buckets so
/// cold model loads and large-batch embeds (which exceed 5s) are measurable in
/// p99 instead of collapsing into `+Inf` (fixes C9).
pub const EMBEDDING_DURATION_BUCKETS: &[f64] = &[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0];

/// Bucket layout for the dimensional `wqm_memexd_processing_duration_seconds`
/// histogram (emitted by A2). Frozen here, ahead of A2, so the per-item
/// processing latency histogram ships with a stable boundary set. A strict
/// superset of the embedding layout's upper region (both end `…10.0, 30.0`) so
/// quantile comparisons across the two histograms align on shared boundaries.
pub const PROCESSING_DURATION_BUCKETS: &[f64] =
    &[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];

/// Bucket layout for `wqm_memexd_search_duration_seconds` (B6). Covers fast
/// in-memory hits up to multi-second cold/hybrid searches. Frozen stable API.
pub const SEARCH_DURATION_BUCKETS: &[f64] =
    &[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0];

/// Bucket layout for `wqm_memexd_search_result_count` (B6) — a histogram of
/// per-search result-set sizes (counts, not seconds). Frozen stable API.
pub const SEARCH_RESULT_COUNT_BUCKETS: &[f64] =
    &[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0];

// ── Small typed builders to reduce verbosity in factories ────────────────

pub(super) fn int_gauge_vec(name: &str, help: &str, labels: &[&str]) -> IntGaugeVec {
    IntGaugeVec::new(Opts::new(name, help), labels).expect("metric can be created")
}

pub(super) fn int_counter_vec(name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
    IntCounterVec::new(Opts::new(name, help), labels).expect("metric can be created")
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
        prometheus::HistogramOpts::new(name, help).buckets(buckets),
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

/// Indexing-coverage metrics (#118): tracked-file counts and size-gate skips.
///
/// - `tracked_files` — number of files in `search.db`'s `file_metadata` per
///   `(tenant, branch)`, exported by a periodic snapshot for repo-coverage and
///   FTS5-pressure visibility.
/// - `files_size_skipped_total` — files the per-extension size gate declined to
///   index, per tenant. Complements #113/#121 by making size-gate skips
///   observable instead of silent.
pub(super) fn create_indexing_metrics() -> (IntGaugeVec, IntCounterVec) {
    let tracked_files = int_gauge_vec(
        "wqm_memexd_tracked_files",
        "Number of tracked files in search.db file_metadata per tenant and branch",
        &["tenant_id", "branch"],
    );
    let files_size_skipped_total = int_counter_vec(
        "wqm_memexd_files_size_skipped_total",
        "Files skipped by the per-extension size gate, per tenant (#118)",
        &["tenant_id"],
    );
    (tracked_files, files_size_skipped_total)
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
        EMBEDDING_DURATION_BUCKETS.to_vec(),
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

/// Dimensional per-item processing-latency histogram (A2). Companion to the
/// existing `wqm_memexd_unified_queue_processing_time_seconds{item_type}`; this
/// one carries the five-way drill-down `{collection, file_type, language,
/// operation, embedding_engine}`. `file_type`/`language` are bounded at
/// emission by the A1 cardinality helper; `operation` is the 8-value queue-op
/// enum; `embedding_engine` is the 6-value bounded provider label. It is NOT a
/// graph metric — it carries no `layer` label. Buckets are the frozen A5
/// [`PROCESSING_DURATION_BUCKETS`] layout.
pub(super) fn create_processing_metrics() -> HistogramVec {
    histogram_vec(
        "wqm_memexd_processing_duration_seconds",
        "Per-item processing pipeline latency in seconds",
        &[
            "collection",
            "file_type",
            "language",
            "operation",
            "embedding_engine",
        ],
        PROCESSING_DURATION_BUCKETS.to_vec(),
    )
}

/// RED/USE coverage metrics (B6): search latency + result-size histograms, the
/// embedder-saturation in-flight gauge, and the SQLite busy/locked counter.
/// Histograms use the frozen [`SEARCH_DURATION_BUCKETS`] /
/// [`SEARCH_RESULT_COUNT_BUCKETS`] layouts.
pub(super) fn create_red_use_metrics() -> (HistogramVec, HistogramVec, IntGauge, IntCounter) {
    let search_duration_seconds = histogram_vec(
        "wqm_memexd_search_duration_seconds",
        "Vector search latency in seconds by collection and mode",
        &["collection", "mode"],
        SEARCH_DURATION_BUCKETS.to_vec(),
    );
    let search_result_count = histogram_vec(
        "wqm_memexd_search_result_count",
        "Number of results returned per search by tenant and collection",
        &["tenant_id", "collection"],
        SEARCH_RESULT_COUNT_BUCKETS.to_vec(),
    );
    let embedding_inflight = IntGauge::new(
        "wqm_memexd_embedding_inflight",
        "Number of in-flight embedding operations (embedder saturation)",
    )
    .expect("metric can be created");
    let sqlite_busy_total = IntCounter::new(
        "wqm_memexd_sqlite_busy_total",
        "Total SQLite busy/locked (SQLITE_BUSY) occurrences (lock-wait saturation)",
    )
    .expect("metric can be created");
    (
        search_duration_seconds,
        search_result_count,
        embedding_inflight,
        sqlite_busy_total,
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
