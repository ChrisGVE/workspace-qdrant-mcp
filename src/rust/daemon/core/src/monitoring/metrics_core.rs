//! Prometheus metrics core: [`DaemonMetrics`] struct and the [`METRICS`]
//! global lazy-static registry.
//!
//! Per-subsystem collector factories live in [`super::metrics_factories`];
//! convenience tracking helpers live in [`super::metrics_helpers`]; and unit
//! tests for the per-record helpers live in `metrics_core_tests`.

use once_cell::sync::Lazy;
use prometheus::{
    self, Encoder, Gauge, GaugeVec, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec,
    Opts, Registry, TextEncoder,
};

use super::metrics_factories::{
    create_dependency_metrics, create_indexing_metrics, create_processing_metrics,
    create_queue_metrics, create_red_use_metrics, create_session_metrics, create_system_metrics,
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

    /// Tracked files in search.db file_metadata (#118)
    /// Labels: tenant_id, branch
    pub tracked_files: IntGaugeVec,

    /// Files skipped by the per-extension size gate (#118)
    /// Labels: tenant_id
    pub files_size_skipped_total: IntCounterVec,

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

    /// Circuit breaker pause events by subsystem (qdrant, sqlite)
    /// Labels: subsystem
    pub circuit_breaker_pauses_total: IntCounterVec,

    // Process resource metrics (#103: sampled every second so a memory
    // runaway alerts instead of killing the daemon).
    /// Resident set size of the memexd process in bytes
    pub process_resident_memory_bytes: IntGauge,
    /// Dirty (physical) footprint of the memexd process in bytes. On macOS
    /// RSS counts allocator-cached freed pages and overstates real usage by
    /// several×; this gauge is the honest signal to alert on.
    pub process_footprint_bytes: IntGauge,
    /// CPU usage of the memexd process in percent (100 = one full core)
    pub process_cpu_percent: Gauge,

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

    // Dimensional processing metrics (A2)
    /// Per-item processing pipeline latency in seconds.
    /// Labels: collection, file_type, language, operation, embedding_engine
    pub processing_duration_seconds: HistogramVec,

    // RED/USE coverage (B6)
    /// Vector search latency in seconds. Labels: collection, mode
    pub search_duration_seconds: HistogramVec,

    /// Result-set size per search. Labels: tenant_id, collection
    pub search_result_count: HistogramVec,

    /// Number of in-flight embedding operations (embedder saturation).
    pub embedding_inflight: IntGauge,

    /// Total SQLite busy/locked (`SQLITE_BUSY`) occurrences.
    pub sqlite_busy_total: IntCounter,

    /// Samples dropped because the metrics-switchboard telemetry ring was full
    /// (telemetry sheds under back-pressure; the control path is unaffected).
    pub switchboard_buffer_full_total: IntCounter,

    /// Telemetry kill switch (from `observability.metrics.enabled`). When false,
    /// the A2 dimensional emission path is skipped so no series are created.
    pub enabled: std::sync::atomic::AtomicBool,
}

/// Intermediate struct holding all created metrics before registration.
struct CreatedMetrics {
    active_sessions: IntGaugeVec,
    total_sessions: IntCounterVec,
    session_duration_seconds: HistogramVec,
    queue_depth: IntGaugeVec,
    queue_processing_time_seconds: HistogramVec,
    queue_items_processed_total: IntCounterVec,
    tenant_documents_total: IntGaugeVec,
    tenant_search_requests_total: IntCounterVec,
    tenant_storage_bytes: GaugeVec,
    tracked_files: IntGaugeVec,
    files_size_skipped_total: IntCounterVec,
    uptime_seconds: GaugeVec,
    ingestion_errors_total: IntCounterVec,
    heartbeat_latency_seconds: HistogramVec,
    watch_errors_total: IntCounterVec,
    watch_consecutive_errors: IntGaugeVec,
    watch_health_status: IntGaugeVec,
    watches_in_backoff: IntGaugeVec,
    watch_recovery_time_seconds: HistogramVec,
    watch_events_throttled_total: IntCounterVec,
    unified_queue_depth: IntGaugeVec,
    unified_queue_processing_time_seconds: HistogramVec,
    unified_queue_items_total: IntCounterVec,
    unified_queue_enqueues_total: IntCounterVec,
    unified_queue_dequeues_total: IntCounterVec,
    unified_queue_stale_items: IntGaugeVec,
    unified_queue_retries_total: IntCounterVec,
    queue_oldest_pending_age_seconds: IntGauge,
    circuit_breaker_pauses_total: IntCounterVec,
    process_resident_memory_bytes: IntGauge,
    process_footprint_bytes: IntGauge,
    process_cpu_percent: Gauge,
    watcher_events_total: IntCounterVec,
    watcher_coalesced_total: IntCounterVec,
    grpc_requests_total: IntCounterVec,
    grpc_request_duration_seconds: HistogramVec,
    embedding_duration_seconds: HistogramVec,
    embedding_batch_size: HistogramVec,
    sqlite_query_duration_seconds: HistogramVec,
    qdrant_request_duration_seconds: HistogramVec,
    qdrant_request_errors_total: IntCounterVec,
    processing_duration_seconds: HistogramVec,
    search_duration_seconds: HistogramVec,
    search_result_count: HistogramVec,
    embedding_inflight: IntGauge,
    sqlite_busy_total: IntCounter,
    switchboard_buffer_full_total: IntCounter,
}

/// Create all metric instances from subsystem factories.
fn create_all_metrics() -> CreatedMetrics {
    let (active_sessions, total_sessions, session_duration_seconds) = create_session_metrics();
    let (queue_depth, queue_processing_time_seconds, queue_items_processed_total) =
        create_queue_metrics();
    let (tenant_documents_total, tenant_search_requests_total, tenant_storage_bytes) =
        create_tenant_metrics();
    let (tracked_files, files_size_skipped_total) = create_indexing_metrics();
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
    let processing_duration_seconds = create_processing_metrics();
    let (search_duration_seconds, search_result_count, embedding_inflight, sqlite_busy_total) =
        create_red_use_metrics();

    let queue_oldest_pending_age_seconds = IntGauge::new(
        "wqm_memexd_queue_oldest_pending_age_seconds",
        "Age in seconds of the oldest pending queue item",
    )
    .expect("metric can be created");

    let switchboard_buffer_full_total = IntCounter::new(
        "wqm_memexd_switchboard_buffer_full_total",
        "Samples dropped due to a full metrics-switchboard telemetry buffer",
    )
    .expect("metric can be created");

    let process_resident_memory_bytes = IntGauge::new(
        "wqm_memexd_process_resident_memory_bytes",
        "Resident set size of the memexd process in bytes",
    )
    .expect("metric can be created");
    let process_footprint_bytes = IntGauge::new(
        "wqm_memexd_process_footprint_bytes",
        "Dirty (physical) footprint of the memexd process in bytes — the honest \
         memory signal; RSS overstates by counting allocator-cached freed pages",
    )
    .expect("metric can be created");
    let process_cpu_percent = Gauge::new(
        "wqm_memexd_process_cpu_percent",
        "CPU usage of the memexd process in percent (100 = one full core)",
    )
    .expect("metric can be created");

    let circuit_breaker_pauses_total = IntCounterVec::new(
        Opts::new(
            "wqm_memexd_circuit_breaker_pauses_total",
            "Circuit breaker pause events by subsystem",
        ),
        &["subsystem"],
    )
    .expect("metric can be created");

    CreatedMetrics {
        active_sessions,
        total_sessions,
        session_duration_seconds,
        queue_depth,
        queue_processing_time_seconds,
        queue_items_processed_total,
        tenant_documents_total,
        tenant_search_requests_total,
        tenant_storage_bytes,
        tracked_files,
        files_size_skipped_total,
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
        circuit_breaker_pauses_total,
        process_resident_memory_bytes,
        process_footprint_bytes,
        process_cpu_percent,
        watcher_events_total,
        watcher_coalesced_total,
        grpc_requests_total,
        grpc_request_duration_seconds,
        embedding_duration_seconds,
        embedding_batch_size,
        sqlite_query_duration_seconds,
        qdrant_request_duration_seconds,
        qdrant_request_errors_total,
        processing_duration_seconds,
        search_duration_seconds,
        search_result_count,
        embedding_inflight,
        sqlite_busy_total,
        switchboard_buffer_full_total,
    }
}

/// Register all metrics with the given registry.
fn register_metrics(registry: &Registry, m: &CreatedMetrics) {
    register_all(
        registry,
        vec![
            Box::new(m.active_sessions.clone()),
            Box::new(m.total_sessions.clone()),
            Box::new(m.session_duration_seconds.clone()),
            Box::new(m.queue_depth.clone()),
            Box::new(m.queue_processing_time_seconds.clone()),
            Box::new(m.queue_items_processed_total.clone()),
            Box::new(m.tenant_documents_total.clone()),
            Box::new(m.tenant_search_requests_total.clone()),
            Box::new(m.tenant_storage_bytes.clone()),
            Box::new(m.tracked_files.clone()),
            Box::new(m.files_size_skipped_total.clone()),
            Box::new(m.uptime_seconds.clone()),
            Box::new(m.ingestion_errors_total.clone()),
            Box::new(m.heartbeat_latency_seconds.clone()),
            Box::new(m.watch_errors_total.clone()),
            Box::new(m.watch_consecutive_errors.clone()),
            Box::new(m.watch_health_status.clone()),
            Box::new(m.watches_in_backoff.clone()),
            Box::new(m.watch_recovery_time_seconds.clone()),
            Box::new(m.watch_events_throttled_total.clone()),
            Box::new(m.unified_queue_depth.clone()),
            Box::new(m.unified_queue_processing_time_seconds.clone()),
            Box::new(m.unified_queue_items_total.clone()),
            Box::new(m.unified_queue_enqueues_total.clone()),
            Box::new(m.unified_queue_dequeues_total.clone()),
            Box::new(m.unified_queue_stale_items.clone()),
            Box::new(m.unified_queue_retries_total.clone()),
            Box::new(m.queue_oldest_pending_age_seconds.clone()),
            Box::new(m.circuit_breaker_pauses_total.clone()),
            Box::new(m.process_resident_memory_bytes.clone()),
            Box::new(m.process_footprint_bytes.clone()),
            Box::new(m.process_cpu_percent.clone()),
            Box::new(m.watcher_events_total.clone()),
            Box::new(m.watcher_coalesced_total.clone()),
            Box::new(m.grpc_requests_total.clone()),
            Box::new(m.grpc_request_duration_seconds.clone()),
            Box::new(m.embedding_duration_seconds.clone()),
            Box::new(m.embedding_batch_size.clone()),
            Box::new(m.sqlite_query_duration_seconds.clone()),
            Box::new(m.qdrant_request_duration_seconds.clone()),
            Box::new(m.qdrant_request_errors_total.clone()),
            Box::new(m.processing_duration_seconds.clone()),
            Box::new(m.search_duration_seconds.clone()),
            Box::new(m.search_result_count.clone()),
            Box::new(m.embedding_inflight.clone()),
            Box::new(m.sqlite_busy_total.clone()),
            Box::new(m.switchboard_buffer_full_total.clone()),
        ],
    );
}

impl DaemonMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let registry = Registry::new();
        let m = create_all_metrics();
        register_metrics(&registry, &m);
        // Embedding-provider families live in a process-global handle; wire
        // them into THIS registry so /metrics and the OTLP bridge export them
        // (#101 — default_registry() is gathered by nothing).
        super::embedding_metrics::register_embedding_provider_metrics(&registry);

        Self {
            registry,
            active_sessions: m.active_sessions,
            total_sessions: m.total_sessions,
            session_duration_seconds: m.session_duration_seconds,
            queue_depth: m.queue_depth,
            queue_processing_time_seconds: m.queue_processing_time_seconds,
            queue_items_processed_total: m.queue_items_processed_total,
            tenant_documents_total: m.tenant_documents_total,
            tenant_search_requests_total: m.tenant_search_requests_total,
            tenant_storage_bytes: m.tenant_storage_bytes,
            tracked_files: m.tracked_files,
            files_size_skipped_total: m.files_size_skipped_total,
            uptime_seconds: m.uptime_seconds,
            ingestion_errors_total: m.ingestion_errors_total,
            heartbeat_latency_seconds: m.heartbeat_latency_seconds,
            watch_errors_total: m.watch_errors_total,
            watch_consecutive_errors: m.watch_consecutive_errors,
            watch_health_status: m.watch_health_status,
            watches_in_backoff: m.watches_in_backoff,
            watch_recovery_time_seconds: m.watch_recovery_time_seconds,
            watch_events_throttled_total: m.watch_events_throttled_total,
            unified_queue_depth: m.unified_queue_depth,
            unified_queue_processing_time_seconds: m.unified_queue_processing_time_seconds,
            unified_queue_items_total: m.unified_queue_items_total,
            unified_queue_enqueues_total: m.unified_queue_enqueues_total,
            unified_queue_dequeues_total: m.unified_queue_dequeues_total,
            unified_queue_stale_items: m.unified_queue_stale_items,
            unified_queue_retries_total: m.unified_queue_retries_total,
            queue_oldest_pending_age_seconds: m.queue_oldest_pending_age_seconds,
            circuit_breaker_pauses_total: m.circuit_breaker_pauses_total,
            process_resident_memory_bytes: m.process_resident_memory_bytes,
            process_footprint_bytes: m.process_footprint_bytes,
            process_cpu_percent: m.process_cpu_percent,
            watcher_events_total: m.watcher_events_total,
            watcher_coalesced_total: m.watcher_coalesced_total,
            grpc_requests_total: m.grpc_requests_total,
            grpc_request_duration_seconds: m.grpc_request_duration_seconds,
            embedding_duration_seconds: m.embedding_duration_seconds,
            embedding_batch_size: m.embedding_batch_size,
            sqlite_query_duration_seconds: m.sqlite_query_duration_seconds,
            qdrant_request_duration_seconds: m.qdrant_request_duration_seconds,
            qdrant_request_errors_total: m.qdrant_request_errors_total,
            processing_duration_seconds: m.processing_duration_seconds,
            search_duration_seconds: m.search_duration_seconds,
            search_result_count: m.search_result_count,
            embedding_inflight: m.embedding_inflight,
            sqlite_busy_total: m.sqlite_busy_total,
            switchboard_buffer_full_total: m.switchboard_buffer_full_total,
            enabled: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Encode all metrics in Prometheus text format
    pub fn encode(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        String::from_utf8(buffer)
            .map_err(|e| prometheus::Error::Msg(format!("metrics buffer is not valid UTF-8: {e}")))
    }
}

impl Default for DaemonMetrics {
    fn default() -> Self {
        Self::new()
    }
}
