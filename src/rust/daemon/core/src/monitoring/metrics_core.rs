//! Prometheus metrics core: [`DaemonMetrics`] struct and the [`METRICS`]
//! global lazy-static registry.
//!
//! Per-subsystem collector factories live in [`super::metrics_factories`];
//! convenience tracking helpers live in [`super::metrics_helpers`]; and unit
//! tests for the per-record helpers live in `metrics_core_tests`.

use once_cell::sync::Lazy;
use prometheus::{
    self, Encoder, Gauge, GaugeVec, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Registry,
    TextEncoder,
};

use super::metrics_factories::{
    create_dependency_metrics, create_file_metadata_metrics, create_graph_metrics,
    create_indexed_project_metrics, create_lsp_metrics, create_per_tenant_eta_metric,
    create_per_tenant_indexing_metric, create_queue_metrics, create_session_metrics,
    create_system_metrics, create_telemetry_extension_metrics, create_tenant_metrics,
    create_unified_queue_metrics, create_watch_metrics, register_all,
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

    /// Tracked file count per indexed project
    /// Labels: watch_id, tenant_id, path, enabled, is_active, is_paused, is_archived,
    /// is_worktree, is_git_tracked
    pub indexed_project_tracked_files: IntGaugeVec,

    /// Qdrant point count per indexed project
    /// Labels: watch_id
    pub indexed_project_points: IntGaugeVec,

    /// Last scan time in Unix epoch seconds per indexed project
    /// Labels: watch_id
    pub indexed_project_last_scan_seconds: IntGaugeVec,

    /// Last activity time in Unix epoch seconds per indexed project
    /// Labels: watch_id
    pub indexed_project_last_activity_seconds: IntGaugeVec,

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

    /// Per-tenant unified queue depth (indexing-progress observability).
    /// Labels: tenant_id, status (pending, in_progress, failed; 'done' excluded
    /// because those rows are deleted by `cleanup_completed_unified_items`).
    pub unified_queue_depth_by_tenant: IntGaugeVec,

    /// Per-tenant indexing ETA (seconds). Set to `-1` when unknown
    /// (cold-start / rate=0 / queue already drained).
    /// Labels: tenant_id
    pub indexing_eta_seconds_by_tenant: IntGaugeVec,

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

    // Process resource metrics (sampled every second from /proc/self).
    /// Resident set size of the memexd process in bytes
    pub process_resident_memory_bytes: IntGauge,
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

    // search.db / file_metadata observability (Task 3 of FTS5 size-guard series).
    /// Number of files in `file_metadata` per (tenant_id, branch).
    pub indexed_files_count: IntGaugeVec,

    /// Sum of `file_metadata.size_bytes` per (tenant_id, branch).
    pub indexed_files_total_bytes: IntGaugeVec,

    /// Current number of files with `fts5_skipped = 1` per (tenant_id, branch).
    /// Refreshed alongside `indexed_files_count` by the search.db exporter.
    pub fts5_skipped_files_count: IntGaugeVec,

    /// Cumulative count of `WQM_FTS5_HARD_CAP` firings per (tenant_id, branch).
    /// Incremented inline by [`enforce_fts5_hard_cap_skip`] — distinct from
    /// `fts5_skipped_files_count` (which is a current snapshot and can go
    /// down when files shrink below the cap and get re-ingested).
    pub fts5_skipped_files_total: IntCounterVec,

    // ── LSP observability ─────────────────────────────────────────────────
    /// Running state per language: 1 = at least one server running, 0 = none.
    /// Labels: language
    pub lsp_server_state: IntGaugeVec,

    /// Cumulative LSP enrichment attempts by outcome.
    /// Labels: status (success, partial, failed, skipped, pending)
    pub lsp_enrichments_total: IntCounterVec,

    /// Number of language server binaries detected in PATH at last scan.
    pub lsp_available_languages: IntGauge,

    /// Number of LSP server instances currently running across all projects.
    pub lsp_active_servers: IntGauge,

    // ── Code-relationship graph observability ─────────────────────────────
    /// Graph node count by tenant and node type. Labels: tenant_id, node_type
    pub graph_nodes: IntGaugeVec,

    /// Graph edge count by tenant and edge type. Labels: tenant_id, edge_type
    pub graph_edges: IntGaugeVec,

    /// Unresolved stub nodes (empty file_path) by tenant. Labels: tenant_id
    pub graph_unresolved_stubs: IntGaugeVec,

    /// Cumulative stub edges repointed by the resolver. Labels: tenant_id
    pub graph_stub_resolved_total: IntCounterVec,

    /// Cumulative graph edges written during ingest. Labels: tenant_id, edge_type
    pub graph_edges_ingested_total: IntCounterVec,
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
    indexed_project_tracked_files: IntGaugeVec,
    indexed_project_points: IntGaugeVec,
    indexed_project_last_scan_seconds: IntGaugeVec,
    indexed_project_last_activity_seconds: IntGaugeVec,
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
    unified_queue_depth_by_tenant: IntGaugeVec,
    indexing_eta_seconds_by_tenant: IntGaugeVec,
    unified_queue_processing_time_seconds: HistogramVec,
    unified_queue_items_total: IntCounterVec,
    unified_queue_enqueues_total: IntCounterVec,
    unified_queue_dequeues_total: IntCounterVec,
    unified_queue_stale_items: IntGaugeVec,
    unified_queue_retries_total: IntCounterVec,
    queue_oldest_pending_age_seconds: IntGauge,
    process_resident_memory_bytes: IntGauge,
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
    indexed_files_count: IntGaugeVec,
    indexed_files_total_bytes: IntGaugeVec,
    fts5_skipped_files_count: IntGaugeVec,
    fts5_skipped_files_total: IntCounterVec,
    lsp_server_state: IntGaugeVec,
    lsp_enrichments_total: IntCounterVec,
    lsp_available_languages: IntGauge,
    lsp_active_servers: IntGauge,
    graph_nodes: IntGaugeVec,
    graph_edges: IntGaugeVec,
    graph_unresolved_stubs: IntGaugeVec,
    graph_stub_resolved_total: IntCounterVec,
    graph_edges_ingested_total: IntCounterVec,
}

/// Create all metric instances from subsystem factories.
fn create_all_metrics() -> CreatedMetrics {
    let (active_sessions, total_sessions, session_duration_seconds) = create_session_metrics();
    let (queue_depth, queue_processing_time_seconds, queue_items_processed_total) =
        create_queue_metrics();
    let (tenant_documents_total, tenant_search_requests_total, tenant_storage_bytes) =
        create_tenant_metrics();
    let (
        indexed_project_tracked_files,
        indexed_project_points,
        indexed_project_last_scan_seconds,
        indexed_project_last_activity_seconds,
    ) = create_indexed_project_metrics();
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
    let unified_queue_depth_by_tenant = create_per_tenant_indexing_metric();
    let indexing_eta_seconds_by_tenant = create_per_tenant_eta_metric();
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

    let queue_oldest_pending_age_seconds = IntGauge::new(
        "wqm_queue_oldest_pending_age_seconds",
        "Age in seconds of the oldest pending queue item",
    )
    .expect("metric can be created");

    let process_resident_memory_bytes = IntGauge::new(
        "memexd_process_resident_memory_bytes",
        "Resident set size of the memexd process in bytes",
    )
    .expect("metric can be created");
    let process_cpu_percent = Gauge::new(
        "memexd_process_cpu_percent",
        "CPU usage of the memexd process in percent (100 = one full core)",
    )
    .expect("metric can be created");

    let (
        indexed_files_count,
        indexed_files_total_bytes,
        fts5_skipped_files_count,
        fts5_skipped_files_total,
    ) = create_file_metadata_metrics();

    let (lsp_server_state, lsp_enrichments_total) = create_lsp_metrics();

    let lsp_available_languages = IntGauge::new(
        "memexd_lsp_available_languages",
        "Language server binaries detected in PATH at last scan",
    )
    .expect("metric can be created");

    let lsp_active_servers = IntGauge::new(
        "memexd_lsp_active_servers",
        "LSP server instances currently running across all projects",
    )
    .expect("metric can be created");

    let (
        graph_nodes,
        graph_edges,
        graph_unresolved_stubs,
        graph_stub_resolved_total,
        graph_edges_ingested_total,
    ) = create_graph_metrics();

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
        indexed_project_tracked_files,
        indexed_project_points,
        indexed_project_last_scan_seconds,
        indexed_project_last_activity_seconds,
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
        unified_queue_depth_by_tenant,
        indexing_eta_seconds_by_tenant,
        unified_queue_processing_time_seconds,
        unified_queue_items_total,
        unified_queue_enqueues_total,
        unified_queue_dequeues_total,
        unified_queue_stale_items,
        unified_queue_retries_total,
        queue_oldest_pending_age_seconds,
        process_resident_memory_bytes,
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
        indexed_files_count,
        indexed_files_total_bytes,
        fts5_skipped_files_count,
        fts5_skipped_files_total,
        lsp_server_state,
        lsp_enrichments_total,
        lsp_available_languages,
        lsp_active_servers,
        graph_nodes,
        graph_edges,
        graph_unresolved_stubs,
        graph_stub_resolved_total,
        graph_edges_ingested_total,
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
            Box::new(m.indexed_project_tracked_files.clone()),
            Box::new(m.indexed_project_points.clone()),
            Box::new(m.indexed_project_last_scan_seconds.clone()),
            Box::new(m.indexed_project_last_activity_seconds.clone()),
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
            Box::new(m.unified_queue_depth_by_tenant.clone()),
            Box::new(m.indexing_eta_seconds_by_tenant.clone()),
            Box::new(m.unified_queue_processing_time_seconds.clone()),
            Box::new(m.unified_queue_items_total.clone()),
            Box::new(m.unified_queue_enqueues_total.clone()),
            Box::new(m.unified_queue_dequeues_total.clone()),
            Box::new(m.unified_queue_stale_items.clone()),
            Box::new(m.unified_queue_retries_total.clone()),
            Box::new(m.queue_oldest_pending_age_seconds.clone()),
            Box::new(m.process_resident_memory_bytes.clone()),
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
            Box::new(m.indexed_files_count.clone()),
            Box::new(m.indexed_files_total_bytes.clone()),
            Box::new(m.fts5_skipped_files_count.clone()),
            Box::new(m.fts5_skipped_files_total.clone()),
            Box::new(m.lsp_server_state.clone()),
            Box::new(m.lsp_enrichments_total.clone()),
            Box::new(m.lsp_available_languages.clone()),
            Box::new(m.lsp_active_servers.clone()),
            Box::new(m.graph_nodes.clone()),
            Box::new(m.graph_edges.clone()),
            Box::new(m.graph_unresolved_stubs.clone()),
            Box::new(m.graph_stub_resolved_total.clone()),
            Box::new(m.graph_edges_ingested_total.clone()),
        ],
    );
}

impl DaemonMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let registry = Registry::new();
        let m = create_all_metrics();
        register_metrics(&registry, &m);

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
            indexed_project_tracked_files: m.indexed_project_tracked_files,
            indexed_project_points: m.indexed_project_points,
            indexed_project_last_scan_seconds: m.indexed_project_last_scan_seconds,
            indexed_project_last_activity_seconds: m.indexed_project_last_activity_seconds,
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
            unified_queue_depth_by_tenant: m.unified_queue_depth_by_tenant,
            indexing_eta_seconds_by_tenant: m.indexing_eta_seconds_by_tenant,
            unified_queue_processing_time_seconds: m.unified_queue_processing_time_seconds,
            unified_queue_items_total: m.unified_queue_items_total,
            unified_queue_enqueues_total: m.unified_queue_enqueues_total,
            unified_queue_dequeues_total: m.unified_queue_dequeues_total,
            unified_queue_stale_items: m.unified_queue_stale_items,
            unified_queue_retries_total: m.unified_queue_retries_total,
            queue_oldest_pending_age_seconds: m.queue_oldest_pending_age_seconds,
            process_resident_memory_bytes: m.process_resident_memory_bytes,
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
            indexed_files_count: m.indexed_files_count,
            indexed_files_total_bytes: m.indexed_files_total_bytes,
            fts5_skipped_files_count: m.fts5_skipped_files_count,
            fts5_skipped_files_total: m.fts5_skipped_files_total,
            lsp_server_state: m.lsp_server_state,
            lsp_enrichments_total: m.lsp_enrichments_total,
            lsp_available_languages: m.lsp_available_languages,
            lsp_active_servers: m.lsp_active_servers,
            graph_nodes: m.graph_nodes,
            graph_edges: m.graph_edges,
            graph_unresolved_stubs: m.graph_unresolved_stubs,
            graph_stub_resolved_total: m.graph_stub_resolved_total,
            graph_edges_ingested_total: m.graph_edges_ingested_total,
        }
    }

    /// Snapshot the search.db `file_metadata` stats for one (tenant, branch).
    /// Called every ~30s by the search.db exporter in `memexd::background`.
    pub fn set_file_metadata_stats(
        &self,
        tenant_id: &str,
        branch: &str,
        file_count: i64,
        total_bytes: i64,
        skipped_count: i64,
    ) {
        self.indexed_files_count
            .with_label_values(&[tenant_id, branch])
            .set(file_count);
        self.indexed_files_total_bytes
            .with_label_values(&[tenant_id, branch])
            .set(total_bytes);
        self.fts5_skipped_files_count
            .with_label_values(&[tenant_id, branch])
            .set(skipped_count);
    }

    /// Increment the FTS5 hard-cap counter for a single skip event.
    pub fn inc_fts5_skipped(&self, tenant_id: &str, branch: &str) {
        self.fts5_skipped_files_total
            .with_label_values(&[tenant_id, branch])
            .inc();
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

    // ── LSP helpers ───────────────────────────────────────────────────────

    /// Increment the LSP enrichment counter for one chunk outcome.
    ///
    /// Called inline by the chunk-embed pipeline when an enrichment attempt
    /// completes (or is skipped). `status` should be one of: `"success"`,
    /// `"partial"`, `"failed"`, `"skipped"`, `"pending"`.
    pub fn inc_lsp_enrichment(&self, status: &str) {
        self.lsp_enrichments_total
            .with_label_values(&[status])
            .inc();
    }

    /// Set the running state for a language (1 = running, 0 = stopped).
    ///
    /// Called by the LSP metrics background task every 30 s.
    pub fn set_lsp_server_state(&self, language: &str, running: bool) {
        self.lsp_server_state
            .with_label_values(&[language])
            .set(if running { 1 } else { 0 });
    }

    /// Update the language-availability and active-server snapshot gauges.
    ///
    /// Called by the LSP metrics background task every 30 s.
    pub fn set_lsp_snapshot(&self, available_languages: i64, active_servers: i64) {
        self.lsp_available_languages.set(available_languages);
        self.lsp_active_servers.set(active_servers);
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
