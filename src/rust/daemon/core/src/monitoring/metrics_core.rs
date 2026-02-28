//! Prometheus metrics core: DaemonMetrics struct and METRICS global
//!
//! This module provides the core metrics collection for the daemon:
//! - DaemonMetrics struct with all metric fields
//! - METRICS global lazy static
//! - Helper methods for session, queue, tenant, system, watch error, and unified queue tracking

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

// ── Factory helpers to reduce verbosity in new() ──────────────────────

fn int_gauge_vec(name: &str, help: &str, labels: &[&str]) -> IntGaugeVec {
    IntGaugeVec::new(Opts::new(name, help).namespace("memexd"), labels)
        .expect("metric can be created")
}

fn int_counter_vec(name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
    IntCounterVec::new(Opts::new(name, help).namespace("memexd"), labels)
        .expect("metric can be created")
}

fn gauge_vec(name: &str, help: &str, labels: &[&str]) -> GaugeVec {
    GaugeVec::new(Opts::new(name, help).namespace("memexd"), labels)
        .expect("metric can be created")
}

fn histogram_vec(name: &str, help: &str, labels: &[&str], buckets: Vec<f64>) -> HistogramVec {
    HistogramVec::new(
        prometheus::HistogramOpts::new(name, help)
            .namespace("memexd")
            .buckets(buckets),
        labels,
    )
    .expect("metric can be created")
}

/// Register a batch of collectors in the registry
fn register_all(registry: &Registry, collectors: Vec<Box<dyn Collector>>) {
    for c in collectors {
        registry.register(c).expect("metric can be registered");
    }
}

// ── Per-category metric constructors ─────────────────────────────────

/// Create session tracking metrics
fn create_session_metrics() -> (IntGaugeVec, IntCounterVec, HistogramVec) {
    let active_sessions = int_gauge_vec(
        "memexd_active_sessions",
        "Number of active sessions by project and priority",
        &["project_id", "priority"],
    );
    let total_sessions = int_counter_vec(
        "memexd_total_sessions",
        "Total number of sessions created (lifetime)",
        &["project_id"],
    );
    let session_duration_seconds = histogram_vec(
        "memexd_session_duration_seconds",
        "Session duration in seconds",
        &["project_id"],
        vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
    );
    (active_sessions, total_sessions, session_duration_seconds)
}

/// Create queue depth and processing metrics
fn create_queue_metrics() -> (IntGaugeVec, HistogramVec, IntCounterVec) {
    let queue_depth = int_gauge_vec(
        "memexd_queue_depth",
        "Current queue depth by priority and collection",
        &["priority", "collection"],
    );
    let queue_processing_time_seconds = histogram_vec(
        "memexd_queue_processing_time_seconds",
        "Queue item processing time in seconds",
        &["priority"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
    );
    let queue_items_processed_total = int_counter_vec(
        "memexd_queue_items_processed_total",
        "Total items processed by priority and status",
        &["priority", "status"],
    );
    (queue_depth, queue_processing_time_seconds, queue_items_processed_total)
}

/// Create per-tenant tracking metrics
fn create_tenant_metrics() -> (IntGaugeVec, IntCounterVec, GaugeVec) {
    let tenant_documents_total = int_gauge_vec(
        "memexd_tenant_documents_total",
        "Total documents per tenant and collection",
        &["tenant_id", "collection"],
    );
    let tenant_search_requests_total = int_counter_vec(
        "memexd_tenant_search_requests_total",
        "Total search requests per tenant",
        &["tenant_id"],
    );
    let tenant_storage_bytes = gauge_vec(
        "memexd_tenant_storage_bytes",
        "Estimated storage bytes per tenant",
        &["tenant_id"],
    );
    (tenant_documents_total, tenant_search_requests_total, tenant_storage_bytes)
}

/// Create system-level metrics (uptime, errors, heartbeat)
fn create_system_metrics() -> (GaugeVec, IntCounterVec, HistogramVec) {
    let uptime_seconds = gauge_vec(
        "memexd_uptime_seconds",
        "Daemon uptime in seconds",
        &[],
    );
    let ingestion_errors_total = int_counter_vec(
        "memexd_ingestion_errors_total",
        "Total ingestion errors by error type",
        &["error_type"],
    );
    let heartbeat_latency_seconds = histogram_vec(
        "memexd_heartbeat_latency_seconds",
        "Heartbeat processing latency in seconds",
        &["project_id"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    );
    (uptime_seconds, ingestion_errors_total, heartbeat_latency_seconds)
}

/// Create watch error and health metrics (Task 461.12)
fn create_watch_metrics() -> (
    IntCounterVec, IntGaugeVec, IntGaugeVec, IntGaugeVec, HistogramVec, IntCounterVec,
) {
    let watch_errors_total = int_counter_vec(
        "memexd_watch_errors_total",
        "Total watch errors by watch_id",
        &["watch_id"],
    );
    let watch_consecutive_errors = int_gauge_vec(
        "memexd_watch_consecutive_errors",
        "Current consecutive errors by watch_id",
        &["watch_id"],
    );
    let watch_health_status = int_gauge_vec(
        "memexd_watch_health_status",
        "Watch health status (1 = in this state)",
        &["watch_id", "health_status"],
    );
    let watches_in_backoff = int_gauge_vec(
        "memexd_watches_in_backoff",
        "Number of watches currently in backoff",
        &[],
    );
    let watch_recovery_time_seconds = histogram_vec(
        "memexd_watch_recovery_time_seconds",
        "Watch error recovery time in seconds",
        &["watch_id"],
        vec![1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0],
    );
    let watch_events_throttled_total = int_counter_vec(
        "memexd_watch_events_throttled_total",
        "Events throttled due to queue depth",
        &["watch_id", "load_level"],
    );
    (
        watch_errors_total, watch_consecutive_errors, watch_health_status,
        watches_in_backoff, watch_recovery_time_seconds, watch_events_throttled_total,
    )
}

/// Create unified queue metrics (Task 37.35)
fn create_unified_queue_metrics() -> (
    IntGaugeVec, HistogramVec, IntCounterVec, IntCounterVec,
    IntCounterVec, IntGaugeVec, IntCounterVec,
) {
    let unified_queue_depth = int_gauge_vec(
        "memexd_unified_queue_depth",
        "Current unified queue depth by item_type and status",
        &["item_type", "status"],
    );
    let unified_queue_processing_time_seconds = histogram_vec(
        "memexd_unified_queue_processing_time_seconds",
        "Unified queue item processing time in seconds",
        &["item_type"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    );
    let unified_queue_items_total = int_counter_vec(
        "memexd_unified_queue_items_total",
        "Total unified queue items processed by item_type, op, and result",
        &["item_type", "op", "result"],
    );
    let unified_queue_enqueues_total = int_counter_vec(
        "memexd_unified_queue_enqueues_total",
        "Total unified queue enqueues by source",
        &["source"],
    );
    let unified_queue_dequeues_total = int_counter_vec(
        "memexd_unified_queue_dequeues_total",
        "Total unified queue dequeues by item_type",
        &["item_type"],
    );
    let unified_queue_stale_items = int_gauge_vec(
        "memexd_unified_queue_stale_items",
        "Stale lease items in unified queue",
        &[],
    );
    let unified_queue_retries_total = int_counter_vec(
        "memexd_unified_queue_retries_total",
        "Unified queue retry count by item_type",
        &["item_type"],
    );
    (
        unified_queue_depth, unified_queue_processing_time_seconds,
        unified_queue_items_total, unified_queue_enqueues_total,
        unified_queue_dequeues_total, unified_queue_stale_items,
        unified_queue_retries_total,
    )
}

impl DaemonMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let registry = Registry::new();

        let (active_sessions, total_sessions, session_duration_seconds) =
            create_session_metrics();
        let (queue_depth, queue_processing_time_seconds, queue_items_processed_total) =
            create_queue_metrics();
        let (tenant_documents_total, tenant_search_requests_total, tenant_storage_bytes) =
            create_tenant_metrics();
        let (uptime_seconds, ingestion_errors_total, heartbeat_latency_seconds) =
            create_system_metrics();
        let (
            watch_errors_total, watch_consecutive_errors, watch_health_status,
            watches_in_backoff, watch_recovery_time_seconds, watch_events_throttled_total,
        ) = create_watch_metrics();
        let (
            unified_queue_depth, unified_queue_processing_time_seconds,
            unified_queue_items_total, unified_queue_enqueues_total,
            unified_queue_dequeues_total, unified_queue_stale_items,
            unified_queue_retries_total,
        ) = create_unified_queue_metrics();

        register_all(&registry, vec![
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
        ]);

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

}

impl Default for DaemonMetrics {
    fn default() -> Self {
        Self::new()
    }
}
