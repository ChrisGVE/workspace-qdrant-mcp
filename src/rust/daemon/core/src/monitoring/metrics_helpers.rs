//! Convenience helper methods for DaemonMetrics
//!
//! Provides organized tracking methods for sessions, queues, tenants,
//! system metrics, watch errors, and unified queue operations.

use std::path::Path;

use super::labels::cardinality::{bounded_file_type, bounded_language, OTHER};
use super::metrics_core::DaemonMetrics;

// ── Session tracking helpers ──────────────────────────────────────

impl DaemonMetrics {
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
    pub fn session_priority_changed(
        &self,
        project_id: &str,
        old_priority: &str,
        new_priority: &str,
    ) {
        self.active_sessions
            .with_label_values(&[project_id, old_priority])
            .dec();
        self.active_sessions
            .with_label_values(&[project_id, new_priority])
            .inc();
    }

    // ── Queue tracking helpers ────────────────────────────────────────

    /// Update queue depth
    pub fn set_queue_depth(&self, priority: &str, collection: &str, depth: i64) {
        self.queue_depth
            .with_label_values(&[priority, collection])
            .set(depth);
    }

    /// Record a queue item processed
    pub fn queue_item_processed(&self, priority: &str, status: &str, processing_time_secs: f64) {
        self.queue_items_processed_total
            .with_label_values(&[priority, status])
            .inc();
        self.queue_processing_time_seconds
            .with_label_values(&[priority])
            .observe(processing_time_secs);
    }

    // ── Tenant tracking helpers ───────────────────────────────────────

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

    // ── System helpers ────────────────────────────────────────────────

    /// Update daemon uptime
    pub fn set_uptime(&self, seconds: f64) {
        self.uptime_seconds
            .with_label_values::<&str>(&[])
            .set(seconds);
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

    // ── Watch error tracking helpers (Task 461.12) ────────────────────

    /// Record a watch error
    pub fn watch_error(&self, watch_id: &str) {
        self.watch_errors_total.with_label_values(&[watch_id]).inc();
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
    pub fn set_watch_health_status(
        &self,
        watch_id: &str,
        old_status: Option<&str>,
        new_status: &str,
    ) {
        if let Some(old) = old_status {
            self.watch_health_status
                .with_label_values(&[watch_id, old])
                .set(0);
        }
        self.watch_health_status
            .with_label_values(&[watch_id, new_status])
            .set(1);
    }

    /// Set the total number of watches currently in backoff
    pub fn set_watches_in_backoff(&self, count: i64) {
        self.watches_in_backoff
            .with_label_values::<&str>(&[])
            .set(count);
    }

    /// Increment watches in backoff count
    pub fn inc_watches_in_backoff(&self) {
        self.watches_in_backoff.with_label_values::<&str>(&[]).inc();
    }

    /// Decrement watches in backoff count
    pub fn dec_watches_in_backoff(&self) {
        self.watches_in_backoff.with_label_values::<&str>(&[]).dec();
    }

    /// Record watch recovery time (from first error to recovery)
    pub fn watch_recovered(&self, watch_id: &str, recovery_time_secs: f64) {
        self.watch_recovery_time_seconds
            .with_label_values(&[watch_id])
            .observe(recovery_time_secs);
        self.set_watch_consecutive_errors(watch_id, 0);
    }

    /// Record a throttled event due to queue depth
    pub fn watch_event_throttled(&self, watch_id: &str, load_level: &str) {
        self.watch_events_throttled_total
            .with_label_values(&[watch_id, load_level])
            .inc();
    }

    // ── Unified Queue helpers (Task 37.35) ────────────────────────────

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
            .with_label_values::<&str>(&[])
            .set(count);
    }

    /// Record a retry for unified queue item
    pub fn unified_queue_retry(&self, item_type: &str) {
        self.unified_queue_retries_total
            .with_label_values(&[item_type])
            .inc();
    }

    // ── Dimensional processing metrics (A2) ───────────────────────────

    /// Enable or disable the A2 dimensional-processing emission path
    /// (telemetry kill switch, from `observability.metrics.enabled`).
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the A2 dimensional-processing emission path is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
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

    // ── RED/USE coverage (B6) ─────────────────────────────────────────

    /// Record one completed daemon vector search: latency (by collection +
    /// mode) and result-set size (by tenant + collection).
    pub fn record_search(
        &self,
        collection: &str,
        mode: &str,
        tenant_id: &str,
        result_count: usize,
        duration: std::time::Duration,
    ) {
        self.search_duration_seconds
            .with_label_values(&[collection, mode])
            .observe(duration.as_secs_f64());
        self.search_result_count
            .with_label_values(&[tenant_id, collection])
            .observe(result_count as f64);
    }

    /// RAII guard tracking one in-flight embedding operation
    /// (`wqm_memexd_embedding_inflight`). Increments on construction and
    /// decrements on drop, so the gauge is correct even on early return/panic.
    pub fn embedding_inflight_guard(&self) -> EmbeddingInflightGuard<'_> {
        self.embedding_inflight.inc();
        EmbeddingInflightGuard {
            gauge: &self.embedding_inflight,
        }
    }

    /// Increment the SQLite busy/locked (`SQLITE_BUSY`) saturation counter.
    pub fn record_sqlite_busy(&self) {
        self.sqlite_busy_total.inc();
    }

    /// Observe one processed item on the dimensional
    /// `wqm_memexd_processing_duration_seconds` histogram.
    ///
    /// `file_path` (when present) derives the bounded `file_type`; `language`
    /// (the detected language name, when known) derives the bounded `language`
    /// label. Both pass through the A1 cardinality cap, so each contributes at
    /// most `N + 1` distinct label values. `operation` is the queue-op enum
    /// string; `embedding_engine` is the bounded provider label. When telemetry
    /// is disabled ([`is_enabled`](Self::is_enabled) is false) this is a no-op
    /// and creates no series (zero-overhead path).
    pub fn record_processing_item(
        &self,
        collection: &str,
        file_path: Option<&Path>,
        language: Option<&str>,
        operation: &str,
        embedding_engine: &str,
        duration_secs: f64,
    ) {
        if !self.is_enabled() {
            return;
        }
        let file_type = file_path.map(bounded_file_type).unwrap_or(OTHER);
        let language = language.map(bounded_language).unwrap_or(OTHER);
        self.processing_duration_seconds
            .with_label_values(&[collection, file_type, language, operation, embedding_engine])
            .observe(duration_secs);
    }
}

/// RAII guard for `wqm_memexd_embedding_inflight`: holds the gauge incremented
/// for its lifetime and decrements it on drop. Obtain via
/// [`DaemonMetrics::embedding_inflight_guard`].
pub struct EmbeddingInflightGuard<'a> {
    gauge: &'a prometheus::IntGauge,
}

impl Drop for EmbeddingInflightGuard<'_> {
    fn drop(&mut self) {
        self.gauge.dec();
    }
}
