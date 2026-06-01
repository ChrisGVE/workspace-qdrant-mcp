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
        self.watches_in_backoff.with_label_values(&[]).set(count);
    }

    /// Increment watches in backoff count
    pub fn inc_watches_in_backoff(&self) {
        self.watches_in_backoff.with_label_values(&[]).inc();
    }

    /// Decrement watches in backoff count
    pub fn dec_watches_in_backoff(&self) {
        self.watches_in_backoff.with_label_values(&[]).dec();
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
            .with_label_values(&[])
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
