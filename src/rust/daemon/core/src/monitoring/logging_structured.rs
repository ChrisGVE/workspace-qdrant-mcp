//! Multi-tenant structured logging functions and error monitor
//!
//! Provides structured logging contexts (SessionContext, QueueContext, SearchContext)
//! and specialized logging functions for session lifecycle, queue operations,
//! search queries, and error tracking.

use std::collections::HashMap;
use tracing::{error, info, warn, debug, instrument};

use crate::error::{WorkspaceError, ErrorSeverity, ErrorMonitor};
use super::logging_perf::{record_operation_metric, record_error_metric};

/// Error tracking with context information
pub fn log_error_with_context(error: &WorkspaceError, context: &str) {
    let error_dict = error.to_dict();

    match error.severity() {
        ErrorSeverity::Low => {
            info!(
                error_category = error.category(),
                error_context = context,
                error_details = ?error_dict,
                "Low severity error: {}", error
            );
        }
        ErrorSeverity::Medium => {
            warn!(
                error_category = error.category(),
                error_context = context,
                error_details = ?error_dict,
                "Medium severity error: {}", error
            );
        }
        ErrorSeverity::High => {
            error!(
                error_category = error.category(),
                error_context = context,
                error_details = ?error_dict,
                "High severity error: {}", error
            );
        }
        ErrorSeverity::Critical => {
            error!(
                error_category = error.category(),
                error_context = context,
                error_details = ?error_dict,
                "Critical error: {}", error
            );
        }
    }

    record_error_metric(error.category());
}

/// Structured logging for configuration events
#[instrument(skip(config))]
pub fn log_configuration_event(event_type: &str, config: &serde_json::Value) {
    info!(event_type = event_type, config = %config, "Configuration event");
}

/// Structured logging for security events
#[instrument]
pub fn log_security_event(
    event_type: &str,
    severity: &str,
    details: &HashMap<String, String>,
) {
    warn!(
        event_type = event_type,
        severity = severity,
        details = ?details,
        "Security event detected"
    );
}

/// Structured logging for performance events
#[instrument]
pub fn log_performance_event(
    operation: &str,
    duration_ms: f64,
    throughput: Option<f64>,
    resource_usage: Option<&HashMap<String, f64>>,
) {
    if duration_ms > 1000.0 {
        warn!(
            operation = operation,
            duration_ms = duration_ms,
            throughput = throughput,
            resource_usage = ?resource_usage,
            "Slow operation detected"
        );
    } else {
        debug!(
            operation = operation,
            duration_ms = duration_ms,
            throughput = throughput,
            resource_usage = ?resource_usage,
            "Performance event"
        );
    }
}

/// Health check logging
#[instrument]
pub fn log_health_check_result(
    component: &str,
    status: &str,
    response_time_ms: f64,
    details: Option<&HashMap<String, String>>,
) {
    match status {
        "healthy" => {
            debug!(component, status, response_time_ms, details = ?details, "Health check passed");
        }
        "degraded" => {
            warn!(component, status, response_time_ms, details = ?details, "Health check degraded");
        }
        "unhealthy" => {
            error!(component, status, response_time_ms, details = ?details, "Health check failed");
        }
        _ => {
            info!(component, status, response_time_ms, details = ?details, "Health check completed");
        }
    }
}

// ── Multi-tenant Structured Logging Functions (Task 412.8) ────────────

/// Session lifecycle events logging context
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub project_id: String,
    pub session_id: Option<String>,
    pub tenant_id: Option<String>,
    pub priority: Option<String>,
}

impl SessionContext {
    pub fn new(project_id: impl Into<String>) -> Self {
        Self {
            project_id: project_id.into(),
            session_id: None,
            tenant_id: None,
            priority: None,
        }
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn with_priority(mut self, priority: impl Into<String>) -> Self {
        self.priority = Some(priority.into());
        self
    }
}

/// Log session registration event
pub fn log_session_register(ctx: &SessionContext, source: &str) {
    info!(
        project_id = %ctx.project_id,
        session_id = ctx.session_id.as_deref().unwrap_or("unknown"),
        tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
        priority = ctx.priority.as_deref().unwrap_or("NORMAL"),
        source = %source,
        "Session registered"
    );
}

/// Log session heartbeat event
pub fn log_session_heartbeat(ctx: &SessionContext, latency_ms: f64) {
    debug!(
        project_id = %ctx.project_id,
        session_id = ctx.session_id.as_deref().unwrap_or("unknown"),
        latency_ms = %latency_ms,
        "Session heartbeat"
    );

    if latency_ms > 100.0 {
        warn!(
            project_id = %ctx.project_id,
            session_id = ctx.session_id.as_deref().unwrap_or("unknown"),
            latency_ms = %latency_ms,
            "Slow heartbeat detected"
        );
    }
}

/// Log session deprioritization event
pub fn log_session_deprioritize(ctx: &SessionContext, reason: &str) {
    info!(
        project_id = %ctx.project_id,
        session_id = ctx.session_id.as_deref().unwrap_or("unknown"),
        tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
        reason = %reason,
        "Session deprioritized"
    );
}

/// Log session cleanup (orphaned sessions)
pub fn log_session_cleanup(ctx: &SessionContext, last_heartbeat_secs: f64) {
    warn!(
        project_id = %ctx.project_id,
        session_id = ctx.session_id.as_deref().unwrap_or("unknown"),
        last_heartbeat_secs = %last_heartbeat_secs,
        "Orphaned session cleaned up"
    );
}

/// Log priority change event
pub fn log_priority_change(
    ctx: &SessionContext,
    old_priority: &str,
    new_priority: &str,
    reason: &str,
) {
    info!(
        project_id = %ctx.project_id,
        session_id = ctx.session_id.as_deref().unwrap_or("unknown"),
        old_priority = %old_priority,
        new_priority = %new_priority,
        reason = %reason,
        "Session priority changed"
    );
}

/// Queue operation logging context
#[derive(Debug, Clone)]
pub struct QueueContext {
    pub collection: String,
    pub priority: String,
    pub tenant_id: Option<String>,
}

impl QueueContext {
    pub fn new(collection: impl Into<String>, priority: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            priority: priority.into(),
            tenant_id: None,
        }
    }

    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }
}

/// Log queue depth change
pub fn log_queue_depth_change(ctx: &QueueContext, old_depth: i64, new_depth: i64) {
    debug!(
        collection = %ctx.collection,
        priority = %ctx.priority,
        tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
        old_depth = %old_depth,
        new_depth = %new_depth,
        "Queue depth changed"
    );

    if new_depth > 1000 {
        warn!(
            collection = %ctx.collection,
            priority = %ctx.priority,
            depth = %new_depth,
            "High queue depth detected (>1000 items)"
        );
    }
}

/// Log queue item enqueue
pub fn log_queue_enqueue(ctx: &QueueContext, document_path: &str, document_id: &str) {
    debug!(
        collection = %ctx.collection,
        priority = %ctx.priority,
        tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
        document_path = %document_path,
        document_id = %document_id,
        "Document enqueued"
    );
}

/// Log queue item processed
pub fn log_queue_processed(
    ctx: &QueueContext,
    document_id: &str,
    processing_time_ms: f64,
    status: &str,
) {
    if status == "success" {
        debug!(
            collection = %ctx.collection,
            priority = %ctx.priority,
            tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
            document_id = %document_id,
            processing_time_ms = %processing_time_ms,
            status = %status,
            "Document processed"
        );
    } else {
        warn!(
            collection = %ctx.collection,
            priority = %ctx.priority,
            tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
            document_id = %document_id,
            processing_time_ms = %processing_time_ms,
            status = %status,
            "Document processing failed"
        );
    }

    if processing_time_ms > 10_000.0 {
        warn!(
            collection = %ctx.collection,
            document_id = %document_id,
            processing_time_ms = %processing_time_ms,
            "Slow document processing (>10s)"
        );
    }
}

/// Search query logging context
#[derive(Debug, Clone)]
pub struct SearchContext {
    pub collection: String,
    pub tenant_id: Option<String>,
    pub query_type: String,
    pub limit: usize,
}

impl SearchContext {
    pub fn new(collection: impl Into<String>, query_type: impl Into<String>, limit: usize) -> Self {
        Self {
            collection: collection.into(),
            tenant_id: None,
            query_type: query_type.into(),
            limit,
        }
    }

    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }
}

/// Log slow query (>1s)
pub fn log_slow_query(
    ctx: &SearchContext,
    query_text: &str,
    duration_ms: f64,
    result_count: usize,
) {
    if duration_ms > 1000.0 {
        warn!(
            collection = %ctx.collection,
            tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
            query_type = %ctx.query_type,
            query_text = %query_text,
            duration_ms = %duration_ms,
            result_count = %result_count,
            limit = %ctx.limit,
            "Slow query detected (>1s)"
        );
    }
}

/// Log search request
pub fn log_search_request(ctx: &SearchContext, query_text: &str) {
    debug!(
        collection = %ctx.collection,
        tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
        query_type = %ctx.query_type,
        query_preview = query_text.chars().take(50).collect::<String>(),
        limit = %ctx.limit,
        "Search request"
    );
}

/// Log search result
pub fn log_search_result(
    ctx: &SearchContext,
    duration_ms: f64,
    result_count: usize,
) {
    debug!(
        collection = %ctx.collection,
        tenant_id = ctx.tenant_id.as_deref().unwrap_or(""),
        query_type = %ctx.query_type,
        duration_ms = %duration_ms,
        result_count = %result_count,
        "Search completed"
    );
}

/// Log ingestion error with context
pub fn log_ingestion_error(
    collection: &str,
    tenant_id: Option<&str>,
    document_path: &str,
    error_type: &str,
    error_msg: &str,
) {
    error!(
        collection = %collection,
        tenant_id = tenant_id.unwrap_or(""),
        document_path = %document_path,
        error_type = %error_type,
        error_msg = %error_msg,
        "Ingestion error"
    );
}

/// Integration with error recovery system
pub struct LoggingErrorMonitor;

impl ErrorMonitor for LoggingErrorMonitor {
    fn report_error(&self, error: &WorkspaceError, context: Option<&str>) {
        log_error_with_context(error, context.unwrap_or("unknown"));
    }

    fn report_recovery(&self, error_category: &str, attempt: u32) {
        info!(
            error_category = error_category,
            attempt = attempt,
            "Error recovery succeeded"
        );
        record_operation_metric(&format!("recovery_{}", error_category), 0.0);
    }

    fn report_circuit_breaker_state(&self, name: &str, state: &str) {
        match state {
            "open" => {
                warn!(circuit_breaker = name, state = state, "Circuit breaker opened");
                record_error_metric("circuit_breaker_open");
            }
            "closed" => {
                info!(circuit_breaker = name, state = state, "Circuit breaker closed");
            }
            "half-open" => {
                info!(circuit_breaker = name, state = state, "Circuit breaker half-open");
            }
            _ => {
                warn!(circuit_breaker = name, state = state, "Unknown circuit breaker state");
            }
        }
    }

    fn get_error_stats(&self) -> crate::error::ErrorStats {
        crate::error::ErrorStats::default()
    }
}
