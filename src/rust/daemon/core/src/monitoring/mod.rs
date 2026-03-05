//! Monitoring subsystem
//!
//! Consolidates logging, metrics, alerting, metrics history,
//! remote monitoring, and tool monitoring into a single module namespace.

pub mod logging_config;
pub mod logging_perf;
pub mod logging_structured;
pub mod metrics_aggregation;
pub mod metrics_alerts;
pub mod metrics_core;
mod metrics_helpers; // impl blocks for DaemonMetrics helper methods
pub mod metrics_history;
pub mod metrics_server;
pub mod remote_monitor;
pub mod tool_detection;
pub mod tool_monitor;

#[cfg(test)]
mod logging_tests;
#[cfg(test)]
mod metrics_history_tests;
#[cfg(test)]
mod metrics_tests;

// ── Backward-compatible re-exports ────────────────────────────────────

// Logging re-exports (previously `crate::logging::*`)
pub use logging_config::{
    get_canonical_log_dir, initialize_daemon_silence, initialize_logging,
    suppress_tty_debug_output, LoggingConfig,
};
pub use logging_perf::{
    get_performance_metrics, record_error_metric, record_operation_metric, track_async_operation,
    PerformanceMetrics,
};
pub use logging_structured::{
    // Additional structured logging functions
    log_configuration_event,
    log_error_with_context,
    log_health_check_result,
    log_ingestion_error,
    log_performance_event,
    log_priority_change,
    log_queue_depth_change,
    log_queue_enqueue,
    log_queue_processed,
    log_search_request,
    log_search_result,
    log_security_event,
    log_session_cleanup,
    log_session_deprioritize,
    log_session_heartbeat,
    log_session_register,
    log_slow_query,
    LoggingErrorMonitor,
    QueueContext,
    SearchContext,
    // Multi-tenant structured logging contexts (Task 412.8)
    SessionContext,
};

// Metrics re-exports (previously `crate::metrics::*`)
pub use metrics_alerts::{
    create_orphaned_session_alert, create_slow_search_alert, Alert, AlertChecker, AlertConfig,
    AlertSeverity, AlertType,
};
pub use metrics_core::{DaemonMetrics, METRICS};
pub use metrics_server::{MetricsServer, MetricsSnapshot};

// Metrics history re-exports (previously `crate::metrics_history::*`)
pub use metrics_aggregation::{
    aggregate_daily, aggregate_hourly, aggregate_weekly, apply_retention, run_aggregation,
    run_maintenance, run_maintenance_now, RetentionConfig,
};
pub use metrics_history::{
    cleanup_old_metrics, get_available_metrics, query_aggregated, query_metrics,
    write_metrics_batch, write_snapshot, AggregatedMetric, MetricsHistoryError,
    MetricsHistoryQuery, MetricsHistoryResult,
};

// Remote monitor re-exports (previously `crate::remote_monitor::*`)
pub use remote_monitor::{
    check_git_state_changes, check_remote_url_changes, GitStateCheckResult, RemoteCheckResult,
};

// Tool monitor re-exports (previously `crate::tool_monitor::*`)
pub use tool_monitor::{MonitoringError, MonitoringResult, RequeueStats, ToolMonitor};
