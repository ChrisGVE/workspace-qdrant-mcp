//! Monitoring subsystem
//!
//! Consolidates logging, metrics, alerting, metrics history,
//! remote monitoring, and tool monitoring into a single module namespace.

pub mod logging_config;
pub mod logging_structured;
pub mod metrics_core;
pub mod metrics_server;
pub mod metrics_alerts;
pub mod metrics_history;
pub mod remote_monitor;
pub mod tool_monitor;

#[cfg(test)]
mod logging_tests;
#[cfg(test)]
mod metrics_tests;

// ── Backward-compatible re-exports ────────────────────────────────────

// Logging re-exports (previously `crate::logging::*`)
pub use logging_config::{
    LoggingConfig, PerformanceMetrics, initialize_logging,
    initialize_daemon_silence, track_async_operation,
    suppress_tty_debug_output, get_canonical_log_dir,
    record_operation_metric, record_error_metric,
    get_performance_metrics,
};
pub use logging_structured::{
    log_error_with_context, LoggingErrorMonitor,
    // Multi-tenant structured logging contexts (Task 412.8)
    SessionContext, QueueContext, SearchContext,
    log_session_register, log_session_heartbeat, log_session_deprioritize,
    log_session_cleanup, log_priority_change, log_queue_depth_change,
    log_queue_enqueue, log_queue_processed, log_slow_query,
    log_search_request, log_search_result, log_ingestion_error,
    // Additional structured logging functions
    log_configuration_event, log_security_event, log_performance_event,
    log_health_check_result,
};

// Metrics re-exports (previously `crate::metrics::*`)
pub use metrics_core::{DaemonMetrics, METRICS};
pub use metrics_server::{MetricsServer, MetricsSnapshot};
pub use metrics_alerts::{
    Alert, AlertChecker, AlertConfig, AlertSeverity, AlertType,
    create_orphaned_session_alert, create_slow_search_alert,
};

// Metrics history re-exports (previously `crate::metrics_history::*`)
pub use metrics_history::{
    write_metrics_batch, write_snapshot, query_metrics, query_aggregated,
    cleanup_old_metrics, get_available_metrics,
    run_aggregation, aggregate_hourly, aggregate_daily, aggregate_weekly,
    apply_retention, run_maintenance, run_maintenance_now,
    MetricsHistoryError, MetricsHistoryResult, MetricsHistoryQuery,
    AggregatedMetric, RetentionConfig,
};

// Remote monitor re-exports (previously `crate::remote_monitor::*`)
pub use remote_monitor::{
    check_remote_url_changes, check_git_state_changes,
    RemoteCheckResult, GitStateCheckResult,
};

// Tool monitor re-exports (previously `crate::tool_monitor::*`)
pub use tool_monitor::{
    ToolMonitor, MonitoringError, MonitoringResult, RequeueStats,
};
