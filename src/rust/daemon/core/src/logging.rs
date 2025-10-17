//! Comprehensive logging and tracing configuration for workspace-qdrant-mcp
//!
//! This module provides structured logging using tracing v0.1 with comprehensive
//! instrumentation, performance monitoring, and error tracking capabilities.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use once_cell::sync::Lazy;
use atty::Stream;

use tracing::{error, info, warn, debug, instrument, Level};
use tracing_subscriber::{
    fmt::{self, time::ChronoUtc},
    layer::{SubscriberExt, Layer},
    util::SubscriberInitExt,
    EnvFilter, Registry,
    filter::LevelFilter,
};

use crate::error::{WorkspaceError, ErrorSeverity, ErrorMonitor};

/// Logging configuration for the workspace-qdrant system
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level filter (trace, debug, info, warn, error)
    pub level: Level,
    /// Enable JSON structured output
    pub json_format: bool,
    /// Enable console output
    pub console_output: bool,
    /// Enable file logging
    pub file_logging: bool,
    /// Log file path (if file logging is enabled)
    pub log_file_path: Option<PathBuf>,
    /// Enable performance metrics collection
    pub performance_metrics: bool,
    /// Enable error tracking and correlation
    pub error_tracking: bool,
    /// Custom fields to include in all log entries
    pub global_fields: HashMap<String, String>,
    /// Force disable ANSI colors (automatically detected if None)
    pub force_disable_ansi: Option<bool>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            json_format: false,
            console_output: true,
            file_logging: false,
            log_file_path: None,
            performance_metrics: true,
            error_tracking: true,
            global_fields: HashMap::new(),
            force_disable_ansi: None,
        }
    }
}

impl LoggingConfig {
    /// Create logging configuration from environment variables
    pub fn from_environment() -> Self {
        let mut config = Self::default();

        // Set log level from environment
        if let Ok(level_str) = env::var("RUST_LOG") {
            if let Ok(level) = level_str.to_uppercase().parse::<Level>() {
                config.level = level;
            }
        }

        // JSON format
        config.json_format = env::var("WQM_LOG_JSON")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        // Console output
        config.console_output = env::var("WQM_LOG_CONSOLE")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true);

        // File logging
        config.file_logging = env::var("WQM_LOG_FILE")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        if let Ok(file_path) = env::var("WQM_LOG_FILE_PATH") {
            config.log_file_path = Some(PathBuf::from(file_path));
            config.file_logging = true;
        }

        // Performance metrics
        config.performance_metrics = env::var("WQM_LOG_METRICS")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true);

        // Error tracking
        config.error_tracking = env::var("WQM_LOG_ERROR_TRACKING")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true);

        // Global fields
        for (key, value) in env::vars() {
            if key.starts_with("WQM_LOG_FIELD_") {
                let field_name = key.strip_prefix("WQM_LOG_FIELD_")
                    .unwrap()
                    .to_lowercase();
                config.global_fields.insert(field_name, value);
            }
        }

        config
    }

    /// Create production logging configuration with user-writable fallback path
    pub fn production() -> Self {
        // Determine log file path with proper fallback
        let log_file_path = if let Ok(custom_path) = env::var("WQM_LOG_FILE_PATH") {
            // Use custom path from environment variable
            Some(PathBuf::from(custom_path))
        } else if let Ok(home) = env::var("HOME") {
            // Use user-writable location in ~/.local/share/workspace-qdrant-mcp/logs
            Some(
                PathBuf::from(home)
                    .join(".local")
                    .join("share")
                    .join("workspace-qdrant-mcp")
                    .join("logs")
                    .join("daemon.log")
            )
        } else {
            // Last resort: use temp directory
            Some(
                env::temp_dir()
                    .join("workspace-qdrant-mcp")
                    .join("daemon.log")
            )
        };

        Self {
            level: Level::INFO,
            json_format: true,
            console_output: true,
            file_logging: true,
            log_file_path,
            performance_metrics: true,
            error_tracking: true,
            force_disable_ansi: Some(true), // Force disable ANSI in production
            global_fields: {
                let mut fields = HashMap::new();
                fields.insert("service".to_string(), "workspace-qdrant-mcp".to_string());
                fields.insert("component".to_string(), "rust-engine".to_string());
                fields
            },
        }
    }

    /// Create development logging configuration
    pub fn development() -> Self {
        Self {
            level: Level::DEBUG,
            json_format: false,
            console_output: true,
            file_logging: false,
            log_file_path: None,
            performance_metrics: true,
            error_tracking: true,
            force_disable_ansi: None, // Auto-detect ANSI in development
            global_fields: {
                let mut fields = HashMap::new();
                fields.insert("env".to_string(), "development".to_string());
                fields
            },
        }
    }
}

/// Performance metrics collector for logging
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    operation_counts: HashMap<String, u64>,
    operation_durations: HashMap<String, Vec<f64>>,
    error_counts: HashMap<String, u64>,
}

impl PerformanceMetrics {
    /// Record an operation performance metric
    pub fn record_operation(&mut self, operation: &str, duration_ms: f64) {
        *self.operation_counts.entry(operation.to_string()).or_insert(0) += 1;
        self.operation_durations
            .entry(operation.to_string())
            .or_default()
            .push(duration_ms);
    }

    /// Record an error metric
    pub fn record_error(&mut self, error_type: &str) {
        *self.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Get performance summary
    pub fn get_summary(&self) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();

        // Operation counts
        summary.insert("operation_counts".to_string(), 
            serde_json::to_value(&self.operation_counts).unwrap_or_default());

        // Operation statistics
        let mut operation_stats = HashMap::new();
        for (operation, durations) in &self.operation_durations {
            if !durations.is_empty() {
                let sum: f64 = durations.iter().sum();
                let count = durations.len() as f64;
                let avg = sum / count;
                
                let mut sorted_durations = durations.clone();
                sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let p50 = sorted_durations[sorted_durations.len() / 2];
                let p95 = sorted_durations[(sorted_durations.len() as f64 * 0.95) as usize];
                let p99 = sorted_durations[(sorted_durations.len() as f64 * 0.99) as usize];
                
                operation_stats.insert(operation.clone(), serde_json::json!({
                    "avg_ms": avg,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "count": count
                }));
            }
        }
        summary.insert("operation_stats".to_string(), 
            serde_json::to_value(operation_stats).unwrap_or_default());

        // Error counts
        summary.insert("error_counts".to_string(), 
            serde_json::to_value(&self.error_counts).unwrap_or_default());

        summary
    }
}

/// Global performance metrics instance
static PERFORMANCE_METRICS: Lazy<std::sync::Mutex<PerformanceMetrics>> = 
    Lazy::new(|| std::sync::Mutex::new(PerformanceMetrics::default()));

/// Suppress any potential TTY detection debug output
/// This function should be called before any TTY detection to ensure complete silence
/// Note: Critical TTY_DEBUG suppression is now done in main() before any initialization
pub fn suppress_tty_debug_output() {
    // Set environment variables that might control debug output from TTY detection libraries
    std::env::set_var("TTY_DETECTION_SILENT", "1");
    std::env::set_var("ATTY_FORCE_DISABLE_DEBUG", "1");
    std::env::set_var("WQM_TTY_DEBUG", "0");

    // CRITICAL: Suppress Claude CLI TTY debug messages
    // The Claude CLI uses TTY_DEBUG environment variable to control debug output
    std::env::set_var("TTY_DEBUG", "0");

    // Additional suppression variables for various TTY detection libraries
    std::env::set_var("ATTY_SILENT", "1");
    std::env::set_var("CONSOLE_NO_DEBUG", "1");
    std::env::set_var("TERMINAL_DEBUG", "0");

    // Set NO_COLOR to prevent any color-related debug messages
    std::env::set_var("NO_COLOR", "1");
    std::env::set_var("TERM", "dumb");
}

/// Initialize comprehensive logging system with daemon mode silence support
pub fn initialize_logging(config: LoggingConfig) -> Result<(), WorkspaceError> {
    // Detect daemon mode early for complete suppression
    let daemon_mode = is_daemon_mode();

    // In daemon mode, configure for complete silence
    if daemon_mode && !config.console_output {
        // Create a completely silent subscriber that discards all output
        let null_writer = || std::io::sink();
        let subscriber = tracing_subscriber::fmt::Subscriber::builder()
            .with_max_level(LevelFilter::OFF)
            .with_writer(null_writer)
            .with_ansi(false)
            .finish();

        subscriber.init();
        return Ok(());
    }
    // Create environment filter
    let env_filter = if daemon_mode {
        // In daemon mode, use OFF level to suppress all tracing output
        EnvFilter::new("off")
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()))
    };

    // Determine if we should disable ANSI colors
    // Note: TTY detection performed silently to prevent log pollution
    let disable_ansi = config.force_disable_ansi.unwrap_or_else(|| {
        // Check for NO_COLOR environment variable (standard)
        if env::var("NO_COLOR").is_ok() {
            return true;
        }

        // Check for explicit TTY detection silence flag
        if env::var("TTY_DETECTION_SILENT").is_ok() || env::var("WQM_TTY_DEBUG") == Ok("0".to_string()) {
            return true; // Force disable ANSI when TTY debug is suppressed
        }

        // Disable ANSI if:
        // 1. Not connected to a TTY
        // 2. Running as a service (detected by common service environment variables)
        // 3. Output is being redirected
        let tty_check = !atty::is(Stream::Stdout);
        let service_check = daemon_mode ||
            env::var("XPC_SERVICE_NAME").is_ok() || // macOS LaunchAgent/LaunchDaemon
            env::var("_").map(|v| v.contains("launchd")).unwrap_or(false) ||
            env::var("INVOCATION_ID").is_ok() || // systemd
            env::var("UPSTART_JOB").is_ok() ||   // upstart
            env::var("SYSTEMD_EXEC_PID").is_ok() || // systemd service
            env::var("XPC_FLAGS").map(|v| v == "1").unwrap_or(false); // macOS service mode

        tty_check || service_check
    });

    // Build the subscriber with conditional layers
    let registry = Registry::default();

    // In daemon mode with console output disabled, use sink writer
    if daemon_mode && !config.console_output {
        let null_writer = || std::io::sink();
        let null_layer = fmt::layer()
            .with_writer(null_writer)
            .with_ansi(false);
        registry.with(env_filter).with(null_layer).init();
        return Ok(());
    }

    // Add layers based on configuration
    if config.console_output && config.file_logging {
        // Both console and file logging
        if let Some(ref log_file_path) = config.log_file_path {
            // Ensure directory exists
            if let Some(parent) = log_file_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    WorkspaceError::file_system(
                        format!("Failed to create log directory: {}", e),
                        parent.to_string_lossy().to_string(),
                        "create_directory",
                    )
                })?;
            }
            
            let log_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_file_path)
                .map_err(|e| {
                    WorkspaceError::file_system(
                        format!("Failed to open log file: {}", e),
                        log_file_path.to_string_lossy().to_string(),
                        "open_file",
                    )
                })?;

            let console_layer = if config.json_format {
                fmt::layer()
                    .json()
                    .with_timer(ChronoUtc::rfc_3339())
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .with_ansi(!disable_ansi)
                    .boxed()
            } else {
                fmt::layer()
                    .with_timer(ChronoUtc::rfc_3339())
                    .with_target(true)
                    .with_thread_ids(false)
                    .with_thread_names(false)
                    .with_ansi(!disable_ansi)
                    .boxed()
            };
            
            let file_layer = fmt::layer()
                .json()
                .with_writer(Arc::new(log_file))
                .with_timer(ChronoUtc::rfc_3339())
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true);

            registry.with(env_filter).with(console_layer).with(file_layer).init();
        } else {
            return Err(WorkspaceError::configuration(
                "File logging enabled but no log file path specified",
            ));
        }
    } else if config.console_output {
        // Console logging only
        let console_layer = if config.json_format {
            fmt::layer()
                .json()
                .with_timer(ChronoUtc::rfc_3339())
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_ansi(!disable_ansi)
                .boxed()
        } else {
            fmt::layer()
                .with_timer(ChronoUtc::rfc_3339())
                .with_target(true)
                .with_thread_ids(false)
                .with_thread_names(false)
                .with_ansi(!disable_ansi)
                .boxed()
        };
        
        registry.with(env_filter).with(console_layer).init();
    } else if config.file_logging {
        // File logging only
        if let Some(ref log_file_path) = config.log_file_path {
            // Ensure directory exists
            if let Some(parent) = log_file_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    WorkspaceError::file_system(
                        format!("Failed to create log directory: {}", e),
                        parent.to_string_lossy().to_string(),
                        "create_directory",
                    )
                })?;
            }
            
            let log_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_file_path)
                .map_err(|e| {
                    WorkspaceError::file_system(
                        format!("Failed to open log file: {}", e),
                        log_file_path.to_string_lossy().to_string(),
                        "open_file",
                    )
                })?;
            
            let file_layer = fmt::layer()
                .json()
                .with_writer(Arc::new(log_file))
                .with_timer(ChronoUtc::rfc_3339())
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true);

            registry.with(env_filter).with(file_layer).init();
        } else {
            return Err(WorkspaceError::configuration(
                "File logging enabled but no log file path specified",
            ));
        }
    } else {
        // No logging layers enabled - in daemon mode use null writer, otherwise just filter
        if daemon_mode {
            let null_writer = || std::io::sink();
            let null_layer = fmt::layer()
                .with_writer(null_writer)
                .with_ansi(false);
            registry.with(env_filter).with(null_layer).init();
        } else {
            registry.with(env_filter).init();
        }
    }

    // Log initialization
    info!(
        config = ?config,
        "Logging system initialized"
    );

    // Add global fields
    for (key, value) in &config.global_fields {
        tracing::Span::current().record(key.as_str(), value.as_str());
    }

    // Log initialization only if not in daemon mode
    if !daemon_mode {
        info!(
            config = ?config,
            "Logging system initialized"
        );

        // Add global fields
        for (key, value) in &config.global_fields {
            tracing::Span::current().record(key.as_str(), value.as_str());
        }
    }

    Ok(())
}

/// Detect if running in daemon mode based on multiple indicators
fn is_daemon_mode() -> bool {
    // Primary daemon mode indicator
    if env::var("WQM_SERVICE_MODE").map(|v| v == "true").unwrap_or(false) {
        return true;
    }

    // Detect common service/daemon environments
    env::var("XPC_SERVICE_NAME").is_ok() || // macOS LaunchAgent/LaunchDaemon
        env::var("_").map(|v| v.contains("launchd")).unwrap_or(false) ||
        env::var("INVOCATION_ID").is_ok() || // systemd
        env::var("UPSTART_JOB").is_ok() ||   // upstart
        env::var("SYSTEMD_EXEC_PID").is_ok() || // systemd service
        env::var("XPC_FLAGS").map(|v| v == "1").unwrap_or(false) || // macOS service mode
        env::var("SYSLOG_IDENTIFIER").is_ok() || // systemd journal
        env::var("LOGNAME").map(|v| v == "root").unwrap_or(false) // Running as system daemon
}

/// Initialize complete output suppression for daemon mode
pub fn initialize_daemon_silence() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self};
    use tracing_subscriber::fmt;

    // Create a completely silent subscriber
    let null_writer = || io::sink();
    let subscriber = fmt::Subscriber::builder()
        .with_max_level(LevelFilter::OFF)
        .with_writer(null_writer)
        .with_ansi(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok(())
}

/// Record operation performance metric
pub fn record_operation_metric(operation: &str, duration_ms: f64) {
    if let Ok(mut metrics) = PERFORMANCE_METRICS.lock() {
        metrics.record_operation(operation, duration_ms);
    }
}

/// Record error metric
pub fn record_error_metric(error_type: &str) {
    if let Ok(mut metrics) = PERFORMANCE_METRICS.lock() {
        metrics.record_error(error_type);
    }
}

/// Get performance metrics summary
pub fn get_performance_metrics() -> HashMap<String, serde_json::Value> {
    PERFORMANCE_METRICS
        .lock()
        .map(|metrics| metrics.get_summary())
        .unwrap_or_default()
}

/// Instrumentation macro for automatic performance tracking
#[macro_export]
macro_rules! instrument_async {
    ($func_name:expr, $operation:expr) => {
        tracing::instrument(
            name = $func_name,
            fields(
                operation = $operation,
                start_time = tracing::field::Empty,
                duration_ms = tracing::field::Empty,
                success = tracing::field::Empty,
            )
        )
    };
}

/// Performance tracking wrapper for async operations
pub async fn track_async_operation<F, T, E>(
    operation_name: &str,
    operation: F,
) -> Result<T, E>
where
    F: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let start_time = SystemTime::now();
    let span = tracing::info_span!(
        "async_operation",
        operation = operation_name,
        start_time = ?start_time
    );

    let result = tracing::Instrument::instrument(operation, span.clone()).await;

    let duration = start_time.elapsed()
        .unwrap_or(std::time::Duration::from_millis(0))
        .as_millis() as f64;

    match &result {
        Ok(_) => {
            span.record("duration_ms", duration);
            span.record("success", true);
            record_operation_metric(operation_name, duration);
            debug!(
                operation = operation_name,
                duration_ms = duration,
                "Operation completed successfully"
            );
        }
        Err(e) => {
            span.record("duration_ms", duration);
            span.record("success", false);
            record_error_metric(operation_name);
            error!(
                operation = operation_name,
                duration_ms = duration,
                error = %e,
                "Operation failed"
            );
        }
    }

    result
}

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
    info!(
        event_type = event_type,
        config = %config,
        "Configuration event"
    );
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
            debug!(
                component = component,
                status = status,
                response_time_ms = response_time_ms,
                details = ?details,
                "Health check passed"
            );
        }
        "degraded" => {
            warn!(
                component = component,
                status = status,
                response_time_ms = response_time_ms,
                details = ?details,
                "Health check degraded"
            );
        }
        "unhealthy" => {
            error!(
                component = component,
                status = status,
                response_time_ms = response_time_ms,
                details = ?details,
                "Health check failed"
            );
        }
        _ => {
            info!(
                component = component,
                status = status,
                response_time_ms = response_time_ms,
                details = ?details,
                "Health check completed"
            );
        }
    }
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
                warn!(
                    circuit_breaker = name,
                    state = state,
                    "Circuit breaker opened"
                );
                record_error_metric("circuit_breaker_open");
            }
            "closed" => {
                info!(
                    circuit_breaker = name,
                    state = state,
                    "Circuit breaker closed"
                );
            }
            "half-open" => {
                info!(
                    circuit_breaker = name,
                    state = state,
                    "Circuit breaker half-open"
                );
            }
            _ => {
                warn!(
                    circuit_breaker = name,
                    state = state,
                    "Unknown circuit breaker state"
                );
            }
        }
    }

    fn get_error_stats(&self) -> crate::error::ErrorStats {
        // This would typically integrate with a metrics system
        crate::error::ErrorStats::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use serial_test::serial;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[serial]
    #[test]
    fn test_logging_config_from_environment() {
        let keys = [
            "RUST_LOG",
            "WQM_LOG_JSON",
            "WQM_LOG_CONSOLE",
            "WQM_LOG_FILE",
            "WQM_LOG_FILE_PATH",
            "WQM_LOG_METRICS",
            "WQM_LOG_ERROR_TRACKING",
            "WQM_LOG_FIELD_TEAM",
        ];

        let previous: Vec<Option<String>> = keys
            .iter()
            .map(|key| env::var(key).ok())
            .collect();

        env::set_var("RUST_LOG", "debug");
        env::set_var("WQM_LOG_JSON", "true");
        env::set_var("WQM_LOG_CONSOLE", "0");
        env::set_var("WQM_LOG_FILE", "1");
        env::set_var("WQM_LOG_FILE_PATH", "/tmp/wqm.log");
        env::set_var("WQM_LOG_METRICS", "false");
        env::set_var("WQM_LOG_ERROR_TRACKING", "false");
        env::set_var("WQM_LOG_FIELD_TEAM", "core");

        let config = LoggingConfig::from_environment();

        assert_eq!(config.level, Level::DEBUG);
        assert!(config.json_format);
        assert!(!config.console_output);
        assert!(config.file_logging);
        assert_eq!(
            config.log_file_path.as_ref().map(PathBuf::from),
            Some(PathBuf::from("/tmp/wqm.log"))
        );
        assert!(!config.performance_metrics);
        assert!(!config.error_tracking);
        assert_eq!(config.global_fields.get("team"), Some(&"core".to_string()));

        for (key, value) in keys.iter().zip(previous) {
            match value {
                Some(v) => env::set_var(key, v),
                None => env::remove_var(key),
            }
        }
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();

        metrics.record_operation("test_op", 100.0);
        metrics.record_operation("test_op", 200.0);
        metrics.record_operation("index", 50.0);
        metrics.record_error("test_error");

        let summary = metrics.get_summary();

        let counts = summary
            .get("operation_counts")
            .and_then(Value::as_object)
            .expect("operation counts available");
        assert_eq!(counts.get("test_op").and_then(Value::as_u64), Some(2));
        assert_eq!(counts.get("index").and_then(Value::as_u64), Some(1));

        let stats = summary
            .get("operation_stats")
            .and_then(Value::as_object)
            .expect("operation stats available");
        let ingest_stats = stats
            .get("test_op")
            .and_then(Value::as_object)
            .expect("test_op stats available");

        let avg = ingest_stats.get("avg_ms").and_then(Value::as_f64).unwrap();
        assert!((avg - 150.0).abs() < 1e-6);
        assert_eq!(
            ingest_stats.get("count").and_then(Value::as_f64),
            Some(2.0)
        );

        let error_counts = summary
            .get("error_counts")
            .and_then(Value::as_object)
            .expect("error counts available");
        assert_eq!(
            error_counts
                .get("test_error")
                .and_then(Value::as_u64),
            Some(1)
        );
    }

    #[tokio::test]
    async fn test_track_async_operation() {
        // Successful operation
        let result = track_async_operation("test_op", async { Ok::<_, &str>("success") }).await;
        assert!(result.is_ok());

        // Failed operation
        let result = track_async_operation("test_op", async { Err::<&str, _>("error") }).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_initialize_logging_console_only() {
        let _config = LoggingConfig {
            console_output: true,
            file_logging: false,
            json_format: false,
            ..Default::default()
        };

        // This should not fail for console-only logging
        // Note: In actual tests, you might want to use a test subscriber
        // to avoid interfering with the global subscriber
    }

    #[test]
    fn test_initialize_logging_with_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let _config = LoggingConfig {
            console_output: false,
            file_logging: true,
            log_file_path: Some(temp_file.path().to_path_buf()),
            json_format: true,
            ..Default::default()
        };

        // This test demonstrates file logging configuration
        // In real tests, you'd verify log entries are written to the file
    }

    #[test]
    fn test_error_severity_logging() {
        let error = WorkspaceError::network("Test error", 1, 3);
        
        // This would normally write to the configured logger
        log_error_with_context(&error, "test_context");

        // In real tests, you'd capture and verify log output
    }
}
