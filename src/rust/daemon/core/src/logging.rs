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

use logroller::{LogRollerBuilder, Rotation, RotationSize, Compression};
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
    /// Maximum log file size in MB before rotation (default: 50)
    pub rotation_size_mb: u64,
    /// Number of rotated log files to keep (default: 5)
    pub rotation_count: usize,
    /// Compress rotated log files with gzip (default: true)
    pub compress_rotated: bool,
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
            rotation_size_mb: 50,
            rotation_count: 5,
            compress_rotated: true,
        }
    }
}

/// Returns the canonical OS-specific log directory for workspace-qdrant logs.
///
/// Precedence:
/// 1. `WQM_LOG_DIR` environment variable (explicit override)
/// 2. Platform-specific default:
///    - Linux: `$XDG_STATE_HOME/workspace-qdrant/logs/` (default: `~/.local/state/workspace-qdrant/logs/`)
///    - macOS: `~/Library/Logs/workspace-qdrant/`
///    - Windows: `%LOCALAPPDATA%\workspace-qdrant\logs\`
///
/// Falls back to a temp directory if home cannot be determined.
pub fn get_canonical_log_dir() -> PathBuf {
    // WQM_LOG_DIR takes highest precedence
    if let Ok(custom_dir) = env::var("WQM_LOG_DIR") {
        return PathBuf::from(custom_dir);
    }

    #[cfg(target_os = "linux")]
    {
        env::var("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| env::temp_dir())
                    .join(".local")
                    .join("state")
            })
            .join("workspace-qdrant")
            .join("logs")
    }

    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join("Library")
            .join("Logs")
            .join("workspace-qdrant")
    }

    #[cfg(target_os = "windows")]
    {
        dirs::data_local_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join("workspace-qdrant")
            .join("logs")
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        // Fallback for other platforms
        dirs::home_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join(".workspace-qdrant")
            .join("logs")
    }
}

impl LoggingConfig {
    /// Create logging configuration from environment variables
    ///
    /// Log level precedence: `WQM_LOG_LEVEL` > `RUST_LOG` > default (INFO)
    pub fn from_environment() -> Self {
        let mut config = Self::default();

        // Set log level from environment
        // Precedence: WQM_LOG_LEVEL > RUST_LOG > default
        if let Ok(level_str) = env::var("WQM_LOG_LEVEL") {
            if let Ok(level) = level_str.to_uppercase().parse::<Level>() {
                config.level = level;
            }
        } else if let Ok(level_str) = env::var("RUST_LOG") {
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

        // Log rotation settings
        if let Ok(size_str) = env::var("WQM_LOG_ROTATION_SIZE_MB") {
            if let Ok(size) = size_str.parse::<u64>() {
                config.rotation_size_mb = size;
            }
        }
        if let Ok(count_str) = env::var("WQM_LOG_ROTATION_COUNT") {
            if let Ok(count) = count_str.parse::<usize>() {
                config.rotation_count = count;
            }
        }
        if let Ok(compress_str) = env::var("WQM_LOG_ROTATION_COMPRESS") {
            config.compress_rotated = compress_str.to_lowercase() == "true" || compress_str == "1";
        }

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

    /// Create production logging configuration with canonical OS log paths
    ///
    /// Respects `WQM_LOG_LEVEL`, `WQM_LOG_DIR`, and `WQM_LOG_FILE_PATH` env vars.
    pub fn production() -> Self {
        // Determine log file path using canonical OS-specific locations
        let log_file_path = if let Ok(custom_path) = env::var("WQM_LOG_FILE_PATH") {
            // Use custom path from environment variable (allows override)
            Some(PathBuf::from(custom_path))
        } else {
            // Use canonical OS-specific log directory (respects WQM_LOG_DIR)
            Some(get_canonical_log_dir().join("daemon.jsonl"))
        };

        // Respect WQM_LOG_LEVEL for production too
        let level = env::var("WQM_LOG_LEVEL")
            .ok()
            .and_then(|s| s.to_uppercase().parse::<Level>().ok())
            .unwrap_or(Level::INFO);

        Self {
            level,
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
                fields.insert("component".to_string(), "rust-daemon".to_string());
                fields
            },
            rotation_size_mb: 50,
            rotation_count: 5,
            compress_rotated: true,
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
            rotation_size_mb: 50,
            rotation_count: 5,
            compress_rotated: true,
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

/// Build a rotating file writer using logroller.
///
/// Creates a size-based rotating file appender with optional gzip compression.
/// The directory is created if it doesn't exist.
fn build_log_roller(
    log_file_path: &PathBuf,
    config: &LoggingConfig,
) -> Result<logroller::LogRoller, WorkspaceError> {
    let parent = log_file_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let filename = log_file_path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("daemon.jsonl"))
        .to_string_lossy();

    // Ensure directory exists
    std::fs::create_dir_all(parent).map_err(|e| {
        WorkspaceError::file_system(
            format!("Failed to create log directory: {}", e),
            parent.to_string_lossy().to_string(),
            "create_directory",
        )
    })?;

    let filename_path = std::path::Path::new(filename.as_ref());
    let mut builder = LogRollerBuilder::new(parent, filename_path)
        .rotation(Rotation::SizeBased(RotationSize::MB(config.rotation_size_mb)))
        .max_keep_files(config.rotation_count as u64);

    if config.compress_rotated {
        builder = builder.compression(Compression::Gzip);
    }

    builder.build().map_err(|e| {
        WorkspaceError::file_system(
            format!("Failed to create rotating log file: {}", e),
            log_file_path.to_string_lossy().to_string(),
            "create_log_roller",
        )
    })
}

/// Initialize comprehensive logging system with daemon mode silence support
pub fn initialize_logging(config: LoggingConfig) -> Result<(), WorkspaceError> {
    // Detect daemon mode early for complete suppression
    let daemon_mode = is_daemon_mode();

    // Only create a fully silent subscriber if BOTH console and file logging are disabled
    if !config.console_output && !config.file_logging {
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

    // Create environment filter using configured level (not "off" in daemon mode)
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()));

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
        // macOS XPC_SERVICE_NAME is "0" in regular terminals, check for actual service name
        let xpc_is_service = env::var("XPC_SERVICE_NAME")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        let service_check = daemon_mode ||
            xpc_is_service ||
            env::var("_").map(|v| v.contains("launchd")).unwrap_or(false) ||
            env::var("INVOCATION_ID").is_ok() || // systemd
            env::var("UPSTART_JOB").is_ok() ||   // upstart
            env::var("SYSTEMD_EXEC_PID").is_ok() || // systemd service
            env::var("XPC_FLAGS").map(|v| v == "1").unwrap_or(false); // macOS service mode

        tty_check || service_check
    });

    // Build the subscriber with conditional layers
    let registry = Registry::default();

    // Note: The fully silent case (no console AND no file) is handled at the start of this function.
    // Below we handle combinations where at least one output is enabled.

    // Add layers based on configuration
    if config.console_output && config.file_logging {
        // Both console and file logging
        if let Some(ref log_file_path) = config.log_file_path {
            let roller = build_log_roller(log_file_path, &config)?;
            let (non_blocking, _guard) = tracing_appender::non_blocking(roller);
            // Leak the guard so the non-blocking writer stays alive for the process lifetime
            std::mem::forget(_guard);

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
                .with_writer(non_blocking)
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
            let roller = build_log_roller(log_file_path, &config)?;
            let (non_blocking, _guard) = tracing_appender::non_blocking(roller);
            std::mem::forget(_guard);

            let file_layer = fmt::layer()
                .json()
                .with_writer(non_blocking)
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

    // macOS LaunchAgent/LaunchDaemon - XPC_SERVICE_NAME is set to "0" in regular
    // terminal sessions, so we check that it's not empty and not "0"
    if let Ok(xpc_name) = env::var("XPC_SERVICE_NAME") {
        if !xpc_name.is_empty() && xpc_name != "0" {
            return true;
        }
    }

    // Detect common service/daemon environments
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

// =====================================================================
// Multi-tenant Structured Logging Functions (Task 412.8)
// =====================================================================

/// Session lifecycle events logging context
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub project_id: String,
    pub session_id: Option<String>,
    pub tenant_id: Option<String>,
    pub priority: Option<String>,
}

impl SessionContext {
    /// Create a new session context
    pub fn new(project_id: impl Into<String>) -> Self {
        Self {
            project_id: project_id.into(),
            session_id: None,
            tenant_id: None,
            priority: None,
        }
    }

    /// Builder method to add session ID
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Builder method to add tenant ID
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Builder method to add priority
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

    // Warn on slow heartbeats (>100ms)
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
    /// Create a new queue context
    pub fn new(collection: impl Into<String>, priority: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            priority: priority.into(),
            tenant_id: None,
        }
    }

    /// Builder method to add tenant ID
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

    // Warn on high queue depth
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

    // Warn on slow processing (>10s)
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
    /// Create a new search context
    pub fn new(collection: impl Into<String>, query_type: impl Into<String>, limit: usize) -> Self {
        Self {
            collection: collection.into(),
            tenant_id: None,
            query_type: query_type.into(),
            limit,
        }
    }

    /// Builder method to add tenant ID
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

// =====================================================================
// End Multi-tenant Structured Logging Functions
// =====================================================================

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
            "WQM_LOG_LEVEL",
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

        env::remove_var("WQM_LOG_LEVEL");
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

    #[serial]
    #[test]
    fn test_wqm_log_level_takes_precedence_over_rust_log() {
        let prev_wqm = env::var("WQM_LOG_LEVEL").ok();
        let prev_rust = env::var("RUST_LOG").ok();

        // Set both; WQM_LOG_LEVEL should win
        env::set_var("WQM_LOG_LEVEL", "WARN");
        env::set_var("RUST_LOG", "TRACE");

        let config = LoggingConfig::from_environment();
        assert_eq!(config.level, Level::WARN);

        // Clean up
        match prev_wqm {
            Some(v) => env::set_var("WQM_LOG_LEVEL", v),
            None => env::remove_var("WQM_LOG_LEVEL"),
        }
        match prev_rust {
            Some(v) => env::set_var("RUST_LOG", v),
            None => env::remove_var("RUST_LOG"),
        }
    }

    #[serial]
    #[test]
    fn test_wqm_log_dir_overrides_default() {
        let prev = env::var("WQM_LOG_DIR").ok();

        env::set_var("WQM_LOG_DIR", "/custom/log/path");
        let dir = get_canonical_log_dir();
        assert_eq!(dir, PathBuf::from("/custom/log/path"));

        // Without the env var, should return a platform default (not /custom)
        env::remove_var("WQM_LOG_DIR");
        let dir = get_canonical_log_dir();
        assert_ne!(dir, PathBuf::from("/custom/log/path"));

        // Clean up
        match prev {
            Some(v) => env::set_var("WQM_LOG_DIR", v),
            None => env::remove_var("WQM_LOG_DIR"),
        }
    }

    #[serial]
    #[test]
    fn test_rotation_config_from_environment() {
        let keys = [
            "WQM_LOG_ROTATION_SIZE_MB",
            "WQM_LOG_ROTATION_COUNT",
            "WQM_LOG_ROTATION_COMPRESS",
        ];
        let previous: Vec<Option<String>> = keys.iter().map(|k| env::var(k).ok()).collect();

        env::set_var("WQM_LOG_ROTATION_SIZE_MB", "100");
        env::set_var("WQM_LOG_ROTATION_COUNT", "10");
        env::set_var("WQM_LOG_ROTATION_COMPRESS", "false");

        let config = LoggingConfig::from_environment();
        assert_eq!(config.rotation_size_mb, 100);
        assert_eq!(config.rotation_count, 10);
        assert!(!config.compress_rotated);

        for (key, value) in keys.iter().zip(previous) {
            match value {
                Some(v) => env::set_var(key, v),
                None => env::remove_var(key),
            }
        }
    }

    #[test]
    fn test_rotation_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.rotation_size_mb, 50);
        assert_eq!(config.rotation_count, 5);
        assert!(config.compress_rotated);
    }

    #[test]
    fn test_build_log_roller_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("test.jsonl");
        let config = LoggingConfig::default();

        let roller = build_log_roller(&log_path, &config);
        assert!(roller.is_ok(), "log roller should be created successfully");
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
