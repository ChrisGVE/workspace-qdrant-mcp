//! Logging configuration, initialization, and performance tracking
//!
//! Provides LoggingConfig, PerformanceMetrics, and the initialize_logging function.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use atty::Stream;

use logroller::{LogRollerBuilder, Rotation, RotationSize, Compression};
use tracing::{info, Level, Subscriber};
use tracing_subscriber::{
    fmt::{self, time::ChronoUtc},
    layer::{SubscriberExt, Layer},
    registry::LookupSpan,
    util::SubscriberInitExt,
    EnvFilter, Registry,
    filter::LevelFilter,
};

use crate::error::WorkspaceError;

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
pub fn get_canonical_log_dir() -> PathBuf {
    wqm_common::paths::get_canonical_log_dir()
}

impl LoggingConfig {
    /// Create logging configuration from environment variables
    ///
    /// Log level precedence: `WQM_LOG_LEVEL` > `RUST_LOG` > default (INFO)
    pub fn from_environment() -> Self {
        let mut config = Self::default();

        if let Ok(level_str) = env::var("WQM_LOG_LEVEL") {
            if let Ok(level) = level_str.to_uppercase().parse::<Level>() {
                config.level = level;
            }
        } else if let Ok(level_str) = env::var("RUST_LOG") {
            if let Ok(level) = level_str.to_uppercase().parse::<Level>() {
                config.level = level;
            }
        }

        config.json_format = env::var("WQM_LOG_JSON")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        config.console_output = env::var("WQM_LOG_CONSOLE")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true);

        config.file_logging = env::var("WQM_LOG_FILE")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        if let Ok(file_path) = env::var("WQM_LOG_FILE_PATH") {
            config.log_file_path = Some(PathBuf::from(file_path));
            config.file_logging = true;
        }

        config.performance_metrics = env::var("WQM_LOG_METRICS")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true);

        config.error_tracking = env::var("WQM_LOG_ERROR_TRACKING")
            .map(|v| v.to_lowercase() != "false" && v != "0")
            .unwrap_or(true);

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
    pub fn production() -> Self {
        let log_file_path = if let Ok(custom_path) = env::var("WQM_LOG_FILE_PATH") {
            Some(PathBuf::from(custom_path))
        } else {
            Some(get_canonical_log_dir().join("daemon.jsonl"))
        };

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
            force_disable_ansi: Some(true),
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
            force_disable_ansi: None,
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

/// Suppress any potential TTY detection debug output
pub fn suppress_tty_debug_output() {
    std::env::set_var("TTY_DETECTION_SILENT", "1");
    std::env::set_var("ATTY_FORCE_DISABLE_DEBUG", "1");
    std::env::set_var("WQM_TTY_DEBUG", "0");
    std::env::set_var("TTY_DEBUG", "0");
    std::env::set_var("ATTY_SILENT", "1");
    std::env::set_var("CONSOLE_NO_DEBUG", "1");
    std::env::set_var("TERMINAL_DEBUG", "0");
    std::env::set_var("NO_COLOR", "1");
    std::env::set_var("TERM", "dumb");
}

/// Build a rotating file writer using logroller.
fn build_log_roller(
    log_file_path: &PathBuf,
    config: &LoggingConfig,
) -> Result<logroller::LogRoller, WorkspaceError> {
    let parent = log_file_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let filename = log_file_path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("daemon.jsonl"))
        .to_string_lossy();

    std::fs::create_dir_all(parent).map_err(|e| {
        WorkspaceError::file_system(
            format!("Failed to create log directory: {}", e),
            parent.to_string_lossy().to_string(),
            "create_directory",
        )
    })?;

    let filename_str: &str = filename.as_ref();
    let filename_path = std::path::Path::new(filename_str);
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

/// Determine whether ANSI escape codes should be disabled for console output.
///
/// Uses the explicit `force_disable_ansi` override when set, otherwise auto-detects
/// based on environment variables, TTY status, and daemon/service indicators.
fn determine_disable_ansi(daemon_mode: bool, force_disable_ansi: Option<bool>) -> bool {
    force_disable_ansi.unwrap_or_else(|| {
        if env::var("NO_COLOR").is_ok() {
            return true;
        }
        if env::var("TTY_DETECTION_SILENT").is_ok()
            || env::var("WQM_TTY_DEBUG") == Ok("0".to_string())
        {
            return true;
        }
        let tty_check = !atty::is(Stream::Stdout);
        let xpc_is_service = env::var("XPC_SERVICE_NAME")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        let service_check = daemon_mode
            || xpc_is_service
            || env::var("_").map(|v| v.contains("launchd")).unwrap_or(false)
            || env::var("INVOCATION_ID").is_ok()
            || env::var("UPSTART_JOB").is_ok()
            || env::var("SYSTEMD_EXEC_PID").is_ok()
            || env::var("XPC_FLAGS").map(|v| v == "1").unwrap_or(false);
        tty_check || service_check
    })
}

/// Build a console tracing layer, choosing JSON or pretty format.
fn build_console_layer<S>(
    json_format: bool,
    disable_ansi: bool,
) -> Box<dyn Layer<S> + Send + Sync>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    if json_format {
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
    }
}

/// Build a non-blocking rotating file writer from config.
///
/// The non-blocking guard is intentionally leaked via `std::mem::forget` so the
/// writer stays alive for the process lifetime.
fn build_non_blocking_writer(
    log_file_path: &PathBuf,
    config: &LoggingConfig,
) -> Result<tracing_appender::non_blocking::NonBlocking, WorkspaceError> {
    let roller = build_log_roller(log_file_path, config)?;
    let (non_blocking, _guard) = tracing_appender::non_blocking(roller);
    std::mem::forget(_guard);
    Ok(non_blocking)
}

/// Initialize comprehensive logging system with daemon mode silence support
pub fn initialize_logging(config: LoggingConfig) -> Result<(), WorkspaceError> {
    let daemon_mode = is_daemon_mode();

    if !config.console_output && !config.file_logging {
        let null_writer = || std::io::sink();
        let subscriber = tracing_subscriber::fmt::Subscriber::builder()
            .with_max_level(LevelFilter::OFF)
            .with_writer(null_writer)
            .with_ansi(false)
            .finish();
        subscriber.init();
        return Ok(());
    }

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()));
    let disable_ansi = determine_disable_ansi(daemon_mode, config.force_disable_ansi);
    let registry = Registry::default();

    if config.console_output && config.file_logging {
        let log_file_path = config.log_file_path.as_ref().ok_or_else(|| {
            WorkspaceError::configuration("File logging enabled but no log file path specified")
        })?;
        let non_blocking = build_non_blocking_writer(log_file_path, &config)?;
        let console_layer = build_console_layer(config.json_format, disable_ansi);
        let file_layer = fmt::layer().json()
            .with_writer(non_blocking)
            .with_timer(ChronoUtc::rfc_3339())
            .with_target(true).with_thread_ids(true).with_thread_names(true);
        registry.with(env_filter).with(console_layer).with(file_layer).init();
    } else if config.console_output {
        let console_layer = build_console_layer(config.json_format, disable_ansi);
        registry.with(env_filter).with(console_layer).init();
    } else if config.file_logging {
        let log_file_path = config.log_file_path.as_ref().ok_or_else(|| {
            WorkspaceError::configuration("File logging enabled but no log file path specified")
        })?;
        let non_blocking = build_non_blocking_writer(log_file_path, &config)?;
        let file_layer = fmt::layer().json()
            .with_writer(non_blocking)
            .with_timer(ChronoUtc::rfc_3339())
            .with_target(true).with_thread_ids(true).with_thread_names(true);
        registry.with(env_filter).with(file_layer).init();
    } else if daemon_mode {
        let null_writer = || std::io::sink();
        let null_layer = fmt::layer().with_writer(null_writer).with_ansi(false);
        registry.with(env_filter).with(null_layer).init();
    } else {
        registry.with(env_filter).init();
    }

    info!(config = ?config, "Logging system initialized");
    for (key, value) in &config.global_fields {
        tracing::Span::current().record(key.as_str(), value.as_str());
    }

    Ok(())
}

/// Detect if running in daemon mode based on multiple indicators
fn is_daemon_mode() -> bool {
    if env::var("WQM_SERVICE_MODE").map(|v| v == "true").unwrap_or(false) {
        return true;
    }
    if let Ok(xpc_name) = env::var("XPC_SERVICE_NAME") {
        if !xpc_name.is_empty() && xpc_name != "0" {
            return true;
        }
    }
    env::var("_").map(|v| v.contains("launchd")).unwrap_or(false) ||
        env::var("INVOCATION_ID").is_ok() ||
        env::var("UPSTART_JOB").is_ok() ||
        env::var("SYSTEMD_EXEC_PID").is_ok() ||
        env::var("XPC_FLAGS").map(|v| v == "1").unwrap_or(false) ||
        env::var("SYSLOG_IDENTIFIER").is_ok() ||
        env::var("LOGNAME").map(|v| v == "root").unwrap_or(false)
}

/// Initialize complete output suppression for daemon mode
pub fn initialize_daemon_silence() -> Result<(), Box<dyn std::error::Error>> {
    use std::io;
    use tracing_subscriber::fmt;

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
