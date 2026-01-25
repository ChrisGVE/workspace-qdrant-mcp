//! Configuration management
//!
//! This module contains configuration management for the priority processing engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::storage::StorageConfig;
use crate::queue_types::ProcessorConfig;
use chrono::Duration as ChronoDuration;

/// Auto-ingestion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoIngestionConfig {
    pub enabled: bool,
    pub auto_create_watches: bool,
    pub include_common_files: bool,
    pub include_source_files: bool,
    pub target_collection_suffix: String,
    pub max_files_per_batch: usize,
    pub batch_delay_seconds: f64,
    pub max_file_size_mb: usize,
    pub recursive_depth: usize,
    pub debounce_seconds: u64,
}

impl Default for AutoIngestionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_create_watches: true,
            include_common_files: true,
            include_source_files: true,
            target_collection_suffix: "scratchbook".to_string(),
            max_files_per_batch: 5,
            batch_delay_seconds: 2.0,
            max_file_size_mb: 50,
            recursive_depth: 5,
            debounce_seconds: 10,
        }
    }
}

/// Logging configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub info_includes_connection_events: bool,
    pub info_includes_transport_details: bool,
    pub info_includes_retry_attempts: bool,
    pub info_includes_fallback_behavior: bool,
    pub error_includes_stack_trace: bool,
    pub error_includes_connection_state: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            info_includes_connection_events: true,
            info_includes_transport_details: true,
            info_includes_retry_attempts: true,
            info_includes_fallback_behavior: true,
            error_includes_stack_trace: true,
            error_includes_connection_state: true,
        }
    }
}

/// Queue processor configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueProcessorSettings {
    /// Number of items to dequeue per batch
    #[serde(default = "default_batch_size")]
    pub batch_size: i32,

    /// Poll interval in milliseconds
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,

    /// Maximum retry attempts before marking failed
    #[serde(default = "default_max_retries")]
    pub max_retries: i32,

    /// Retry delays in seconds [1m, 5m, 15m, 1h]
    #[serde(default = "default_retry_delays_seconds")]
    pub retry_delays_seconds: Vec<u64>,

    /// Target throughput (docs/min) for monitoring
    #[serde(default = "default_target_throughput")]
    pub target_throughput: u64,

    /// Enable detailed metrics logging
    #[serde(default = "default_enable_metrics")]
    pub enable_metrics: bool,
}

fn default_batch_size() -> i32 { 10 }
fn default_poll_interval_ms() -> u64 { 500 }
fn default_max_retries() -> i32 { 5 }
fn default_retry_delays_seconds() -> Vec<u64> { vec![60, 300, 900, 3600] }
fn default_target_throughput() -> u64 { 1000 }
fn default_enable_metrics() -> bool { true }

impl Default for QueueProcessorSettings {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            poll_interval_ms: default_poll_interval_ms(),
            max_retries: default_max_retries(),
            retry_delays_seconds: default_retry_delays_seconds(),
            target_throughput: default_target_throughput(),
            enable_metrics: default_enable_metrics(),
        }
    }
}

impl QueueProcessorSettings {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size <= 0 {
            return Err("batch_size must be greater than 0".to_string());
        }
        if self.batch_size > 1000 {
            return Err("batch_size should not exceed 1000".to_string());
        }

        if self.poll_interval_ms == 0 {
            return Err("poll_interval_ms must be greater than 0".to_string());
        }
        if self.poll_interval_ms > 60_000 {
            return Err("poll_interval_ms should not exceed 60 seconds".to_string());
        }

        if self.max_retries < 0 {
            return Err("max_retries must be non-negative".to_string());
        }
        if self.max_retries > 20 {
            return Err("max_retries should not exceed 20".to_string());
        }

        if self.retry_delays_seconds.is_empty() {
            return Err("retry_delays_seconds cannot be empty".to_string());
        }

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_QUEUE_BATCH_SIZE") {
            if let Ok(parsed) = val.parse() {
                self.batch_size = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_QUEUE_POLL_INTERVAL_MS") {
            if let Ok(parsed) = val.parse() {
                self.poll_interval_ms = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_QUEUE_MAX_RETRIES") {
            if let Ok(parsed) = val.parse() {
                self.max_retries = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_QUEUE_TARGET_THROUGHPUT") {
            if let Ok(parsed) = val.parse() {
                self.target_throughput = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_QUEUE_ENABLE_METRICS") {
            self.enable_metrics = val.to_lowercase() == "true" || val == "1";
        }
    }
}

/// Convert QueueProcessorSettings to ProcessorConfig
impl From<QueueProcessorSettings> for ProcessorConfig {
    fn from(settings: QueueProcessorSettings) -> Self {
        ProcessorConfig {
            batch_size: settings.batch_size,
            poll_interval_ms: settings.poll_interval_ms,
            max_retries: settings.max_retries,
            retry_delays: settings.retry_delays_seconds
                .into_iter()
                .map(|s| ChronoDuration::seconds(s as i64))
                .collect(),
            target_throughput: settings.target_throughput,
            enable_metrics: settings.enable_metrics,
            // Task 21: Use defaults for new fields
            worker_count: 4,
            backpressure_threshold: 1000,
            parallel_processing: true,
        }
    }
}

/// Tool monitoring configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Check interval in hours
    #[serde(default = "default_check_interval_hours")]
    pub check_interval_hours: u64,

    /// Check on daemon startup
    #[serde(default = "default_check_on_startup")]
    pub check_on_startup: bool,

    /// Enable tool availability monitoring
    #[serde(default = "default_enable_monitoring")]
    pub enable_monitoring: bool,
}

fn default_check_interval_hours() -> u64 { 24 }
fn default_check_on_startup() -> bool { true }
fn default_enable_monitoring() -> bool { true }

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            check_interval_hours: default_check_interval_hours(),
            check_on_startup: default_check_on_startup(),
            enable_monitoring: default_enable_monitoring(),
        }
    }
}

impl MonitoringConfig {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.check_interval_hours == 0 {
            return Err("check_interval_hours must be greater than 0".to_string());
        }
        if self.check_interval_hours > 8760 { // 1 year
            return Err("check_interval_hours should not exceed 8760 (1 year)".to_string());
        }

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_MONITOR_CHECK_INTERVAL_HOURS") {
            if let Ok(parsed) = val.parse() {
                self.check_interval_hours = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_MONITOR_CHECK_ON_STARTUP") {
            self.check_on_startup = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = env::var("WQM_MONITOR_ENABLE") {
            self.enable_monitoring = val.to_lowercase() == "true" || val == "1";
        }
    }
}

/// Observability configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Collection interval in seconds
    #[serde(default = "default_collection_interval")]
    pub collection_interval: u64,

    /// Basic metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Detailed telemetry configuration
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct MetricsConfig {
    #[serde(default)]
    pub enabled: bool,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_history_retention")]
    pub history_retention: usize,

    #[serde(default = "default_telemetry_enabled")]
    pub cpu_usage: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub memory_usage: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub latency: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub queue_depth: bool,

    #[serde(default = "default_telemetry_enabled")]
    pub throughput: bool,
}

fn default_collection_interval() -> u64 { 60 }
fn default_history_retention() -> usize { 120 }
fn default_telemetry_enabled() -> bool { true }

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            history_retention: default_history_retention(),
            cpu_usage: default_telemetry_enabled(),
            memory_usage: default_telemetry_enabled(),
            latency: default_telemetry_enabled(),
            queue_depth: default_telemetry_enabled(),
            throughput: default_telemetry_enabled(),
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            collection_interval: default_collection_interval(),
            metrics: MetricsConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

/// Git integration configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitConfig {
    /// Enable Git branch detection
    #[serde(default = "default_enable_branch_detection")]
    pub enable_branch_detection: bool,

    /// Cache TTL in seconds for branch info
    #[serde(default = "default_cache_ttl_seconds")]
    pub cache_ttl_seconds: u64,
}

fn default_enable_branch_detection() -> bool { true }
fn default_cache_ttl_seconds() -> u64 { 60 }

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            enable_branch_detection: default_enable_branch_detection(),
            cache_ttl_seconds: default_cache_ttl_seconds(),
        }
    }
}

impl GitConfig {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.cache_ttl_seconds == 0 {
            return Err("cache_ttl_seconds must be greater than 0".to_string());
        }
        if self.cache_ttl_seconds > 3600 {
            return Err("cache_ttl_seconds should not exceed 3600 (1 hour)".to_string());
        }

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_GIT_ENABLE_BRANCH_DETECTION") {
            self.enable_branch_detection = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = env::var("WQM_GIT_CACHE_TTL_SECONDS") {
            if let Ok(parsed) = val.parse() {
                self.cache_ttl_seconds = parsed;
            }
        }
    }
}

/// Complete daemon configuration that matches the TOML structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Log file path
    pub log_file: Option<PathBuf>,
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: Option<usize>,
    /// Default timeout for tasks in milliseconds
    pub default_timeout_ms: Option<usize>,
    /// Enable task preemption
    pub enable_preemption: bool,
    /// Document processing chunk size
    pub chunk_size: usize,
    /// Log level configuration
    pub log_level: String,
    /// Auto-ingestion configuration
    pub auto_ingestion: AutoIngestionConfig,
    /// Project path (workspace directory)
    pub project_path: Option<PathBuf>,
    /// Qdrant connection configuration
    pub qdrant: StorageConfig,
    /// Enhanced logging configuration
    pub logging: LoggingConfig,
    /// Queue processor configuration
    #[serde(default)]
    pub queue_processor: QueueProcessorSettings,
    /// Tool monitoring configuration
    #[serde(default)]
    pub monitoring: MonitoringConfig,
    /// Git integration configuration
    #[serde(default)]
    pub git: GitConfig,
    /// Observability configuration (metrics and telemetry)
    #[serde(default)]
    pub observability: ObservabilityConfig,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            log_file: None,
            max_concurrent_tasks: Some(4),
            default_timeout_ms: Some(30_000),
            enable_preemption: true,
            chunk_size: 1000,
            log_level: "info".to_string(),
            auto_ingestion: AutoIngestionConfig::default(),
            project_path: None,
            qdrant: StorageConfig::default(),
            logging: LoggingConfig::default(),
            queue_processor: QueueProcessorSettings::default(),
            monitoring: MonitoringConfig::default(),
            git: GitConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

impl DaemonConfig {
    /// Create a daemon-mode configuration optimized for MCP stdio protocol compliance
    /// This configuration disables compatibility checking to prevent console output
    pub fn daemon_mode() -> Self {
        let mut config = Self::default();
        config.qdrant = StorageConfig::daemon_mode(); // Use silent StorageConfig
        config
    }
}

/// Processing engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: Option<usize>,
    /// Default timeout for tasks in milliseconds
    pub default_timeout_ms: Option<u64>,
    /// Enable task preemption
    pub enable_preemption: bool,
    /// Document processing chunk size
    pub chunk_size: usize,
    /// Log level configuration
    pub log_level: String,
    /// Enable metrics collection (derived from observability.metrics.enabled)
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds (derived from observability.collection_interval)
    pub metrics_interval_secs: u64,
    /// Database path for SQLite state/queue (Task 21)
    pub database_path: Option<PathBuf>,
    /// Queue processor batch size
    pub queue_batch_size: Option<i32>,
    /// Queue processor poll interval in milliseconds
    pub queue_poll_interval_ms: Option<u64>,
    /// Number of queue processor workers
    pub queue_worker_count: Option<usize>,
    /// Enable parallel queue processing
    pub queue_parallel_processing: Option<bool>,
    /// Queue backpressure threshold
    pub queue_backpressure_threshold: Option<i64>,
}

impl From<DaemonConfig> for Config {
    fn from(daemon_config: DaemonConfig) -> Self {
        Self {
            max_concurrent_tasks: daemon_config.max_concurrent_tasks,
            default_timeout_ms: daemon_config.default_timeout_ms.map(|t| t as u64),
            enable_preemption: daemon_config.enable_preemption,
            chunk_size: daemon_config.chunk_size,
            log_level: daemon_config.log_level,
            enable_metrics: daemon_config.observability.metrics.enabled,
            metrics_interval_secs: daemon_config.observability.collection_interval,
            // Queue processor configuration (Task 21)
            database_path: None, // Will use default path in daemon
            queue_batch_size: Some(daemon_config.queue_processor.batch_size),
            queue_poll_interval_ms: Some(daemon_config.queue_processor.poll_interval_ms),
            queue_worker_count: Some(4), // Default worker count
            queue_parallel_processing: Some(true), // Default to parallel processing
            queue_backpressure_threshold: Some(1000), // Default backpressure threshold
        }
    }
}

impl Config {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self {
            max_concurrent_tasks: Some(4),
            default_timeout_ms: Some(30_000),
            enable_preemption: true,
            chunk_size: 1000,
            log_level: "info".to_string(),
            enable_metrics: true,
            metrics_interval_secs: 60,
            // Queue processor defaults (Task 21)
            database_path: None,
            queue_batch_size: Some(10),
            queue_poll_interval_ms: Some(500),
            queue_worker_count: Some(4),
            queue_parallel_processing: Some(true),
            queue_backpressure_threshold: Some(1000),
        }
    }

    /// Enable daemon mode with silent operation
    pub fn daemon_mode() -> Self {
        Self::new()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_processor_settings_defaults() {
        let settings = QueueProcessorSettings::default();
        assert_eq!(settings.batch_size, 10);
        assert_eq!(settings.poll_interval_ms, 500);
        assert_eq!(settings.max_retries, 5);
        assert_eq!(settings.retry_delays_seconds, vec![60, 300, 900, 3600]);
        assert_eq!(settings.target_throughput, 1000);
        assert!(settings.enable_metrics);
    }

    #[test]
    fn test_queue_processor_settings_validation() {
        let mut settings = QueueProcessorSettings::default();

        // Valid settings
        assert!(settings.validate().is_ok());

        // Invalid batch_size
        settings.batch_size = 0;
        assert!(settings.validate().is_err());
        settings.batch_size = 1001;
        assert!(settings.validate().is_err());
        settings.batch_size = 10;

        // Invalid poll_interval
        settings.poll_interval_ms = 0;
        assert!(settings.validate().is_err());
        settings.poll_interval_ms = 61_000;
        assert!(settings.validate().is_err());
        settings.poll_interval_ms = 500;

        // Invalid max_retries
        settings.max_retries = -1;
        assert!(settings.validate().is_err());
        settings.max_retries = 21;
        assert!(settings.validate().is_err());
        settings.max_retries = 5;

        // Empty retry_delays
        settings.retry_delays_seconds = vec![];
        assert!(settings.validate().is_err());
    }

    #[test]
    fn test_monitoring_config_defaults() {
        let config = MonitoringConfig::default();
        assert_eq!(config.check_interval_hours, 24);
        assert!(config.check_on_startup);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_monitoring_config_validation() {
        let mut config = MonitoringConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid check_interval_hours
        config.check_interval_hours = 0;
        assert!(config.validate().is_err());
        config.check_interval_hours = 8761;
        assert!(config.validate().is_err());
        config.check_interval_hours = 24;

        // Valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_git_config_defaults() {
        let config = GitConfig::default();
        assert!(config.enable_branch_detection);
        assert_eq!(config.cache_ttl_seconds, 60);
    }

    #[test]
    fn test_git_config_validation() {
        let mut config = GitConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid cache_ttl_seconds
        config.cache_ttl_seconds = 0;
        assert!(config.validate().is_err());
        config.cache_ttl_seconds = 3601;
        assert!(config.validate().is_err());
        config.cache_ttl_seconds = 60;

        // Valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_daemon_config_defaults() {
        let config = DaemonConfig::default();
        assert_eq!(config.queue_processor.batch_size, 10);
        assert_eq!(config.monitoring.check_interval_hours, 24);
        assert!(config.monitoring.enable_monitoring);
        assert!(config.git.enable_branch_detection);
        assert_eq!(config.git.cache_ttl_seconds, 60);
    }
}
