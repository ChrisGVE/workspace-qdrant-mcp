//! Configuration management
//!
//! This module contains configuration management for the priority processing engine

use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::path::PathBuf;
use crate::storage::StorageConfig;
use crate::queue_processor::ProcessorConfig;
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
    pub default_timeout_ms: Option<u64>,
    /// Enable task preemption
    pub enable_preemption: bool,
    /// Document processing chunk size
    pub chunk_size: usize,
    /// Log level configuration
    pub log_level: String,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,
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
            enable_metrics: true,
            metrics_interval_secs: 60,
            auto_ingestion: AutoIngestionConfig::default(),
            project_path: None,
            qdrant: StorageConfig::default(),
            logging: LoggingConfig::default(),
            queue_processor: QueueProcessorSettings::default(),
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
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,
}

impl From<DaemonConfig> for Config {
    fn from(daemon_config: DaemonConfig) -> Self {
        Self {
            max_concurrent_tasks: daemon_config.max_concurrent_tasks,
            default_timeout_ms: daemon_config.default_timeout_ms,
            enable_preemption: daemon_config.enable_preemption,
            chunk_size: daemon_config.chunk_size,
            log_level: daemon_config.log_level,
            enable_metrics: daemon_config.enable_metrics,
            metrics_interval_secs: daemon_config.metrics_interval_secs,
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
        }
    }

    /// Create configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            max_concurrent_tasks: Some(8),
            default_timeout_ms: Some(10_000),
            enable_preemption: true,
            chunk_size: 2000,
            log_level: "warn".to_string(),
            enable_metrics: true,
            metrics_interval_secs: 30,
        }
    }

    /// Create configuration optimized for responsiveness (MCP servers)
    pub fn responsive() -> Self {
        Self {
            max_concurrent_tasks: Some(2),
            default_timeout_ms: Some(5_000),
            enable_preemption: true,
            chunk_size: 500,
            log_level: "debug".to_string(),
            enable_metrics: true,
            metrics_interval_secs: 10,
        }
    }

    /// Create configuration for resource-constrained environments
    pub fn low_resource() -> Self {
        Self {
            max_concurrent_tasks: Some(1),
            default_timeout_ms: Some(60_000),
            enable_preemption: false,
            chunk_size: 500,
            log_level: "error".to_string(),
            enable_metrics: false,
            metrics_interval_secs: 300,
        }
    }

    /// Get default timeout as Duration
    pub fn default_timeout(&self) -> Option<Duration> {
        self.default_timeout_ms.map(Duration::from_millis)
    }

    /// Get metrics interval as Duration
    pub fn metrics_interval(&self) -> Duration {
        Duration::from_secs(self.metrics_interval_secs)
    }

    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if let Some(max_concurrent) = self.max_concurrent_tasks {
            if max_concurrent == 0 {
                return Err("max_concurrent_tasks must be greater than 0".to_string());
            }
            if max_concurrent > 100 {
                return Err("max_concurrent_tasks should not exceed 100".to_string());
            }
        }

        if let Some(timeout) = self.default_timeout_ms {
            if timeout == 0 {
                return Err("default_timeout_ms must be greater than 0".to_string());
            }
            if timeout > 300_000 {
                return Err("default_timeout_ms should not exceed 5 minutes".to_string());
            }
        }

        if self.chunk_size == 0 {
            return Err("chunk_size must be greater than 0".to_string());
        }
        if self.chunk_size > 10_000 {
            return Err("chunk_size should not exceed 10,000".to_string());
        }

        if !matches!(self.log_level.as_str(), "trace" | "debug" | "info" | "warn" | "error") {
            return Err("log_level must be one of: trace, debug, info, warn, error".to_string());
        }

        Ok(())
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
    fn test_config_creation() {
        let config = Config::new();
        assert_eq!(config.max_concurrent_tasks, Some(4));
        assert_eq!(config.default_timeout_ms, Some(30_000));
        assert!(config.enable_preemption);
        assert_eq!(config.chunk_size, 1000);
        // LSP configuration is handled separately via LspConfig
    }

    #[test]
    fn test_high_throughput_config() {
        let config = Config::high_throughput();
        assert_eq!(config.max_concurrent_tasks, Some(8));
        assert_eq!(config.default_timeout_ms, Some(10_000));
        assert_eq!(config.chunk_size, 2000);
    }

    #[test]
    fn test_responsive_config() {
        let config = Config::responsive();
        assert_eq!(config.max_concurrent_tasks, Some(2));
        assert_eq!(config.default_timeout_ms, Some(5_000));
        assert_eq!(config.chunk_size, 500);
        // LSP configuration is handled separately via LspConfig
    }

    #[test]
    fn test_low_resource_config() {
        let config = Config::low_resource();
        assert_eq!(config.max_concurrent_tasks, Some(1));
        assert_eq!(config.default_timeout_ms, Some(60_000));
        assert!(!config.enable_preemption);
        assert!(!config.enable_metrics);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::new();
        assert!(config.validate().is_ok());

        // Test invalid max_concurrent_tasks
        config.max_concurrent_tasks = Some(0);
        assert!(config.validate().is_err());

        config.max_concurrent_tasks = Some(150);
        assert!(config.validate().is_err());

        // Reset and test invalid timeout
        config.max_concurrent_tasks = Some(4);
        config.default_timeout_ms = Some(0);
        assert!(config.validate().is_err());

        // Test invalid chunk size
        config.default_timeout_ms = Some(30_000);
        config.chunk_size = 0;
        assert!(config.validate().is_err());

        config.chunk_size = 20_000;
        assert!(config.validate().is_err());

        // Test invalid log level
        config.chunk_size = 1000;
        config.log_level = "invalid".to_string();
        assert!(config.validate().is_err());
    }


    #[test]
    fn test_duration_helpers() {
        let config = Config {
            default_timeout_ms: Some(5000),
            metrics_interval_secs: 120,
            ..Default::default()
        };

        assert_eq!(config.default_timeout(), Some(Duration::from_millis(5000)));
        assert_eq!(config.metrics_interval(), Duration::from_secs(120));

        let config_no_timeout = Config {
            default_timeout_ms: None,
            ..Default::default()
        };
        assert_eq!(config_no_timeout.default_timeout(), None);
    }

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
        assert!(settings.validate().is_ok());

        // Test invalid batch_size
        settings.batch_size = 0;
        assert!(settings.validate().is_err());

        settings.batch_size = 2000;
        assert!(settings.validate().is_err());

        // Reset and test invalid poll_interval
        settings.batch_size = 10;
        settings.poll_interval_ms = 0;
        assert!(settings.validate().is_err());

        settings.poll_interval_ms = 100_000;
        assert!(settings.validate().is_err());

        // Test invalid max_retries
        settings.poll_interval_ms = 500;
        settings.max_retries = -1;
        assert!(settings.validate().is_err());

        settings.max_retries = 50;
        assert!(settings.validate().is_err());

        // Test empty retry_delays
        settings.max_retries = 5;
        settings.retry_delays_seconds = vec![];
        assert!(settings.validate().is_err());
    }

    #[test]
    fn test_daemon_config_includes_queue_processor() {
        let config = DaemonConfig::default();
        assert_eq!(config.queue_processor.batch_size, 10);
        assert_eq!(config.queue_processor.poll_interval_ms, 500);
    }

    #[test]
    fn test_queue_processor_settings_conversion() {
        let settings = QueueProcessorSettings::default();
        let processor_config: ProcessorConfig = settings.into();

        assert_eq!(processor_config.batch_size, 10);
        assert_eq!(processor_config.poll_interval_ms, 500);
        assert_eq!(processor_config.max_retries, 5);
        assert_eq!(processor_config.retry_delays.len(), 4);
        assert_eq!(processor_config.target_throughput, 1000);
        assert!(processor_config.enable_metrics);
    }
}
