//! Processing configuration (queue processor, startup warmup)

use crate::queue_types::ProcessorConfig;
use chrono::Duration as ChronoDuration;
use serde::{Deserialize, Serialize};

// --- Queue Processor Settings ---

fn default_batch_size() -> i32 {
    10
}
fn default_poll_interval_ms() -> u64 {
    500
}
fn default_max_retries() -> i32 {
    5
}
pub(crate) fn default_retry_delays_seconds() -> Vec<u64> {
    vec![60, 300, 900, 3600]
}
fn default_target_throughput() -> u64 {
    1000
}
fn default_enable_metrics() -> bool {
    true
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
            retry_delays: settings
                .retry_delays_seconds
                .into_iter()
                .map(|s| ChronoDuration::seconds(s as i64))
                .collect(),
            target_throughput: settings.target_throughput,
            enable_metrics: settings.enable_metrics,
            // Task 21: Use defaults for new fields
            worker_count: 4,
            backpressure_threshold: 1000,
        }
    }
}

// --- Startup Config ---

fn default_warmup_delay_secs() -> u64 {
    5
}
fn default_warmup_window_secs() -> u64 {
    30
}
fn default_warmup_max_concurrent_embeddings() -> usize {
    1
}
fn default_startup_enqueue_batch_size() -> usize {
    50
}
fn default_startup_enqueue_batch_delay_ms() -> u64 {
    100
}
fn default_reconcile_modified_on_startup() -> bool {
    true
}

/// Startup warmup configuration section (Task 577)
///
/// Controls the daemon's startup behavior to reduce initial CPU spike
/// by throttling queue processing during the warmup window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupConfig {
    /// Delay in seconds before queue processor starts consuming (default: 5)
    #[serde(default = "default_warmup_delay_secs")]
    pub warmup_delay_secs: u64,

    /// Duration in seconds of the warmup window with reduced limits (default: 30)
    #[serde(default = "default_warmup_window_secs")]
    pub warmup_window_secs: u64,

    /// Max concurrent embeddings during warmup (default: 1, normal: 2)
    #[serde(default = "default_warmup_max_concurrent_embeddings")]
    pub warmup_max_concurrent_embeddings: usize,

    /// Batch size for startup recovery enqueuing (default: 50)
    #[serde(default = "default_startup_enqueue_batch_size")]
    pub startup_enqueue_batch_size: usize,

    /// Delay in ms between enqueue batches (default: 100)
    #[serde(default = "default_startup_enqueue_batch_delay_ms")]
    pub startup_enqueue_batch_delay_ms: u64,

    /// On startup, re-hash each tracked file still on disk and enqueue an Update
    /// when its content changed while the daemon was down (or in an inactive
    /// project, which live watch events skip). Catches edits the notify watcher
    /// never saw. Default: true. Disable for very large watch roots where
    /// re-hashing every file at boot is too costly.
    #[serde(default = "default_reconcile_modified_on_startup")]
    pub reconcile_modified_on_startup: bool,
}

impl Default for StartupConfig {
    fn default() -> Self {
        Self {
            warmup_delay_secs: default_warmup_delay_secs(),
            warmup_window_secs: default_warmup_window_secs(),
            warmup_max_concurrent_embeddings: default_warmup_max_concurrent_embeddings(),
            startup_enqueue_batch_size: default_startup_enqueue_batch_size(),
            startup_enqueue_batch_delay_ms: default_startup_enqueue_batch_delay_ms(),
            reconcile_modified_on_startup: default_reconcile_modified_on_startup(),
        }
    }
}

impl StartupConfig {
    /// Validate configuration settings.
    ///
    /// Bounds:
    /// - `warmup_delay_secs` in [0, 600]
    /// - `warmup_window_secs` in [0, 600]
    /// - `warmup_max_concurrent_embeddings` must be > 0
    /// - `startup_enqueue_batch_size` in [1, 1000]
    pub fn validate(&self) -> Result<(), String> {
        if self.warmup_delay_secs > 600 {
            return Err("warmup_delay_secs must not exceed 600".to_string());
        }
        if self.warmup_window_secs > 600 {
            return Err("warmup_window_secs must not exceed 600".to_string());
        }
        if self.warmup_max_concurrent_embeddings == 0 {
            return Err("warmup_max_concurrent_embeddings must be greater than 0".to_string());
        }
        if self.startup_enqueue_batch_size == 0 {
            return Err("startup_enqueue_batch_size must be at least 1".to_string());
        }
        if self.startup_enqueue_batch_size > 1000 {
            return Err("startup_enqueue_batch_size must not exceed 1000".to_string());
        }
        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_STARTUP_WARMUP_DELAY_SECS") {
            if let Ok(parsed) = val.parse() {
                self.warmup_delay_secs = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_STARTUP_WARMUP_WINDOW_SECS") {
            if let Ok(parsed) = val.parse() {
                self.warmup_window_secs = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_STARTUP_MAX_CONCURRENT_EMBEDDINGS") {
            if let Ok(parsed) = val.parse() {
                self.warmup_max_concurrent_embeddings = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_STARTUP_ENQUEUE_BATCH_SIZE") {
            if let Ok(parsed) = val.parse() {
                self.startup_enqueue_batch_size = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_STARTUP_ENQUEUE_BATCH_DELAY_MS") {
            if let Ok(parsed) = val.parse() {
                self.startup_enqueue_batch_delay_ms = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_STARTUP_RECONCILE_MODIFIED") {
            if let Ok(parsed) = val.parse() {
                self.reconcile_modified_on_startup = parsed;
            }
        }
    }

    /// Create from environment variables only
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env_overrides();
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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
    fn test_startup_config_defaults() {
        let config = StartupConfig::default();
        assert_eq!(config.warmup_delay_secs, 5);
        assert_eq!(config.warmup_window_secs, 30);
        assert_eq!(config.warmup_max_concurrent_embeddings, 1);
        assert_eq!(config.startup_enqueue_batch_size, 50);
        assert_eq!(config.startup_enqueue_batch_delay_ms, 100);
    }

    #[test]
    #[serial]
    fn test_startup_config_env_overrides() {
        let mut config = StartupConfig::default();

        // Set environment variables
        std::env::set_var("WQM_STARTUP_WARMUP_DELAY_SECS", "10");
        std::env::set_var("WQM_STARTUP_WARMUP_WINDOW_SECS", "60");
        std::env::set_var("WQM_STARTUP_MAX_CONCURRENT_EMBEDDINGS", "2");
        std::env::set_var("WQM_STARTUP_ENQUEUE_BATCH_SIZE", "100");
        std::env::set_var("WQM_STARTUP_ENQUEUE_BATCH_DELAY_MS", "200");

        config.apply_env_overrides();

        assert_eq!(config.warmup_delay_secs, 10);
        assert_eq!(config.warmup_window_secs, 60);
        assert_eq!(config.warmup_max_concurrent_embeddings, 2);
        assert_eq!(config.startup_enqueue_batch_size, 100);
        assert_eq!(config.startup_enqueue_batch_delay_ms, 200);

        // Clean up
        std::env::remove_var("WQM_STARTUP_WARMUP_DELAY_SECS");
        std::env::remove_var("WQM_STARTUP_WARMUP_WINDOW_SECS");
        std::env::remove_var("WQM_STARTUP_MAX_CONCURRENT_EMBEDDINGS");
        std::env::remove_var("WQM_STARTUP_ENQUEUE_BATCH_SIZE");
        std::env::remove_var("WQM_STARTUP_ENQUEUE_BATCH_DELAY_MS");
    }

    #[test]
    #[serial]
    fn test_startup_config_from_env() {
        std::env::set_var("WQM_STARTUP_WARMUP_DELAY_SECS", "3");
        std::env::set_var("WQM_STARTUP_WARMUP_WINDOW_SECS", "15");

        let config = StartupConfig::from_env();
        assert_eq!(config.warmup_delay_secs, 3);
        assert_eq!(config.warmup_window_secs, 15);
        // Other fields should be defaults
        assert_eq!(config.warmup_max_concurrent_embeddings, 1);

        std::env::remove_var("WQM_STARTUP_WARMUP_DELAY_SECS");
        std::env::remove_var("WQM_STARTUP_WARMUP_WINDOW_SECS");
    }

    #[test]
    fn test_startup_config_validate_default_ok() {
        let config = StartupConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_startup_config_validate_rejects_zero_max_concurrent() {
        let config = StartupConfig {
            warmup_max_concurrent_embeddings: 0,
            ..StartupConfig::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("warmup_max_concurrent_embeddings"));
    }

    #[test]
    fn test_startup_config_validate_rejects_zero_batch_size() {
        let config = StartupConfig {
            startup_enqueue_batch_size: 0,
            ..StartupConfig::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("startup_enqueue_batch_size"));
    }

    #[test]
    fn test_startup_config_validate_rejects_batch_size_over_1000() {
        let config = StartupConfig {
            startup_enqueue_batch_size: 1001,
            ..StartupConfig::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("startup_enqueue_batch_size"));
    }

    #[test]
    fn test_startup_config_validate_accepts_boundary_values() {
        let config_min = StartupConfig {
            startup_enqueue_batch_size: 1,
            ..StartupConfig::default()
        };
        assert!(config_min.validate().is_ok());

        let config_max = StartupConfig {
            startup_enqueue_batch_size: 1000,
            ..StartupConfig::default()
        };
        assert!(config_max.validate().is_ok());
    }

    #[test]
    fn test_startup_config_validate_rejects_excessive_warmup_delay() {
        let config = StartupConfig {
            warmup_delay_secs: 601,
            ..StartupConfig::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("warmup_delay_secs"));
    }

    #[test]
    fn test_startup_config_serialization() {
        let config = StartupConfig {
            warmup_delay_secs: 8,
            warmup_window_secs: 45,
            warmup_max_concurrent_embeddings: 1,
            startup_enqueue_batch_size: 75,
            startup_enqueue_batch_delay_ms: 150,
            reconcile_modified_on_startup: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"warmup_delay_secs\":8"));
        assert!(json.contains("\"warmup_window_secs\":45"));

        let deserialized: StartupConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.warmup_delay_secs, 8);
        assert_eq!(deserialized.warmup_window_secs, 45);
    }
}
