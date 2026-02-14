//! Configuration management
//!
//! This module contains configuration management for the priority processing engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::storage::StorageConfig;
use crate::queue_types::ProcessorConfig;
use chrono::Duration as ChronoDuration;
use wqm_common::yaml_defaults::{self, YamlConfig};

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

/// Embedding generation configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSettings {
    /// Maximum number of cached embedding results
    #[serde(default = "default_cache_max_entries")]
    pub cache_max_entries: usize,

    /// Directory for storing downloaded model files
    /// Default: Uses system-appropriate cache directory (~/.cache/fastembed/)
    #[serde(default)]
    pub model_cache_dir: Option<PathBuf>,
}

fn default_cache_max_entries() -> usize { 1000 }

impl Default for EmbeddingSettings {
    fn default() -> Self {
        Self {
            cache_max_entries: default_cache_max_entries(),
            model_cache_dir: None,
        }
    }
}

impl EmbeddingSettings {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.cache_max_entries == 0 {
            return Err("cache_max_entries must be greater than 0".to_string());
        }
        if self.cache_max_entries > 100_000 {
            return Err("cache_max_entries should not exceed 100,000".to_string());
        }

        // Validate model_cache_dir if specified
        if let Some(ref path) = self.model_cache_dir {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    return Err(format!(
                        "Parent directory for model_cache_dir does not exist: {}",
                        parent.display()
                    ));
                }
            }
        }

        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_EMBEDDING_CACHE_MAX_ENTRIES") {
            if let Ok(parsed) = val.parse() {
                self.cache_max_entries = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_EMBEDDING_MODEL_CACHE_DIR") {
            self.model_cache_dir = Some(PathBuf::from(val));
        }
    }
}

/// LSP (Language Server Protocol) configuration settings
///
/// These settings configure the daemon's LSP integration for code intelligence
/// features. LSP servers are started for active projects and provide enrichment
/// data like references, type information, and import resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspSettings {
    /// User PATH for finding language servers
    /// CLI stores user's PATH here so daemon can find servers
    #[serde(default)]
    pub user_path: Option<String>,

    /// Maximum number of LSP servers per active project
    #[serde(default = "default_max_servers_per_project")]
    pub max_servers_per_project: usize,

    /// Auto-start LSP servers when project becomes active
    #[serde(default = "default_true")]
    pub auto_start_on_activation: bool,

    /// Delay in seconds before stopping servers after project deactivation
    #[serde(default = "default_deactivation_delay")]
    pub deactivation_delay_secs: u64,

    /// Enable caching of LSP enrichment query results
    #[serde(default = "default_true")]
    pub enable_enrichment_cache: bool,

    /// TTL in seconds for cached enrichment data
    #[serde(default = "default_cache_ttl")]
    pub cache_ttl_secs: u64,

    /// Timeout in seconds for LSP server startup
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Timeout in seconds for individual LSP requests
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Interval in seconds between health checks
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval_secs: u64,

    /// Maximum restart attempts before marking server unavailable
    #[serde(default = "default_max_restart_attempts")]
    pub max_restart_attempts: u32,

    /// Backoff multiplier for restart delays
    #[serde(default = "default_backoff_multiplier")]
    pub restart_backoff_multiplier: f64,

    /// Enable auto-restart of failed servers
    #[serde(default = "default_true")]
    pub enable_auto_restart: bool,

    /// Stability period in seconds before resetting restart count
    #[serde(default = "default_stability_reset")]
    pub stability_reset_secs: u64,
}

fn default_max_servers_per_project() -> usize { 3 }
fn default_true() -> bool { true }
fn default_deactivation_delay() -> u64 { 60 }
fn default_cache_ttl() -> u64 { 300 }
fn default_startup_timeout() -> u64 { 30 }
fn default_request_timeout() -> u64 { 10 }
fn default_health_check_interval() -> u64 { 60 }
fn default_max_restart_attempts() -> u32 { 3 }
fn default_backoff_multiplier() -> f64 { 2.0 }
fn default_stability_reset() -> u64 { 3600 }

impl Default for LspSettings {
    fn default() -> Self {
        Self {
            user_path: None,
            max_servers_per_project: default_max_servers_per_project(),
            auto_start_on_activation: default_true(),
            deactivation_delay_secs: default_deactivation_delay(),
            enable_enrichment_cache: default_true(),
            cache_ttl_secs: default_cache_ttl(),
            startup_timeout_secs: default_startup_timeout(),
            request_timeout_secs: default_request_timeout(),
            health_check_interval_secs: default_health_check_interval(),
            max_restart_attempts: default_max_restart_attempts(),
            restart_backoff_multiplier: default_backoff_multiplier(),
            enable_auto_restart: default_true(),
            stability_reset_secs: default_stability_reset(),
        }
    }
}

impl LspSettings {
    /// Validate LSP configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.max_servers_per_project == 0 {
            return Err("max_servers_per_project must be greater than 0".to_string());
        }
        if self.cache_ttl_secs == 0 {
            return Err("cache_ttl_secs must be greater than 0".to_string());
        }
        if self.startup_timeout_secs == 0 {
            return Err("startup_timeout_secs must be greater than 0".to_string());
        }
        if self.request_timeout_secs == 0 {
            return Err("request_timeout_secs must be greater than 0".to_string());
        }
        if self.health_check_interval_secs == 0 {
            return Err("health_check_interval_secs must be greater than 0".to_string());
        }
        if self.restart_backoff_multiplier < 1.0 {
            return Err("restart_backoff_multiplier must be >= 1.0".to_string());
        }
        Ok(())
    }
}

/// Tree-sitter grammar configuration for dynamic grammar loading
///
/// These settings configure how tree-sitter grammars are loaded at runtime.
/// Dynamic loading allows adding language support without recompiling the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarConfig {
    /// Directory for storing grammar shared libraries
    /// Default: ~/.workspace-qdrant/grammars
    #[serde(default = "default_grammar_cache_dir")]
    pub cache_dir: PathBuf,

    /// List of required language grammars for daemon startup
    /// Daemon verifies these exist on startup
    #[serde(default = "default_required_grammars")]
    pub required: Vec<String>,

    /// Auto-download missing grammars from download sources
    #[serde(default = "default_true")]
    pub auto_download: bool,

    /// Expected tree-sitter ABI version for grammar compatibility
    #[serde(default = "default_tree_sitter_version")]
    pub tree_sitter_version: String,

    /// Base URL template for grammar downloads
    /// Supports {language}, {version}, {platform}, {ext} placeholders
    #[serde(default = "default_download_base_url")]
    pub download_base_url: String,

    /// Verify checksums of downloaded grammars
    #[serde(default = "default_true")]
    pub verify_checksums: bool,

    /// Load grammars on first use instead of at startup
    #[serde(default = "default_true")]
    pub lazy_loading: bool,

    /// Interval in hours to check for grammar updates
    /// Default: 168 (weekly)
    #[serde(default = "default_grammar_check_interval")]
    pub check_interval_hours: u32,
}

fn default_grammar_check_interval() -> u32 {
    168 // Weekly
}

fn default_grammar_cache_dir() -> PathBuf {
    // Use ~ which will be expanded by unified_config
    PathBuf::from("~/.workspace-qdrant/grammars")
}

fn default_required_grammars() -> Vec<String> {
    vec![
        "rust".to_string(),
        "python".to_string(),
        "javascript".to_string(),
        "typescript".to_string(),
        "go".to_string(),
        "java".to_string(),
        "c".to_string(),
        "cpp".to_string(),
    ]
}

fn default_tree_sitter_version() -> String {
    "0.24".to_string()
}

fn default_download_base_url() -> String {
    "https://github.com/tree-sitter/tree-sitter-{language}/releases/download/v{version}/tree-sitter-{language}-{platform}.{ext}".to_string()
}

impl Default for GrammarConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_grammar_cache_dir(),
            required: default_required_grammars(),
            auto_download: default_true(),
            tree_sitter_version: default_tree_sitter_version(),
            download_base_url: default_download_base_url(),
            verify_checksums: default_true(),
            lazy_loading: default_true(),
            check_interval_hours: default_grammar_check_interval(),
        }
    }
}

impl GrammarConfig {
    /// Validate grammar configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.tree_sitter_version.is_empty() {
            return Err("tree_sitter_version cannot be empty".to_string());
        }
        if self.download_base_url.is_empty() && self.auto_download {
            return Err("download_base_url required when auto_download is enabled".to_string());
        }
        Ok(())
    }

    /// Get the expanded cache directory (with ~ and env vars expanded)
    pub fn expanded_cache_dir(&self) -> PathBuf {
        let path_str = self.cache_dir.to_string_lossy().into_owned();
        let expanded = shellexpand::tilde(&path_str);
        PathBuf::from(expanded.into_owned())
    }
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

    /// Inter-item delay in ms during warmup (default: 200, normal: 50)
    #[serde(default = "default_warmup_inter_item_delay_ms")]
    pub warmup_inter_item_delay_ms: u64,

    /// Batch size for startup recovery enqueuing (default: 50)
    #[serde(default = "default_startup_enqueue_batch_size")]
    pub startup_enqueue_batch_size: usize,

    /// Delay in ms between enqueue batches (default: 100)
    #[serde(default = "default_startup_enqueue_batch_delay_ms")]
    pub startup_enqueue_batch_delay_ms: u64,
}

fn default_warmup_delay_secs() -> u64 { 5 }
fn default_warmup_window_secs() -> u64 { 30 }
fn default_warmup_max_concurrent_embeddings() -> usize { 1 }
fn default_warmup_inter_item_delay_ms() -> u64 { 200 }
fn default_startup_enqueue_batch_size() -> usize { 50 }
fn default_startup_enqueue_batch_delay_ms() -> u64 { 100 }

impl Default for StartupConfig {
    fn default() -> Self {
        Self {
            warmup_delay_secs: default_warmup_delay_secs(),
            warmup_window_secs: default_warmup_window_secs(),
            warmup_max_concurrent_embeddings: default_warmup_max_concurrent_embeddings(),
            warmup_inter_item_delay_ms: default_warmup_inter_item_delay_ms(),
            startup_enqueue_batch_size: default_startup_enqueue_batch_size(),
            startup_enqueue_batch_delay_ms: default_startup_enqueue_batch_delay_ms(),
        }
    }
}

impl StartupConfig {
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

        if let Ok(val) = env::var("WQM_STARTUP_INTER_ITEM_DELAY_MS") {
            if let Ok(parsed) = val.parse() {
                self.warmup_inter_item_delay_ms = parsed;
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
    }

    /// Create from environment variables only
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env_overrides();
        config
    }
}

/// Resource limits configuration section
///
/// Controls how the daemon manages system resources to be a good neighbor.
/// Four levels: OS priority, processing pacing, embedding concurrency, memory pressure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimitsConfig {
    /// Unix nice level for the daemon process (-20 highest priority, 19 lowest)
    /// Default: 10 (low priority - daemon should yield to interactive processes)
    #[serde(default = "default_nice_level")]
    pub nice_level: i32,

    /// Delay in milliseconds between processing items (breathing room)
    /// Default: 50ms
    #[serde(default = "default_inter_item_delay_ms")]
    pub inter_item_delay_ms: u64,

    /// Maximum concurrent embedding operations (semaphore on ONNX ops)
    /// Default: 2
    #[serde(default = "default_max_concurrent_embeddings")]
    pub max_concurrent_embeddings: usize,

    /// Pause processing when system memory usage exceeds this percentage
    /// Default: 70
    #[serde(default = "default_max_memory_percent")]
    pub max_memory_percent: u8,

    /// Number of ONNX intra-op threads per embedding session.
    /// Default: 2 (sufficient for all-MiniLM-L6-v2, leaves CPU for other work)
    #[serde(default = "default_onnx_intra_threads")]
    pub onnx_intra_threads: usize,
}

fn default_nice_level() -> i32 { 10 }
fn default_inter_item_delay_ms() -> u64 { 50 }
fn default_max_concurrent_embeddings() -> usize { 1 }
fn default_max_memory_percent() -> u8 { 70 }
fn default_onnx_intra_threads() -> usize { 2 }

impl Default for ResourceLimitsConfig {
    fn default() -> Self {
        Self {
            nice_level: default_nice_level(),
            inter_item_delay_ms: default_inter_item_delay_ms(),
            max_concurrent_embeddings: default_max_concurrent_embeddings(),
            max_memory_percent: default_max_memory_percent(),
            onnx_intra_threads: default_onnx_intra_threads(),
        }
    }
}

impl ResourceLimitsConfig {
    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.nice_level < -20 || self.nice_level > 19 {
            return Err("nice_level must be between -20 and 19".to_string());
        }
        if self.inter_item_delay_ms > 5000 {
            return Err("inter_item_delay_ms should not exceed 5000".to_string());
        }
        if self.max_concurrent_embeddings == 0 || self.max_concurrent_embeddings > 8 {
            return Err("max_concurrent_embeddings must be between 1 and 8".to_string());
        }
        if self.max_memory_percent < 20 || self.max_memory_percent > 95 {
            return Err("max_memory_percent must be between 20 and 95".to_string());
        }
        if self.onnx_intra_threads == 0 || self.onnx_intra_threads > 16 {
            return Err("onnx_intra_threads must be between 1 and 16".to_string());
        }
        Ok(())
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        use std::env;

        if let Ok(val) = env::var("WQM_RESOURCE_NICE_LEVEL") {
            if let Ok(parsed) = val.parse() {
                self.nice_level = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_RESOURCE_INTER_ITEM_DELAY_MS") {
            if let Ok(parsed) = val.parse() {
                self.inter_item_delay_ms = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS") {
            if let Ok(parsed) = val.parse() {
                self.max_concurrent_embeddings = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_RESOURCE_MAX_MEMORY_PERCENT") {
            if let Ok(parsed) = val.parse() {
                self.max_memory_percent = parsed;
            }
        }

        if let Ok(val) = env::var("WQM_RESOURCE_ONNX_INTRA_THREADS") {
            if let Ok(parsed) = val.parse() {
                self.onnx_intra_threads = parsed;
            }
        }
    }
}

/// Update channel for daemon self-updates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum UpdateChannel {
    /// Stable releases only (default)
    #[default]
    Stable,
    /// Beta releases for testing
    Beta,
    /// Development builds (may be unstable)
    Dev,
}

/// Daemon self-update configuration
///
/// Controls how the daemon checks for and applies updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatesConfig {
    /// Check for updates on daemon startup
    #[serde(default = "default_true")]
    pub auto_check: bool,

    /// Update channel to follow
    #[serde(default)]
    pub channel: UpdateChannel,

    /// Only notify about updates, don't auto-install
    /// When true, updates are announced but not automatically applied
    #[serde(default = "default_true")]
    pub notify_only: bool,

    /// Interval in hours to check for updates (when auto_check is true)
    /// Default: 24 (daily)
    #[serde(default = "default_update_check_interval")]
    pub check_interval_hours: u32,
}

fn default_update_check_interval() -> u32 {
    24 // Daily
}

impl Default for UpdatesConfig {
    fn default() -> Self {
        Self {
            auto_check: default_true(),
            channel: UpdateChannel::default(),
            notify_only: default_true(),
            check_interval_hours: default_update_check_interval(),
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
    /// Embedding generation configuration
    #[serde(default)]
    pub embedding: EmbeddingSettings,
    /// LSP (Language Server Protocol) integration configuration
    #[serde(default)]
    pub lsp: LspSettings,
    /// Tree-sitter grammar configuration for dynamic loading
    #[serde(default)]
    pub grammars: GrammarConfig,
    /// Daemon self-update configuration
    #[serde(default)]
    pub updates: UpdatesConfig,
    /// Resource limits configuration
    #[serde(default)]
    pub resource_limits: ResourceLimitsConfig,
    /// Startup warmup configuration (Task 577)
    #[serde(default)]
    pub startup: StartupConfig,
    /// Daemon endpoint configuration for service discovery
    #[serde(default)]
    pub daemon_endpoint: DaemonEndpointConfig,
}

/// Daemon endpoint configuration for service discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonEndpointConfig {
    /// Daemon host address
    pub host: String,
    /// gRPC service port
    pub grpc_port: u16,
    /// Health check endpoint path
    pub health_endpoint: String,
    /// Optional authentication token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_token: Option<String>,
}

impl Default for DaemonEndpointConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            grpc_port: 50051,
            health_endpoint: "/health".to_string(),
            auth_token: None,
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        DaemonConfig::from(&*yaml_defaults::DEFAULT_YAML_CONFIG)
    }
}

impl From<&YamlConfig> for DaemonConfig {
    fn from(yaml: &YamlConfig) -> Self {
        Self {
            log_file: None,
            max_concurrent_tasks: Some(yaml.performance.max_concurrent_tasks),
            default_timeout_ms: Some(yaml.performance.default_timeout_ms() as usize),
            enable_preemption: yaml.performance.enable_preemption,
            chunk_size: yaml.performance.chunk_size,
            log_level: "info".to_string(),
            auto_ingestion: AutoIngestionConfig {
                enabled: yaml.auto_ingestion.enabled,
                auto_create_watches: yaml.auto_ingestion.auto_create_watches,
                include_common_files: yaml.auto_ingestion.include_common_files,
                include_source_files: yaml.auto_ingestion.include_source_files,
                target_collection_suffix: "scratchbook".to_string(),
                max_files_per_batch: yaml.auto_ingestion.max_files_per_batch,
                batch_delay_seconds: 2.0,
                max_file_size_mb: 50,
                recursive_depth: 5,
                debounce_seconds: yaml.auto_ingestion.debounce_seconds(),
            },
            project_path: None,
            qdrant: StorageConfig {
                url: yaml.qdrant.url.clone(),
                api_key: yaml.qdrant.api_key.clone(),
                timeout_ms: yaml.qdrant.timeout,
                pool_size: yaml.qdrant.pool.max_connections,
                dense_vector_size: yaml.qdrant.default_collection.vector_size,
                ..StorageConfig::default()
            },
            logging: LoggingConfig::default(),
            queue_processor: QueueProcessorSettings {
                batch_size: yaml.queue_processor.batch_size,
                poll_interval_ms: yaml.queue_processor.poll_interval_ms,
                max_retries: yaml.queue_processor.max_retries,
                retry_delays_seconds: default_retry_delays_seconds(),
                target_throughput: yaml.queue_processor.target_throughput,
                enable_metrics: yaml.queue_processor.enable_metrics,
            },
            monitoring: MonitoringConfig::default(),
            git: GitConfig {
                enable_branch_detection: yaml.git.track_branch_lifecycle,
                cache_ttl_seconds: yaml.git.branch_scan_interval_seconds,
            },
            observability: ObservabilityConfig {
                collection_interval: yaml.observability.collection_interval_secs(),
                metrics: MetricsConfig {
                    enabled: yaml.observability.metrics.enabled,
                },
                telemetry: TelemetryConfig {
                    enabled: yaml.observability.telemetry.enabled,
                    history_retention: yaml.observability.telemetry.history_retention,
                    cpu_usage: yaml.observability.telemetry.cpu_usage,
                    memory_usage: yaml.observability.telemetry.memory_usage,
                    latency: yaml.observability.telemetry.latency,
                    queue_depth: yaml.observability.telemetry.queue_depth,
                    throughput: yaml.observability.telemetry.throughput,
                },
            },
            embedding: EmbeddingSettings {
                cache_max_entries: yaml.embedding.cache_max_entries,
                model_cache_dir: yaml.embedding.model_cache_dir.as_ref().map(PathBuf::from),
            },
            lsp: LspSettings {
                user_path: yaml.lsp.user_path.clone(),
                max_servers_per_project: yaml.lsp.max_servers_per_project,
                auto_start_on_activation: yaml.lsp.auto_start_on_activation,
                deactivation_delay_secs: yaml.lsp.deactivation_delay_secs,
                enable_enrichment_cache: yaml.lsp.enable_enrichment_cache,
                cache_ttl_secs: yaml.lsp.cache_ttl_secs,
                startup_timeout_secs: yaml.lsp.startup_timeout_secs,
                request_timeout_secs: yaml.lsp.request_timeout_secs,
                health_check_interval_secs: yaml.lsp.health_check_interval_secs,
                max_restart_attempts: yaml.lsp.max_restart_attempts,
                restart_backoff_multiplier: yaml.lsp.restart_backoff_multiplier,
                enable_auto_restart: true,
                stability_reset_secs: 3600,
            },
            grammars: GrammarConfig {
                cache_dir: PathBuf::from(&yaml.grammars.cache_dir),
                required: yaml.grammars.required.clone(),
                auto_download: yaml.grammars.auto_download,
                tree_sitter_version: yaml.grammars.tree_sitter_version.clone(),
                download_base_url: yaml.grammars.download_base_url.clone(),
                verify_checksums: yaml.grammars.verify_checksums,
                lazy_loading: yaml.grammars.lazy_loading,
                check_interval_hours: yaml.grammars.check_interval_hours,
            },
            updates: UpdatesConfig {
                auto_check: yaml.updates.auto_check,
                channel: match yaml.updates.channel.as_str() {
                    "beta" => UpdateChannel::Beta,
                    "dev" => UpdateChannel::Dev,
                    _ => UpdateChannel::Stable,
                },
                notify_only: yaml.updates.notify_only,
                check_interval_hours: yaml.updates.check_interval_hours,
            },
            resource_limits: ResourceLimitsConfig::default(),
            startup: StartupConfig::default(),
            daemon_endpoint: DaemonEndpointConfig {
                host: yaml.grpc.host.clone(),
                grpc_port: yaml.grpc.port,
                health_endpoint: "/health".to_string(),
                auth_token: None,
            },
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
    /// Queue backpressure threshold
    pub queue_backpressure_threshold: Option<i64>,
    /// Resource limits for daemon processing
    pub resource_limits: ResourceLimitsConfig,
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
            queue_backpressure_threshold: Some(1000), // Default backpressure threshold
            resource_limits: daemon_config.resource_limits,
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
            queue_backpressure_threshold: Some(1000),
            resource_limits: ResourceLimitsConfig::default(),
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
        assert_eq!(config.git.cache_ttl_seconds, 5); // from YAML branch_scan_interval_seconds
        // Embedding settings defaults
        assert_eq!(config.embedding.cache_max_entries, 1000);
        assert!(config.embedding.model_cache_dir.is_none());
    }

    #[test]
    fn test_embedding_settings_defaults() {
        let settings = EmbeddingSettings::default();
        assert_eq!(settings.cache_max_entries, 1000);
        assert!(settings.model_cache_dir.is_none());
    }

    #[test]
    fn test_embedding_settings_validation() {
        let mut settings = EmbeddingSettings::default();

        // Valid settings
        assert!(settings.validate().is_ok());

        // Invalid cache_max_entries
        settings.cache_max_entries = 0;
        assert!(settings.validate().is_err());
        settings.cache_max_entries = 100_001;
        assert!(settings.validate().is_err());
        settings.cache_max_entries = 1000;

        // Valid with custom cache dir (parent exists)
        settings.model_cache_dir = Some(PathBuf::from("/tmp/test_cache"));
        assert!(settings.validate().is_ok());

        // Reset for other tests
        settings.model_cache_dir = None;
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_lsp_settings_defaults() {
        let settings = LspSettings::default();
        assert!(settings.user_path.is_none());
        assert_eq!(settings.max_servers_per_project, 3);
        assert!(settings.auto_start_on_activation);
        assert_eq!(settings.deactivation_delay_secs, 60);
        assert!(settings.enable_enrichment_cache);
        assert_eq!(settings.cache_ttl_secs, 300);
        assert_eq!(settings.startup_timeout_secs, 30);
        assert_eq!(settings.request_timeout_secs, 10);
        assert_eq!(settings.health_check_interval_secs, 60);
        assert_eq!(settings.max_restart_attempts, 3);
        assert_eq!(settings.restart_backoff_multiplier, 2.0);
        assert!(settings.enable_auto_restart);
        assert_eq!(settings.stability_reset_secs, 3600);
    }

    #[test]
    fn test_lsp_settings_validation() {
        let mut settings = LspSettings::default();

        // Valid settings
        assert!(settings.validate().is_ok());

        // Invalid max_servers_per_project
        settings.max_servers_per_project = 0;
        assert!(settings.validate().is_err());
        settings.max_servers_per_project = 3;

        // Invalid cache_ttl_secs
        settings.cache_ttl_secs = 0;
        assert!(settings.validate().is_err());
        settings.cache_ttl_secs = 300;

        // Invalid startup_timeout_secs
        settings.startup_timeout_secs = 0;
        assert!(settings.validate().is_err());
        settings.startup_timeout_secs = 30;

        // Invalid request_timeout_secs
        settings.request_timeout_secs = 0;
        assert!(settings.validate().is_err());
        settings.request_timeout_secs = 10;

        // Invalid health_check_interval_secs
        settings.health_check_interval_secs = 0;
        assert!(settings.validate().is_err());
        settings.health_check_interval_secs = 60;

        // Invalid restart_backoff_multiplier
        settings.restart_backoff_multiplier = 0.5;
        assert!(settings.validate().is_err());
        settings.restart_backoff_multiplier = 2.0;

        // All valid again
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_lsp_settings_serialization() {
        let settings = LspSettings {
            user_path: Some("/usr/local/bin".to_string()),
            max_servers_per_project: 5,
            ..Default::default()
        };

        // Serialize to JSON
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("\"max_servers_per_project\":5"));
        assert!(json.contains("\"/usr/local/bin\""));

        // Deserialize back
        let deserialized: LspSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.user_path, Some("/usr/local/bin".to_string()));
        assert_eq!(deserialized.max_servers_per_project, 5);
    }

    #[test]
    fn test_grammar_config_defaults() {
        let config = GrammarConfig::default();
        assert_eq!(config.cache_dir, PathBuf::from("~/.workspace-qdrant/grammars"));
        assert!(config.required.contains(&"rust".to_string()));
        assert!(config.required.contains(&"python".to_string()));
        assert!(config.auto_download);
        assert_eq!(config.tree_sitter_version, "0.24");
        assert!(config.verify_checksums);
        assert!(config.lazy_loading);
    }

    #[test]
    fn test_grammar_config_validation() {
        let mut config = GrammarConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid tree_sitter_version
        config.tree_sitter_version = String::new();
        assert!(config.validate().is_err());
        config.tree_sitter_version = "0.24".to_string();

        // Invalid download_base_url when auto_download enabled
        config.download_base_url = String::new();
        assert!(config.validate().is_err());
        config.auto_download = false;
        assert!(config.validate().is_ok()); // Empty URL ok when auto_download disabled
    }

    #[test]
    fn test_grammar_config_expanded_cache_dir() {
        let config = GrammarConfig::default();
        let expanded = config.expanded_cache_dir();
        // Should expand ~ to home directory
        assert!(!expanded.to_string_lossy().contains('~'));
        assert!(expanded.to_string_lossy().ends_with("grammars"));
    }

    #[test]
    fn test_grammar_config_serialization() {
        let config = GrammarConfig {
            cache_dir: PathBuf::from("/custom/grammars"),
            required: vec!["rust".to_string(), "python".to_string()],
            auto_download: false,
            tree_sitter_version: "0.24".to_string(),
            download_base_url: "https://example.com".to_string(),
            verify_checksums: true,
            lazy_loading: true,
            check_interval_hours: 168, // Weekly
        };

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"/custom/grammars\""));
        assert!(json.contains("\"auto_download\":false"));

        // Deserialize back
        let deserialized: GrammarConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.cache_dir, PathBuf::from("/custom/grammars"));
        assert!(!deserialized.auto_download);
    }

    #[test]
    fn test_daemon_config_includes_grammars() {
        let config = DaemonConfig::default();
        // Verify grammars field exists and has defaults
        assert!(!config.grammars.required.is_empty());
        assert!(config.grammars.auto_download);
    }

    #[test]
    fn test_resource_limits_config_defaults() {
        let config = ResourceLimitsConfig::default();
        assert_eq!(config.nice_level, 10);
        assert_eq!(config.inter_item_delay_ms, 50);
        assert_eq!(config.max_concurrent_embeddings, 1);
        assert_eq!(config.max_memory_percent, 70);
        assert_eq!(config.onnx_intra_threads, 2);
    }

    #[test]
    fn test_resource_limits_config_validation() {
        let mut config = ResourceLimitsConfig::default();

        // Valid settings
        assert!(config.validate().is_ok());

        // Invalid nice_level (too low)
        config.nice_level = -21;
        assert!(config.validate().is_err());
        // Invalid nice_level (too high)
        config.nice_level = 20;
        assert!(config.validate().is_err());
        config.nice_level = 10;

        // Invalid inter_item_delay_ms (too high)
        config.inter_item_delay_ms = 5001;
        assert!(config.validate().is_err());
        config.inter_item_delay_ms = 50;

        // Invalid max_concurrent_embeddings (zero)
        config.max_concurrent_embeddings = 0;
        assert!(config.validate().is_err());
        // Invalid max_concurrent_embeddings (too high)
        config.max_concurrent_embeddings = 9;
        assert!(config.validate().is_err());
        config.max_concurrent_embeddings = 2;

        // Invalid max_memory_percent (too low)
        config.max_memory_percent = 19;
        assert!(config.validate().is_err());
        // Invalid max_memory_percent (too high)
        config.max_memory_percent = 96;
        assert!(config.validate().is_err());
        config.max_memory_percent = 70;

        // Valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_config_boundary_values() {
        // Test boundary values are accepted
        let mut config = ResourceLimitsConfig::default();

        config.nice_level = -20;
        assert!(config.validate().is_ok());
        config.nice_level = 19;
        assert!(config.validate().is_ok());

        config.inter_item_delay_ms = 0;
        assert!(config.validate().is_ok());
        config.inter_item_delay_ms = 5000;
        assert!(config.validate().is_ok());

        config.max_concurrent_embeddings = 1;
        assert!(config.validate().is_ok());
        config.max_concurrent_embeddings = 8;
        assert!(config.validate().is_ok());

        config.max_memory_percent = 20;
        assert!(config.validate().is_ok());
        config.max_memory_percent = 95;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_resource_limits_env_overrides() {
        let mut config = ResourceLimitsConfig::default();

        // Set environment variables
        std::env::set_var("WQM_RESOURCE_NICE_LEVEL", "5");
        std::env::set_var("WQM_RESOURCE_INTER_ITEM_DELAY_MS", "100");
        std::env::set_var("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS", "4");
        std::env::set_var("WQM_RESOURCE_MAX_MEMORY_PERCENT", "80");

        config.apply_env_overrides();

        assert_eq!(config.nice_level, 5);
        assert_eq!(config.inter_item_delay_ms, 100);
        assert_eq!(config.max_concurrent_embeddings, 4);
        assert_eq!(config.max_memory_percent, 80);

        // Clean up
        std::env::remove_var("WQM_RESOURCE_NICE_LEVEL");
        std::env::remove_var("WQM_RESOURCE_INTER_ITEM_DELAY_MS");
        std::env::remove_var("WQM_RESOURCE_MAX_CONCURRENT_EMBEDDINGS");
        std::env::remove_var("WQM_RESOURCE_MAX_MEMORY_PERCENT");
    }

    #[test]
    fn test_daemon_config_includes_resource_limits() {
        let config = DaemonConfig::default();
        assert_eq!(config.resource_limits.nice_level, 10);
        assert_eq!(config.resource_limits.inter_item_delay_ms, 50);
        assert_eq!(config.resource_limits.max_concurrent_embeddings, 1);
        assert_eq!(config.resource_limits.max_memory_percent, 70);
        assert_eq!(config.resource_limits.onnx_intra_threads, 2);
    }

    #[test]
    fn test_resource_limits_serialization() {
        let config = ResourceLimitsConfig {
            nice_level: 5,
            inter_item_delay_ms: 100,
            max_concurrent_embeddings: 4,
            max_memory_percent: 80,
            onnx_intra_threads: 2,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"nice_level\":5"));
        assert!(json.contains("\"max_memory_percent\":80"));

        let deserialized: ResourceLimitsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nice_level, 5);
        assert_eq!(deserialized.max_memory_percent, 80);
    }

    #[test]
    fn test_startup_config_defaults() {
        let config = StartupConfig::default();
        assert_eq!(config.warmup_delay_secs, 5);
        assert_eq!(config.warmup_window_secs, 30);
        assert_eq!(config.warmup_max_concurrent_embeddings, 1);
        assert_eq!(config.warmup_inter_item_delay_ms, 200);
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
        std::env::set_var("WQM_STARTUP_INTER_ITEM_DELAY_MS", "300");
        std::env::set_var("WQM_STARTUP_ENQUEUE_BATCH_SIZE", "100");
        std::env::set_var("WQM_STARTUP_ENQUEUE_BATCH_DELAY_MS", "200");

        config.apply_env_overrides();

        assert_eq!(config.warmup_delay_secs, 10);
        assert_eq!(config.warmup_window_secs, 60);
        assert_eq!(config.warmup_max_concurrent_embeddings, 2);
        assert_eq!(config.warmup_inter_item_delay_ms, 300);
        assert_eq!(config.startup_enqueue_batch_size, 100);
        assert_eq!(config.startup_enqueue_batch_delay_ms, 200);

        // Clean up
        std::env::remove_var("WQM_STARTUP_WARMUP_DELAY_SECS");
        std::env::remove_var("WQM_STARTUP_WARMUP_WINDOW_SECS");
        std::env::remove_var("WQM_STARTUP_MAX_CONCURRENT_EMBEDDINGS");
        std::env::remove_var("WQM_STARTUP_INTER_ITEM_DELAY_MS");
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
    fn test_startup_config_serialization() {
        let config = StartupConfig {
            warmup_delay_secs: 8,
            warmup_window_secs: 45,
            warmup_max_concurrent_embeddings: 1,
            warmup_inter_item_delay_ms: 250,
            startup_enqueue_batch_size: 75,
            startup_enqueue_batch_delay_ms: 150,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"warmup_delay_secs\":8"));
        assert!(json.contains("\"warmup_window_secs\":45"));

        let deserialized: StartupConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.warmup_delay_secs, 8);
        assert_eq!(deserialized.warmup_window_secs, 45);
    }

    #[test]
    fn test_daemon_config_includes_startup() {
        let config = DaemonConfig::default();
        assert_eq!(config.startup.warmup_delay_secs, 5);
        assert_eq!(config.startup.warmup_window_secs, 30);
    }
}
