//! Configuration management
//!
//! This module contains configuration management for the priority processing engine.
//! Domain-specific configs are organized into submodules and re-exported here.

mod code_intelligence;
mod embedding;
mod ingestion;
mod integration;
mod observability;
mod processing;
mod resource_limits;

// Re-export all public types for backward compatibility
pub use code_intelligence::{GrammarConfig, LspSettings};
pub use embedding::EmbeddingSettings;
pub use ingestion::AutoIngestionConfig;
pub use integration::{GitConfig, UpdateChannel, UpdatesConfig};
pub use observability::{
    LoggingConfig, MetricsConfig, MonitoringConfig, ObservabilityConfig, TelemetryConfig,
};
pub use processing::{QueueProcessorSettings, StartupConfig};
pub use resource_limits::{detect_physical_cores, ResourceLimitsConfig};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::storage::StorageConfig;
use processing::default_retry_delays_seconds;
use wqm_common::yaml_defaults::{self, YamlConfig};

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
            auto_ingestion: build_auto_ingestion_config(yaml),
            project_path: None,
            qdrant: build_storage_config(yaml),
            logging: LoggingConfig::default(),
            queue_processor: build_queue_processor_settings(yaml),
            monitoring: MonitoringConfig::default(),
            git: build_git_config(yaml),
            observability: build_observability_config(yaml),
            embedding: build_embedding_settings(yaml),
            lsp: build_lsp_settings(yaml),
            grammars: build_grammar_config(yaml),
            updates: build_updates_config(yaml),
            resource_limits: build_resource_limits_config(yaml),
            startup: StartupConfig::default(),
            daemon_endpoint: build_daemon_endpoint_config(yaml),
        }
    }
}

fn build_auto_ingestion_config(yaml: &YamlConfig) -> AutoIngestionConfig {
    AutoIngestionConfig {
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
    }
}

fn build_storage_config(yaml: &YamlConfig) -> StorageConfig {
    StorageConfig {
        url: yaml.qdrant.url.clone(),
        api_key: yaml.qdrant.api_key.clone(),
        timeout_ms: yaml.qdrant.timeout,
        pool_size: yaml.qdrant.pool.max_connections,
        dense_vector_size: yaml.qdrant.default_collection.vector_size,
        ..StorageConfig::default()
    }
}

fn build_queue_processor_settings(yaml: &YamlConfig) -> QueueProcessorSettings {
    QueueProcessorSettings {
        batch_size: yaml.queue_processor.batch_size,
        poll_interval_ms: yaml.queue_processor.poll_interval_ms,
        max_retries: yaml.queue_processor.max_retries,
        retry_delays_seconds: default_retry_delays_seconds(),
        target_throughput: yaml.queue_processor.target_throughput,
        enable_metrics: yaml.queue_processor.enable_metrics,
    }
}

fn build_git_config(yaml: &YamlConfig) -> GitConfig {
    GitConfig {
        enable_branch_detection: yaml.git.track_branch_lifecycle,
        cache_ttl_seconds: yaml.git.branch_scan_interval_seconds,
    }
}

fn build_observability_config(yaml: &YamlConfig) -> ObservabilityConfig {
    ObservabilityConfig {
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
    }
}

fn build_embedding_settings(yaml: &YamlConfig) -> EmbeddingSettings {
    EmbeddingSettings {
        cache_max_entries: yaml.embedding.cache_max_entries,
        model_cache_dir: yaml.embedding.model_cache_dir.as_ref().map(PathBuf::from),
    }
}

fn build_lsp_settings(yaml: &YamlConfig) -> LspSettings {
    LspSettings {
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
    }
}

fn build_grammar_config(yaml: &YamlConfig) -> GrammarConfig {
    GrammarConfig {
        cache_dir: PathBuf::from(&yaml.grammars.cache_dir),
        required: yaml.grammars.required.clone(),
        auto_download: yaml.grammars.auto_download,
        tree_sitter_version: yaml.grammars.tree_sitter_version.clone(),
        download_base_url: yaml.grammars.download_base_url.clone(),
        verify_checksums: yaml.grammars.verify_checksums,
        lazy_loading: yaml.grammars.lazy_loading,
        check_interval_hours: yaml.grammars.check_interval_hours,
        idle_update_check_enabled: yaml.grammars.idle_update_check_enabled,
        idle_update_check_delay_secs: yaml.grammars.idle_update_check_delay_secs,
    }
}

fn build_updates_config(yaml: &YamlConfig) -> UpdatesConfig {
    UpdatesConfig {
        auto_check: yaml.updates.auto_check,
        channel: match yaml.updates.channel.as_str() {
            "beta" => UpdateChannel::Beta,
            "dev" => UpdateChannel::Dev,
            _ => UpdateChannel::Stable,
        },
        notify_only: yaml.updates.notify_only,
        check_interval_hours: yaml.updates.check_interval_hours,
    }
}

fn build_resource_limits_config(yaml: &YamlConfig) -> ResourceLimitsConfig {
    ResourceLimitsConfig {
        nice_level: yaml.resource_limits.nice_level,
        inter_item_delay_ms: yaml.resource_limits.inter_item_delay_ms,
        max_concurrent_embeddings: yaml.resource_limits.max_concurrent_embeddings,
        max_memory_percent: yaml.resource_limits.max_memory_percent,
        onnx_intra_threads: yaml.resource_limits.onnx_intra_threads,
        idle_threshold_secs: yaml.resource_limits.idle_threshold_secs,
        idle_confirmation_secs: yaml.resource_limits.idle_confirmation_secs,
        ramp_up_step_secs: yaml.resource_limits.ramp_up_step_secs,
        ramp_down_step_secs: yaml.resource_limits.ramp_down_step_secs,
        burst_hold_secs: yaml.resource_limits.burst_hold_secs,
        burst_concurrency_multiplier: yaml.resource_limits.burst_concurrency_multiplier,
        burst_inter_item_delay_ms: yaml.resource_limits.burst_inter_item_delay_ms,
        cpu_pressure_threshold: yaml.resource_limits.cpu_pressure_threshold,
        idle_poll_interval_secs: yaml.resource_limits.idle_poll_interval_secs,
        active_concurrency_multiplier: yaml.resource_limits.active_concurrency_multiplier,
        active_inter_item_delay_ms: yaml.resource_limits.active_inter_item_delay_ms,
    }
}

fn build_daemon_endpoint_config(yaml: &YamlConfig) -> DaemonEndpointConfig {
    DaemonEndpointConfig {
        host: yaml.grpc.host.clone(),
        grpc_port: yaml.grpc.port,
        health_endpoint: "/health".to_string(),
        auth_token: None,
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
    fn test_daemon_config_includes_grammars() {
        let config = DaemonConfig::default();
        // YAML default: required is empty (grammars downloaded on first use)
        assert!(config.grammars.required.is_empty());
        assert!(config.grammars.auto_download);
    }

    #[test]
    fn test_daemon_config_includes_resource_limits() {
        let config = DaemonConfig::default();
        assert_eq!(config.resource_limits.nice_level, 10);
        assert_eq!(config.resource_limits.inter_item_delay_ms, 50);
        assert_eq!(
            config.resource_limits.max_concurrent_embeddings, 0,
            "default is 0 (auto-detect)"
        );
        assert_eq!(config.resource_limits.max_memory_percent, 70);
        assert_eq!(
            config.resource_limits.onnx_intra_threads, 0,
            "default is 0 (auto-detect)"
        );
    }

    #[test]
    fn test_daemon_config_includes_startup() {
        let config = DaemonConfig::default();
        assert_eq!(config.startup.warmup_delay_secs, 5);
        assert_eq!(config.startup.warmup_window_secs, 30);
    }
}
