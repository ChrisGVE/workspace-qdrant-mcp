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
mod url_ingestion;

// Re-export all public types for backward compatibility
pub use code_intelligence::{GrammarConfig, LspSettings};
pub use embedding::EmbeddingSettings;
pub use ingestion::{AutoIngestionConfig, IngestionLimitsConfig};
pub use integration::{GitConfig, UpdateChannel, UpdatesConfig};
pub use observability::{
    LoggingConfig, MetricsConfig, MonitoringConfig, ObservabilityConfig, OtlpExportConfig,
    OtlpProtocol, PrometheusExportConfig, TelemetryConfig,
};
pub use processing::{QueueProcessorSettings, StartupConfig};
pub use resource_limits::{detect_physical_cores, ResourceLimitsConfig};
pub use url_ingestion::UrlIngestionConfig;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::storage::StorageConfig;
use processing::default_retry_delays_seconds;
use wqm_common::paths::MountMap;
use wqm_common::yaml_defaults::{self, YamlConfig, YamlMountEntry};

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

impl DaemonEndpointConfig {
    /// Validate configuration settings.
    ///
    /// - `host` must be non-empty.
    /// - `grpc_port` must be non-zero.
    /// - `health_endpoint` must be empty or start with `/`.
    pub fn validate(&self) -> Result<(), String> {
        if self.host.trim().is_empty() {
            return Err("host must not be empty".to_string());
        }
        if self.grpc_port == 0 {
            return Err("grpc_port must be non-zero".to_string());
        }
        if !self.health_endpoint.is_empty() && !self.health_endpoint.starts_with('/') {
            return Err("health_endpoint must be empty or start with '/'".to_string());
        }
        Ok(())
    }
}

/// Complete daemon configuration that matches the TOML structure.
///
/// `#[serde(default)]` at the struct level makes a partial/minimal config file
/// parse cleanly: any field absent from the YAML falls back to the value in
/// [`DaemonConfig::default()`] (which is derived from the embedded
/// `default_configuration.yaml`). This is the intended "user config overlays
/// the built-in defaults" behavior — `load_config_file` deserializes the user
/// YAML directly into this struct (no separate merge step), so without this a
/// minimal config (e.g. only `qdrant`/`embedding`) crashed the daemon on parse
/// with "missing field `enable_preemption`" — fatal on `docker restart`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
    /// Per-extension ingestion size limits (Task 14)
    #[serde(default)]
    pub ingestion_limits: IngestionLimitsConfig,
    /// URL ingestion fetch limits and SSRF policy (T5).
    #[serde(default)]
    pub url_ingestion: UrlIngestionConfig,
    /// Host-↔-container mount-map entries (see docs/specs/16-path-abstraction.md §5).
    ///
    /// Raw form preserved for serde round-trip; the validated [`MountMap`]
    /// is obtained via [`DaemonConfig::build_mount_map`]. Default is the
    /// empty list, which yields an identity map.
    #[serde(default)]
    pub mounts: Vec<YamlMountEntry>,
    /// Memexd control-port override (spec 16 §10.1).
    ///
    /// When `Some(p)`, memexd binds the cross-process single-instance
    /// lock to `127.0.0.1:p` instead of the built-in default 7799. The
    /// CLI flag `--control-port` and the `WQM_CONTROL_PORT` env var
    /// take precedence over this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_port: Option<u16>,
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
            ingestion_limits: IngestionLimitsConfig::default(),
            url_ingestion: build_url_ingestion_config(yaml),
            mounts: yaml.mounts.clone(),
            control_port: None,
        }
    }
}

fn build_url_ingestion_config(yaml: &YamlConfig) -> UrlIngestionConfig {
    UrlIngestionConfig {
        connect_timeout_secs: yaml.url_ingestion.connect_timeout_secs,
        read_timeout_secs: yaml.url_ingestion.read_timeout_secs,
        max_redirects: yaml.url_ingestion.max_redirects,
        max_body_bytes: yaml.url_ingestion.max_body_bytes,
        allow_private_networks: yaml.url_ingestion.allow_private_networks,
        allowed_content_types: yaml.url_ingestion.allowed_content_types.clone(),
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
        max_concurrent_items: yaml.queue_processor.max_concurrent_items,
    }
}

fn build_git_config(yaml: &YamlConfig) -> GitConfig {
    GitConfig {
        enable_branch_detection: yaml.git.track_branch_lifecycle,
        cache_ttl_seconds: yaml.git.branch_scan_interval_seconds,
    }
}

fn build_observability_config(yaml: &YamlConfig) -> ObservabilityConfig {
    let y_telemetry = &yaml.observability.telemetry;
    let protocol =
        OtlpProtocol::parse(&y_telemetry.otlp.protocol).unwrap_or(OtlpProtocol::HttpProtobuf);

    ObservabilityConfig {
        collection_interval: yaml.observability.collection_interval_secs(),
        metrics: MetricsConfig {
            enabled: yaml.observability.metrics.enabled,
        },
        telemetry: TelemetryConfig {
            enabled: y_telemetry.enabled,
            history_retention: y_telemetry.history_retention,
            cpu_usage: y_telemetry.cpu_usage,
            memory_usage: y_telemetry.memory_usage,
            latency: y_telemetry.latency,
            queue_depth: y_telemetry.queue_depth,
            throughput: y_telemetry.throughput,
            service_name: y_telemetry.service_name.clone(),
            prometheus: PrometheusExportConfig {
                enabled: y_telemetry.prometheus.enabled,
                port: y_telemetry.prometheus.port,
                bind: y_telemetry.prometheus.bind.clone(),
            },
            otlp: OtlpExportConfig {
                enabled: y_telemetry.otlp.enabled,
                endpoint: y_telemetry.otlp.endpoint.clone(),
                protocol,
                sample_rate: y_telemetry.otlp.sample_rate,
                headers: y_telemetry.otlp.headers.clone(),
            },
        },
    }
}

fn build_embedding_settings(yaml: &YamlConfig) -> EmbeddingSettings {
    EmbeddingSettings {
        cache_max_entries: yaml.embedding.cache_max_entries,
        model_cache_dir: yaml.embedding.model_cache_dir.as_ref().map(PathBuf::from),
        provider: yaml.embedding.provider.clone(),
        model: yaml.embedding.model.clone(),
        base_url: yaml.embedding.base_url.clone(),
        remote_batch_size: yaml.embedding.remote_batch_size,
        api_key_env_var: yaml.embedding.api_key_env_var.clone(),
        output_dim: yaml.embedding.output_dim,
        health_probe_cache_secs: yaml.embedding.health_probe_cache_secs,
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
        warmup_grace_secs: yaml.lsp.warmup_grace_secs,
        health_check_interval_secs: yaml.lsp.health_check_interval_secs,
        max_restart_attempts: yaml.lsp.max_restart_attempts,
        restart_backoff_multiplier: yaml.lsp.restart_backoff_multiplier,
        enable_auto_restart: true,
        stability_reset_secs: 3600,
        idle_timeout_secs: yaml.lsp.idle_timeout_secs,
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
        grammar_idle_timeout_secs: yaml.grammars.grammar_idle_timeout_secs,
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
        linux_idle_source: yaml.resource_limits.linux_idle_source.clone(),
        linux_idle_load_threshold: yaml.resource_limits.linux_idle_load_threshold,
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

    /// Validate all sub-configuration sections, returning the first error encountered.
    pub fn validate(&self) -> Result<(), String> {
        self.queue_processor
            .validate()
            .map_err(|e| format!("queue_processor: {e}"))?;
        self.monitoring
            .validate()
            .map_err(|e| format!("monitoring: {e}"))?;
        self.git.validate().map_err(|e| format!("git: {e}"))?;
        self.observability
            .validate()
            .map_err(|e| format!("observability: {e}"))?;
        self.embedding
            .validate()
            .map_err(|e| format!("embedding: {e}"))?;
        self.lsp.validate().map_err(|e| format!("lsp: {e}"))?;
        self.grammars
            .validate()
            .map_err(|e| format!("grammars: {e}"))?;
        self.updates
            .validate()
            .map_err(|e| format!("updates: {e}"))?;
        // resource_limits uses 0 as sentinel for auto-detect; resolve
        // hardware-specific defaults on a temporary clone before validating.
        let mut resolved_limits = self.resource_limits.clone();
        resolved_limits.resolve_auto_values();
        resolved_limits
            .validate()
            .map_err(|e| format!("resource_limits: {e}"))?;
        self.startup
            .validate()
            .map_err(|e| format!("startup: {e}"))?;
        self.daemon_endpoint
            .validate()
            .map_err(|e| format!("daemon_endpoint: {e}"))?;
        self.ingestion_limits
            .validate()
            .map_err(|e| format!("ingestion_limits: {e}"))?;
        self.auto_ingestion
            .validate()
            .map_err(|e| format!("auto_ingestion: {e}"))?;
        self.validate_mounts().map_err(|e| format!("mounts: {e}"))?;
        Ok(())
    }

    /// Construct the [`MountMap`] declared by this configuration.
    ///
    /// Tilde expansion, absolute-path validation, `..`-rejection, and
    /// duplicate host/container detection are all delegated to
    /// [`MountMap::from_yaml_entries`]. An empty `mounts` list yields the
    /// identity map (spec 16 §5.5).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`wqm_common::paths::PathError`] formatted
    /// as a string when any entry fails canonicalisation or validation.
    pub fn build_mount_map(&self) -> Result<MountMap, String> {
        MountMap::from_yaml_entries(&self.mounts).map_err(|e| e.to_string())
    }

    /// Validate the declared mount-map entries without keeping the result.
    ///
    /// Called by [`DaemonConfig::validate`]. The actual [`MountMap`]
    /// instance is constructed once at process startup via
    /// [`DaemonConfig::build_mount_map`] and is immutable for the process
    /// lifetime (spec 16 §5.3).
    pub fn validate_mounts(&self) -> Result<(), String> {
        // Empty list is the identity map — explicitly always valid.
        if self.mounts.is_empty() {
            return Ok(());
        }
        // Discarding the returned MountMap is intentional: this call site
        // only checks well-formedness. Live construction happens elsewhere.
        let _ = self.build_mount_map()?;
        Ok(())
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
    /// Max items dispatched concurrently within a single batch (default 1 =
    /// byte-identical sequential behavior). Plumbed into
    /// `UnifiedProcessorConfig::max_concurrent_items`.
    pub queue_max_concurrent_items: Option<usize>,
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
            queue_max_concurrent_items: Some(daemon_config.queue_processor.max_concurrent_items),
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
            queue_max_concurrent_items: Some(1),
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
    fn test_minimal_config_parses_with_defaults() {
        // Regression guard: a minimal user config (only qdrant/embedding) must
        // deserialize cleanly into DaemonConfig, with all omitted fields falling
        // back to defaults. Before `#[serde(default)]` on DaemonConfig this
        // crashed with "missing field `enable_preemption`" — fatal on restart.
        let minimal = r#"
qdrant:
  url: http://qdrant:6334
embedding:
  provider: fastembed
  output_dim: 384
"#;
        let config: DaemonConfig =
            serde_yaml_ng::from_str(minimal).expect("minimal config must parse");
        // Omitted scalars fall back to embedded defaults.
        assert!(config.enable_preemption);
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.log_level, "info");
        // Present fields are honored.
        assert_eq!(config.qdrant.url, "http://qdrant:6334");
    }

    #[test]
    fn test_empty_config_parses_to_defaults() {
        // An entirely empty document must parse to defaults (never crash).
        let config: DaemonConfig =
            serde_yaml_ng::from_str("{}").expect("empty config must parse");
        assert!(config.enable_preemption);
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

    // ── DaemonEndpointConfig::validate() tests ──────────────────────────────

    #[test]
    fn test_daemon_endpoint_config_validate_default_ok() {
        let config = DaemonEndpointConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_daemon_endpoint_config_validate_rejects_empty_host() {
        let empty = DaemonEndpointConfig {
            host: "".to_string(),
            ..DaemonEndpointConfig::default()
        };
        assert!(empty.validate().is_err());

        let whitespace = DaemonEndpointConfig {
            host: "   ".to_string(),
            ..DaemonEndpointConfig::default()
        };
        assert!(whitespace.validate().is_err());
    }

    #[test]
    fn test_daemon_endpoint_config_validate_rejects_zero_grpc_port() {
        let config = DaemonEndpointConfig {
            grpc_port: 0,
            ..DaemonEndpointConfig::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("grpc_port"));
    }

    #[test]
    fn test_daemon_endpoint_config_validate_rejects_bad_health_endpoint() {
        let config = DaemonEndpointConfig {
            health_endpoint: "health".to_string(),
            ..DaemonEndpointConfig::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("health_endpoint"));
    }

    #[test]
    fn test_daemon_endpoint_config_validate_accepts_empty_health_endpoint() {
        let config = DaemonEndpointConfig {
            health_endpoint: "".to_string(),
            ..DaemonEndpointConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    // ── DaemonConfig::validate() tests ──────────────────────────────────────

    #[test]
    fn test_daemon_config_validate_default_ok() {
        assert!(DaemonConfig::default().validate().is_ok());
    }

    #[test]
    fn test_daemon_config_validate_propagates_queue_error() {
        let mut config = DaemonConfig::default();
        config.queue_processor.batch_size = 0;
        let result = config.validate();
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("queue_processor:"),
            "expected 'queue_processor:' in '{msg}'"
        );
    }

    #[test]
    fn test_daemon_config_validate_propagates_observability_error() {
        let mut config = DaemonConfig::default();
        config.observability.collection_interval = 0;
        let result = config.validate();
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("observability:"),
            "expected 'observability:' in '{msg}'"
        );
    }

    #[test]
    fn test_daemon_config_validate_short_circuits_on_first_error() {
        let mut config = DaemonConfig::default();
        // queue_processor is first in the chain
        config.queue_processor.batch_size = 0;
        config.observability.collection_interval = 0;
        let result = config.validate();
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("queue_processor:"),
            "expected 'queue_processor:' in '{msg}'"
        );
    }
    // ── Mount-map (T3) tests ────────────────────────────────────────────────

    fn mk_mount(host: &str, container: &str) -> YamlMountEntry {
        YamlMountEntry {
            host: host.to_string(),
            container: container.to_string(),
        }
    }

    #[test]
    fn mounts_empty_yields_identity_map() {
        // T3.10: default DaemonConfig has no mount entries → identity map.
        let cfg = DaemonConfig::default();
        assert!(cfg.mounts.is_empty(), "default mounts list must be empty");
        let map = cfg.build_mount_map().expect("identity map always builds");
        assert!(
            map.is_identity(),
            "empty mounts list must yield identity map"
        );
        assert_eq!(map.len(), 0);
        // Validation succeeds.
        assert!(cfg.validate_mounts().is_ok());
    }

    #[test]
    fn mounts_valid_entries_with_tilde_expansion_load() {
        // T3.11: a leading `~` is expanded once on load.
        let home = dirs::home_dir().expect("HOME must be set");
        let home_str = home.to_str().expect("home path must be UTF-8");

        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![
            mk_mount("/Users/chris/dev", "/Users/chris/dev"),
            mk_mount("~/reference", "/mnt/reference"),
        ];

        let map = cfg.build_mount_map().expect("valid mounts must load");
        assert_eq!(map.len(), 2);
        assert!(!map.is_identity());

        // Round-trip the host of the tilde entry via from_canonical lookup.
        let expanded = format!("{home_str}/reference");
        let canon = wqm_common::paths::CanonicalPath::from_user_input(&expanded)
            .expect("expanded path must canonicalise");
        // Indirect check: the canonical form contains the expanded home.
        assert!(canon.as_str().starts_with(home_str));
        assert!(cfg.validate_mounts().is_ok());
    }

    #[test]
    fn mounts_overlapping_entries_allowed() {
        // T3.12: overlap (one entry's host is a prefix of another) is allowed —
        // longest-prefix-wins is well-defined.
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![
            mk_mount("/Users/chris", "/mnt/user"),
            mk_mount("/Users/chris/dev", "/mnt/dev"),
        ];
        let map = cfg.build_mount_map().expect("overlap must be allowed");
        assert_eq!(map.len(), 2);
        assert!(cfg.validate_mounts().is_ok());
    }

    #[test]
    fn mounts_duplicate_host_prefix_rejected() {
        // T3.13: two entries with identical canonical host → reject.
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![
            mk_mount("/Users/chris", "/mnt/a"),
            mk_mount("/Users/chris", "/mnt/b"),
        ];
        let err = cfg
            .build_mount_map()
            .expect_err("duplicate host must error");
        assert!(
            err.contains("duplicate host"),
            "expected duplicate-host message in '{err}'"
        );
        // validate_mounts surfaces the same error.
        assert!(cfg.validate_mounts().is_err());
    }

    #[test]
    fn mounts_duplicate_container_prefix_rejected() {
        // T3.14: duplicate container canonical form → reject.
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![
            mk_mount("/Users/chris/a", "/mnt/shared"),
            mk_mount("/Users/chris/b", "/mnt/shared"),
        ];
        let err = cfg
            .build_mount_map()
            .expect_err("duplicate container must error");
        assert!(
            err.contains("duplicate container"),
            "expected duplicate-container message in '{err}'"
        );
        assert!(cfg.validate_mounts().is_err());
    }

    #[test]
    fn mounts_relative_host_path_rejected() {
        // T3.15: a relative host path is rejected.
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![mk_mount("relative/host", "/mnt/x")];
        assert!(cfg.build_mount_map().is_err());
        assert!(cfg.validate_mounts().is_err());
    }

    #[test]
    fn mounts_relative_container_path_rejected() {
        // T3.16: a relative container path is rejected.
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![mk_mount("/Users/chris/dev", "relative/container")];
        assert!(cfg.build_mount_map().is_err());
        assert!(cfg.validate_mounts().is_err());
    }

    #[test]
    fn mounts_parent_dir_segment_rejected() {
        // Spec §3.1 rule 4: `..` in either host or container is rejected.
        let mut cfg_host = DaemonConfig::default();
        cfg_host.mounts = vec![mk_mount("/Users/chris/../other", "/mnt/x")];
        assert!(cfg_host.build_mount_map().is_err());

        let mut cfg_container = DaemonConfig::default();
        cfg_container.mounts = vec![mk_mount("/Users/chris/dev", "/mnt/../other")];
        assert!(cfg_container.build_mount_map().is_err());
    }

    #[test]
    fn mounts_validate_chain_surfaces_mounts_prefix() {
        // T3.9 / T3.17: DaemonConfig::validate() routes mount errors with the
        // `mounts:` prefix so log readers can identify the failing section.
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![mk_mount("relative/path", "/mnt/x")];
        let err = cfg
            .validate()
            .expect_err("invalid mount must fail validate");
        assert!(
            err.starts_with("mounts:"),
            "expected 'mounts:' prefix in '{err}'"
        );
    }

    #[test]
    fn mounts_yaml_round_trip_preserves_entries() {
        // Acceptance (T3.19): full config-load round-trip survives the new
        // section without loss.
        let mut original = DaemonConfig::default();
        original.mounts = vec![
            mk_mount("/Users/chris/dev", "/Users/chris/dev"),
            mk_mount("/Volumes/External/books", "/mnt/external-books"),
        ];
        let yaml = serde_yaml_ng::to_string(&original).expect("serialise");
        let restored: DaemonConfig = serde_yaml_ng::from_str(&yaml).expect("deserialise");
        assert_eq!(restored.mounts, original.mounts);
        assert_eq!(restored.mounts.len(), 2);
        // Validation still passes after round-trip.
        assert!(restored.validate().is_ok());
    }

    #[test]
    fn mounts_acceptance_edge_cases() {
        // T3.20: edge cases bundled into a single acceptance test.

        // 1. Missing `mounts:` section: serialise a default and strip the
        //    mounts line so the deserialiser must rely on serde defaults.
        let full = serde_yaml_ng::to_string(&DaemonConfig::default()).expect("serialise default");
        let no_section: String = full
            .lines()
            .filter(|l| !l.starts_with("mounts:"))
            .map(|l| format!("{l}\n"))
            .collect();
        let cfg: DaemonConfig = serde_yaml_ng::from_str(&no_section)
            .expect("DaemonConfig must deserialise without mounts section");
        assert!(cfg.mounts.is_empty());
        assert!(cfg.build_mount_map().unwrap().is_identity());

        // 2. Explicit empty list embedded in an otherwise complete config.
        let mut empty = DaemonConfig::default();
        empty.mounts.clear();
        let yaml_empty = serde_yaml_ng::to_string(&empty).expect("serialise");
        let cfg: DaemonConfig =
            serde_yaml_ng::from_str(&yaml_empty).expect("empty mounts list parses");
        assert!(cfg.mounts.is_empty());

        // 3. Mirror mount — identical host and container.
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![mk_mount("/Users/chris/dev", "/Users/chris/dev")];
        let map = cfg.build_mount_map().expect("mirror is valid");
        assert_eq!(map.len(), 1);
        assert!(!map.is_identity(), "mirror is not the identity map");

        // 4. Three-entry mount set survives end-to-end validate().
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![
            mk_mount("/Users/chris/dev", "/Users/chris/dev"),
            mk_mount("/Volumes/External/books", "/mnt/external-books"),
            mk_mount("/Users/chris/reference", "/mnt/reference"),
        ];
        assert!(cfg.validate().is_ok());
    }
}
