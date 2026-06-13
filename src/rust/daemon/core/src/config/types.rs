//! Configuration struct/enum definitions for the daemon.
//!
//! Holds the typed config VIEW: [`DaemonEndpointConfig`], [`DaemonConfig`]
//! (with its serde defaults and secret-redacting `Debug`), and the processing
//! [`Config`]. Construction from YAML lives in [`super::build`]; validation and
//! mount-map handling live in [`super::validate`].

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::storage::StorageConfig;
use wqm_common::yaml_defaults::{self, YamlMountEntry};

use super::{
    AutoIngestionConfig, ConceptConfig, EmbeddingSettings, GitConfig, GrammarConfig,
    GraphRagConfig, IngestionLimitsConfig, LoggingConfig, LspSettings, MonitoringConfig,
    NarrativeConfig, ObservabilityConfig, QueueHealthConfig, QueueProcessorSettings,
    ResourceLimitsConfig, StartupConfig, UpdatesConfig, UrlIngestionConfig,
};

/// Daemon endpoint configuration for service discovery.
#[derive(Clone, Serialize, Deserialize)]
pub struct DaemonEndpointConfig {
    /// Daemon host address
    pub host: String,
    /// gRPC service port
    pub grpc_port: u16,
    /// Health check endpoint path
    pub health_endpoint: String,
    /// Optional authentication token.
    ///
    /// Never serialized (WI-g1): `skip_serializing_if` still wrote the token
    /// when present. Use `skip_serializing` so `save_config` never persists the
    /// secret; deserialization still loads an operator-provided value.
    #[serde(skip_serializing)]
    pub auth_token: Option<String>,
}

/// Manual `Debug` (WI-g2): `auth_token` is rendered as `Some("[REDACTED]")`/`None`
/// so the secret never appears in `{:?}` / `{:#?}` output or any log line that
/// debug-prints the config. All other fields print verbatim.
impl std::fmt::Debug for DaemonEndpointConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DaemonEndpointConfig")
            .field("host", &self.host)
            .field("grpc_port", &self.grpc_port)
            .field("health_endpoint", &self.health_endpoint)
            .field(
                "auth_token",
                &self.auth_token.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
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
    /// Queue-health monitoring thresholds (#133 `[queue_health]`)
    #[serde(default)]
    pub queue_health: QueueHealthConfig,
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
    /// Concept-edge emission thresholds (IMPLEMENTS_CONCEPT / COVERS_TOPIC)
    #[serde(default)]
    pub concept: ConceptConfig,
    /// Cross-boundary graph-RAG traversal caps and fusion (search.graph_rag.*)
    #[serde(default)]
    pub graph_rag: GraphRagConfig,
    /// Code-relationship graph backend selection (`graph.*`).
    #[serde(default)]
    pub graph: crate::graph::GraphConfig,
    /// Narrative extraction thresholds and safety limits.
    #[serde(default)]
    pub narrative: NarrativeConfig,
    /// URL ingestion fetch limits and SSRF policy (T5).
    #[serde(default)]
    pub url_ingestion: UrlIngestionConfig,
    /// Host-↔-container mount-map entries (see docs/specs/16-path-abstraction.md §5).
    ///
    /// Raw form preserved for serde round-trip; the validated [`MountMap`]
    /// is obtained via [`DaemonConfig::build_mount_map`]. Default is the
    /// empty list, which yields an identity map.
    ///
    /// [`MountMap`]: wqm_common::paths::MountMap
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

/// Processing engine configuration.
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
    /// Code-relationship graph backend selection (`graph.*`).
    pub graph: crate::graph::GraphConfig,
    /// Cross-boundary graph-RAG traversal fan-out caps (`search.graph_rag.*`).
    pub graph_rag: GraphRagConfig,
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
            graph: daemon_config.graph,
            graph_rag: daemon_config.graph_rag,
        }
    }
}

impl Config {
    /// Create new configuration with defaults.
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
            graph: crate::graph::GraphConfig::default(),
            graph_rag: GraphRagConfig::default(),
        }
    }

    /// Enable daemon mode with silent operation.
    pub fn daemon_mode() -> Self {
        Self::new()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
