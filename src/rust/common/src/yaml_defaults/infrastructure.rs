//! Infrastructure configuration sections: Qdrant, gRPC, performance, watching,
//! observability, and resource limits.

use std::collections::HashMap;

use serde::Deserialize;

use super::duration_serde;
use super::parse_duration_to_ms;

// ── Qdrant ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlQdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    #[serde(deserialize_with = "duration_serde::deserialize")]
    pub timeout: u64,
    pub prefer_grpc: bool,
    pub transport: String,
    #[serde(default)]
    pub pool: YamlQdrantPoolConfig,
    #[serde(default)]
    pub default_collection: YamlDefaultCollectionConfig,
}

impl Default for YamlQdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout: 30_000,
            prefer_grpc: true,
            transport: "grpc".to_string(),
            pool: YamlQdrantPoolConfig::default(),
            default_collection: YamlDefaultCollectionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlQdrantPoolConfig {
    pub max_connections: usize,
    pub min_idle_connections: usize,
}

impl Default for YamlQdrantPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_idle_connections: 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlDefaultCollectionConfig {
    pub vector_size: u64,
    pub distance_metric: String,
    #[serde(default)]
    pub hnsw: YamlHnswConfig,
}

impl Default for YamlDefaultCollectionConfig {
    fn default() -> Self {
        Self {
            vector_size: 384,
            distance_metric: "Cosine".to_string(),
            hnsw: YamlHnswConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlHnswConfig {
    pub m: u64,
    pub ef_construct: u64,
}

impl Default for YamlHnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construct: 100,
        }
    }
}

// ── Graph backend ─────────────────────────────────────────────────────────

/// Code-relationship graph backend configuration (`graph.*`).
///
/// Selects the graph store implementation and carries LadybugDB tuning knobs.
/// The core crate maps this into its own `GraphConfig` (common cannot depend
/// on core).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlGraphConfig {
    /// Backend selector: `"sqlite"` (default) or `"ladybug"`.
    pub backend: String,
    /// LadybugDB buffer pool size in bytes (ignored for sqlite).
    pub buffer_pool_size: u64,
    /// LadybugDB max worker threads (ignored for sqlite).
    pub max_threads: u64,
}

impl Default for YamlGraphConfig {
    fn default() -> Self {
        Self {
            backend: "sqlite".to_string(),
            buffer_pool_size: 256 * 1024 * 1024,
            max_threads: 2,
        }
    }
}

// ── gRPC ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlGrpcConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub fallback_to_direct: bool,
    pub max_retries: u32,
}

impl Default for YamlGrpcConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            host: "127.0.0.1".to_string(),
            port: 50051,
            fallback_to_direct: true,
            max_retries: 3,
        }
    }
}

// ── Performance ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlPerformanceConfig {
    pub max_concurrent_tasks: usize,
    #[serde(default = "default_performance_timeout")]
    pub default_timeout: String,
    pub enable_preemption: bool,
    pub chunk_size: usize,
}

fn default_performance_timeout() -> String {
    "30s".to_string()
}

impl Default for YamlPerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 4,
            default_timeout: "30s".to_string(),
            enable_preemption: true,
            chunk_size: 1000,
        }
    }
}

impl YamlPerformanceConfig {
    /// Get default_timeout as milliseconds
    pub fn default_timeout_ms(&self) -> u64 {
        parse_duration_to_ms(&self.default_timeout).unwrap_or(30_000)
    }
}

// ── Watching ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct YamlWatchingConfig {
    // Watching section may have fields used by daemon only;
    // we only capture what's needed for shared defaults
    pub debounce_ms: Option<u64>,
}

// ── Observability ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlObservabilityConfig {
    pub collection_interval: Option<String>,
    #[serde(default)]
    pub metrics: YamlMetricsConfig,
    #[serde(default)]
    pub telemetry: YamlTelemetryConfig,
}

impl Default for YamlObservabilityConfig {
    fn default() -> Self {
        Self {
            collection_interval: Some("60s".to_string()),
            metrics: YamlMetricsConfig::default(),
            telemetry: YamlTelemetryConfig::default(),
        }
    }
}

impl YamlObservabilityConfig {
    /// Get collection_interval as seconds
    pub fn collection_interval_secs(&self) -> u64 {
        self.collection_interval
            .as_deref()
            .and_then(|s| parse_duration_to_ms(s).map(|ms| ms / 1000))
            .unwrap_or(60)
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct YamlMetricsConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlTelemetryConfig {
    pub enabled: bool,
    pub history_retention: usize,
    pub cpu_usage: bool,
    pub memory_usage: bool,
    pub latency: bool,
    pub queue_depth: bool,
    pub throughput: bool,

    /// OpenTelemetry resource `service.name` attribute applied to spans
    /// and metrics exported by the daemon.
    pub service_name: String,

    /// Prometheus pull-based metrics exposition settings.
    #[serde(default)]
    pub prometheus: YamlPrometheusConfig,

    /// OpenTelemetry Protocol push-based export settings (traces + metrics).
    #[serde(default)]
    pub otlp: YamlOtlpConfig,
}

impl Default for YamlTelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            history_retention: 120,
            cpu_usage: true,
            memory_usage: true,
            latency: true,
            queue_depth: true,
            throughput: true,
            service_name: "memexd".to_string(),
            prometheus: YamlPrometheusConfig::default(),
            otlp: YamlOtlpConfig::default(),
        }
    }
}

/// Prometheus HTTP exposition endpoint settings.
///
/// When `enabled` is true, the daemon binds an HTTP server on
/// `{bind}:{port}` and serves Prometheus text-format metrics at `/metrics`
/// plus a `/health` endpoint. Disabled by default to avoid opening ports
/// unexpectedly.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlPrometheusConfig {
    pub enabled: bool,
    pub port: u16,
    pub bind: String,
}

impl Default for YamlPrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // Adjacent to Qdrant gRPC (6334); keeps workspace-qdrant ports
            // clustered under 6333-6337 to avoid collisions with common
            // defaults like 8080/9090.
            port: 6337,
            bind: "0.0.0.0".to_string(),
        }
    }
}

/// OpenTelemetry Protocol push exporter settings for traces (and optionally
/// metrics). Disabled by default; when enabled, spans are batched and shipped
/// to `endpoint` using `protocol`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlOtlpConfig {
    pub enabled: bool,
    pub endpoint: String,
    /// Wire protocol: `http/protobuf` (default) or `grpc`.
    pub protocol: String,
    /// Trace sampler ratio in the closed interval `[0.0, 1.0]`. Metrics
    /// exporters are unaffected by this ratio.
    pub sample_rate: f64,
    /// Additional headers to attach to OTLP requests (e.g. auth tokens).
    pub headers: HashMap<String, String>,
}

impl Default for YamlOtlpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4318".to_string(),
            protocol: "http/protobuf".to_string(),
            sample_rate: 1.0,
            headers: HashMap::new(),
        }
    }
}

// ── Resource Limits ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlResourceLimitsConfig {
    /// Unix nice level for the daemon process (-20 highest, 19 lowest)
    pub nice_level: i32,
    /// Max concurrent embedding operations (0 = auto-detect)
    pub max_concurrent_embeddings: usize,
    /// Pause processing when available memory falls below (100 - this)%
    pub max_memory_percent: u8,
    /// ONNX intra-op threads per embedding session (0 = auto-detect)
    pub onnx_intra_threads: usize,
    /// Seconds of no user input before considering idle
    pub idle_threshold_secs: u64,
    /// Seconds of sustained idle required before the first upward level transition
    pub idle_confirmation_secs: u64,
    /// Seconds to spend at each level during ramp-up (after confirmation)
    pub ramp_up_step_secs: u64,
    /// Seconds of sustained user activity required before each downward level transition
    pub ramp_down_step_secs: u64,
    /// Minimum seconds to hold at Burst before allowing ramp-down
    pub burst_hold_secs: u64,
    /// Multiplier for burst-mode max_concurrent_embeddings (relative to normal)
    pub burst_concurrency_multiplier: f64,
    /// CPU load fraction above which burst is suppressed
    pub cpu_pressure_threshold: f64,
    /// How often to poll idle state (seconds)
    pub idle_poll_interval_secs: u64,
    /// Multiplier for active processing mode (user present, queue has work)
    pub active_concurrency_multiplier: f64,
    /// Linux idle-detection backend (`"none"` or `"proc"`).
    pub linux_idle_source: String,
    /// Normalized load-average threshold for the `/proc` Linux heuristic.
    pub linux_idle_load_threshold: f64,
}

impl Default for YamlResourceLimitsConfig {
    fn default() -> Self {
        Self {
            nice_level: 10,
            max_concurrent_embeddings: 0,
            max_memory_percent: 70,
            onnx_intra_threads: 0,
            idle_threshold_secs: 120,
            idle_confirmation_secs: 300,
            ramp_up_step_secs: 120,
            ramp_down_step_secs: 300,
            burst_hold_secs: 600,
            burst_concurrency_multiplier: 2.0,
            cpu_pressure_threshold: 0.6,
            idle_poll_interval_secs: 5,
            active_concurrency_multiplier: 1.5,
            linux_idle_source: "none".to_string(),
            linux_idle_load_threshold: 0.1,
        }
    }
}

// ── Mounts ──────────────────────────────────────────────────────────────

/// One host ↔ container directory pair, as declared in `config.yaml`.
///
/// Carries unprocessed strings exactly as read from the YAML file. Tilde
/// expansion, absolute-path validation, duplicate detection, and the
/// conversion to [`wqm_common::paths::MountMap`] happen in
/// [`crate::paths::MountMap::new`] or in the daemon's config-load
/// validation step (`DaemonConfig::validate`).
///
/// See `docs/specs/16-path-abstraction.md` §5.1 for the schema and §5.3
/// for the validation rules.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct YamlMountEntry {
    /// Host-side directory (the path as seen by the daemon's running
    /// process). May contain a leading `~` which is expanded on load.
    pub host: String,
    /// Container-side directory (the canonical/storage form). May contain
    /// a leading `~` which is expanded on load.
    pub container: String,
}
