//! Infrastructure configuration sections: Qdrant, gRPC, performance, watching,
//! observability, and resource limits.

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

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlWatchingConfig {
    // Watching section may have fields used by daemon only;
    // we only capture what's needed for shared defaults
    pub debounce_ms: Option<u64>,
}

impl Default for YamlWatchingConfig {
    fn default() -> Self {
        Self {
            debounce_ms: None,
        }
    }
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
        }
    }
}

// ── Resource Limits ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlResourceLimitsConfig {
    /// Unix nice level for the daemon process (-20 highest, 19 lowest)
    pub nice_level: i32,
    /// Delay in ms between processing items
    pub inter_item_delay_ms: u64,
    /// Max concurrent embedding operations (0 = auto-detect)
    pub max_concurrent_embeddings: usize,
    /// Pause processing when memory exceeds this %
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
    /// Inter-item delay in burst mode (ms)
    pub burst_inter_item_delay_ms: u64,
    /// CPU load fraction above which burst is suppressed
    pub cpu_pressure_threshold: f64,
    /// How often to poll idle state (seconds)
    pub idle_poll_interval_secs: u64,
    /// Multiplier for active processing mode (user present, queue has work)
    pub active_concurrency_multiplier: f64,
    /// Inter-item delay in active processing mode (ms)
    pub active_inter_item_delay_ms: u64,
}

impl Default for YamlResourceLimitsConfig {
    fn default() -> Self {
        Self {
            nice_level: 10,
            inter_item_delay_ms: 50,
            max_concurrent_embeddings: 0,
            max_memory_percent: 70,
            onnx_intra_threads: 0,
            idle_threshold_secs: 120,
            idle_confirmation_secs: 300,
            ramp_up_step_secs: 120,
            ramp_down_step_secs: 300,
            burst_hold_secs: 600,
            burst_concurrency_multiplier: 2.0,
            burst_inter_item_delay_ms: 0,
            cpu_pressure_threshold: 0.6,
            idle_poll_interval_secs: 5,
            active_concurrency_multiplier: 1.5,
            active_inter_item_delay_ms: 25,
        }
    }
}
