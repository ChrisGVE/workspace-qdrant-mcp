//! Embedded YAML defaults parsed at startup
//!
//! Provides `DEFAULT_YAML` (the raw YAML string) and `DEFAULT_YAML_CONFIG`
//! (a lazily-parsed struct). Both daemon and CLI import these to derive their
//! defaults from the single source of truth: `assets/default_configuration.yaml`.

use serde::Deserialize;
use std::sync::LazyLock;

/// The raw default configuration YAML, embedded at compile time.
pub static DEFAULT_YAML: &str =
    include_str!("../../../../assets/default_configuration.yaml");

/// Parsed default configuration, initialized once on first access.
pub static DEFAULT_YAML_CONFIG: LazyLock<YamlConfig> = LazyLock::new(|| {
    serde_yml::from_str(DEFAULT_YAML).expect("default_configuration.yaml must parse successfully")
});

// =============================================================================
// Duration string parsing
// =============================================================================

/// Parse a duration string like "30s", "5m", "1h" into milliseconds.
///
/// Supported suffixes: `ms`, `s`, `m`, `h`.
/// Bare numbers are treated as seconds.
pub fn parse_duration_to_ms(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    if let Some(num) = s.strip_suffix("ms") {
        return num.trim().parse::<u64>().ok();
    }
    if let Some(num) = s.strip_suffix('s') {
        return num.trim().parse::<u64>().ok().map(|n| n * 1000);
    }
    if let Some(num) = s.strip_suffix('m') {
        return num.trim().parse::<u64>().ok().map(|n| n * 60_000);
    }
    if let Some(num) = s.strip_suffix('h') {
        return num.trim().parse::<u64>().ok().map(|n| n * 3_600_000);
    }

    // Bare number → seconds
    s.parse::<u64>().ok().map(|n| n * 1000)
}

/// Parse a size string like "50MB", "100KB" into bytes.
pub fn parse_size_to_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    if let Some(num) = s.strip_suffix("GB") {
        return num.trim().parse::<u64>().ok().map(|n| n * 1_073_741_824);
    }
    if let Some(num) = s.strip_suffix("MB") {
        return num.trim().parse::<u64>().ok().map(|n| n * 1_048_576);
    }
    if let Some(num) = s.strip_suffix("KB") {
        return num.trim().parse::<u64>().ok().map(|n| n * 1024);
    }
    if let Some(num) = s.strip_suffix('B') {
        return num.trim().parse::<u64>().ok();
    }

    // Bare number → bytes
    s.parse::<u64>().ok()
}

// =============================================================================
// Serde helpers for duration strings
// =============================================================================

mod duration_serde {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize a duration string (e.g. "30s") into milliseconds (u64).
    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        super::parse_duration_to_ms(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid duration: {}", s)))
    }
}

#[allow(dead_code)]
mod optional_duration_serde {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an optional duration string into `Option<u64>` milliseconds.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            Some(s) => super::parse_duration_to_ms(&s)
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom(format!("invalid duration: {}", s))),
            None => Ok(None),
        }
    }
}

// =============================================================================
// Top-level YAML config struct
// =============================================================================

/// Top-level configuration parsed from `default_configuration.yaml`.
///
/// Only the sections used by daemon and CLI are included. Unknown sections
/// are silently ignored via `#[serde(default)]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlConfig {
    pub qdrant: YamlQdrantConfig,
    pub grpc: YamlGrpcConfig,
    pub auto_ingestion: YamlAutoIngestionConfig,
    pub queue_processor: YamlQueueProcessorConfig,
    pub git: YamlGitConfig,
    pub embedding: YamlEmbeddingConfig,
    pub lsp: YamlLspConfig,
    pub grammars: YamlGrammarsConfig,
    pub updates: YamlUpdatesConfig,
    pub performance: YamlPerformanceConfig,
    pub watching: YamlWatchingConfig,
    pub observability: YamlObservabilityConfig,
}

impl Default for YamlConfig {
    fn default() -> Self {
        // Use sub-struct defaults directly. Do NOT parse DEFAULT_YAML here —
        // that would recurse infinitely because serde's `#[serde(default)]`
        // calls this method for missing fields during parsing.
        Self {
            qdrant: YamlQdrantConfig::default(),
            grpc: YamlGrpcConfig::default(),
            auto_ingestion: YamlAutoIngestionConfig::default(),
            queue_processor: YamlQueueProcessorConfig::default(),
            git: YamlGitConfig::default(),
            embedding: YamlEmbeddingConfig::default(),
            lsp: YamlLspConfig::default(),
            grammars: YamlGrammarsConfig::default(),
            updates: YamlUpdatesConfig::default(),
            performance: YamlPerformanceConfig::default(),
            watching: YamlWatchingConfig::default(),
            observability: YamlObservabilityConfig::default(),
        }
    }
}

// =============================================================================
// Section structs
// =============================================================================

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

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlAutoIngestionConfig {
    pub enabled: bool,
    pub auto_create_watches: bool,
    pub include_common_files: bool,
    pub include_source_files: bool,
    pub max_files_per_batch: usize,
    pub debounce: Option<String>,
}

impl Default for YamlAutoIngestionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_create_watches: true,
            include_common_files: true,
            include_source_files: true,
            max_files_per_batch: 5,
            debounce: Some("10s".to_string()),
        }
    }
}

impl YamlAutoIngestionConfig {
    /// Get debounce as seconds
    pub fn debounce_seconds(&self) -> u64 {
        self.debounce
            .as_deref()
            .and_then(|s| parse_duration_to_ms(s).map(|ms| ms / 1000))
            .unwrap_or(10)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlQueueProcessorConfig {
    pub enabled: bool,
    pub batch_size: i32,
    pub poll_interval_ms: u64,
    pub max_retries: i32,
    pub target_throughput: u64,
    pub enable_metrics: bool,
    pub worker_count: usize,
    pub backpressure_threshold: i64,
}

impl Default for YamlQueueProcessorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_size: 10,
            poll_interval_ms: 500,
            max_retries: 5,
            target_throughput: 1000,
            enable_metrics: true,
            worker_count: 4,
            backpressure_threshold: 1000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlGitConfig {
    pub track_branch_lifecycle: bool,
    pub auto_delete_branch_documents: bool,
    pub branch_scan_interval_seconds: u64,
    pub rename_correlation_timeout_ms: u64,
    pub default_branch_detection: String,
}

impl Default for YamlGitConfig {
    fn default() -> Self {
        Self {
            track_branch_lifecycle: true,
            auto_delete_branch_documents: true,
            branch_scan_interval_seconds: 5,
            rename_correlation_timeout_ms: 500,
            default_branch_detection: "head".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlEmbeddingConfig {
    pub model: String,
    pub enable_sparse_vectors: bool,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub batch_size: usize,
    pub cache_enabled: bool,
    pub cache_max_entries: usize,
    pub model_cache_dir: Option<String>,
}

impl Default for YamlEmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            enable_sparse_vectors: true,
            chunk_size: 384,
            chunk_overlap: 58,
            batch_size: 50,
            cache_enabled: true,
            cache_max_entries: 1000,
            model_cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlLspConfig {
    pub user_path: Option<String>,
    pub max_servers_per_project: usize,
    pub auto_start_on_activation: bool,
    pub deactivation_delay_secs: u64,
    pub enable_enrichment_cache: bool,
    pub cache_ttl_secs: u64,
    pub startup_timeout_secs: u64,
    pub request_timeout_secs: u64,
    pub health_check_interval_secs: u64,
    pub max_restart_attempts: u32,
    pub restart_backoff_multiplier: f64,
}

impl Default for YamlLspConfig {
    fn default() -> Self {
        Self {
            user_path: None,
            max_servers_per_project: 3,
            auto_start_on_activation: true,
            deactivation_delay_secs: 60,
            enable_enrichment_cache: true,
            cache_ttl_secs: 300,
            startup_timeout_secs: 30,
            request_timeout_secs: 10,
            health_check_interval_secs: 60,
            max_restart_attempts: 3,
            restart_backoff_multiplier: 2.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlGrammarsConfig {
    pub cache_dir: String,
    pub required: Vec<String>,
    pub auto_download: bool,
    pub tree_sitter_version: String,
    pub download_base_url: String,
    pub verify_checksums: bool,
    pub lazy_loading: bool,
    pub check_interval_hours: u32,
}

impl Default for YamlGrammarsConfig {
    fn default() -> Self {
        Self {
            cache_dir: "~/.workspace-qdrant/grammars".to_string(),
            required: vec![
                "rust".into(),
                "python".into(),
                "javascript".into(),
                "typescript".into(),
                "go".into(),
                "java".into(),
                "c".into(),
                "cpp".into(),
            ],
            auto_download: true,
            tree_sitter_version: "0.24".to_string(),
            download_base_url: "https://github.com/tree-sitter/tree-sitter-{language}/releases/download/v{version}/tree-sitter-{language}-{platform}.{ext}".to_string(),
            verify_checksums: true,
            lazy_loading: true,
            check_interval_hours: 168,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlUpdatesConfig {
    pub auto_check: bool,
    pub channel: String,
    pub notify_only: bool,
    pub check_interval_hours: u32,
}

impl Default for YamlUpdatesConfig {
    fn default() -> Self {
        Self {
            auto_check: true,
            channel: "stable".to_string(),
            notify_only: true,
            check_interval_hours: 24,
        }
    }
}

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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_yaml_parses() {
        let config: YamlConfig =
            serde_yml::from_str(DEFAULT_YAML).expect("YAML should parse");
        // Spot-check key values
        assert_eq!(config.qdrant.url, "http://localhost:6333");
        assert_eq!(config.grpc.port, 50051);
        assert_eq!(config.performance.chunk_size, 1000);
        assert_eq!(config.performance.max_concurrent_tasks, 4);
        assert!(config.performance.enable_preemption);
    }

    #[test]
    fn test_lazy_lock_config() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.qdrant.url, "http://localhost:6333");
        assert_eq!(config.grpc.port, 50051);
        assert_eq!(config.grpc.host, "127.0.0.1");
        assert!(config.grpc.enabled);
    }

    #[test]
    fn test_qdrant_timeout_parsed() {
        let config = &*DEFAULT_YAML_CONFIG;
        // YAML says "30s" which should parse to 30000ms
        assert_eq!(config.qdrant.timeout, 30_000);
    }

    #[test]
    fn test_qdrant_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert!(config.qdrant.prefer_grpc);
        assert_eq!(config.qdrant.transport, "grpc");
        assert_eq!(config.qdrant.pool.max_connections, 10);
        assert_eq!(config.qdrant.default_collection.vector_size, 384);
        assert_eq!(config.qdrant.default_collection.hnsw.m, 16);
        assert_eq!(config.qdrant.default_collection.hnsw.ef_construct, 100);
    }

    #[test]
    fn test_grpc_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.grpc.host, "127.0.0.1");
        assert_eq!(config.grpc.port, 50051);
        assert!(config.grpc.enabled);
        assert!(config.grpc.fallback_to_direct);
        assert_eq!(config.grpc.max_retries, 3);
    }

    #[test]
    fn test_auto_ingestion_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert!(config.auto_ingestion.enabled);
        assert!(config.auto_ingestion.auto_create_watches);
        assert_eq!(config.auto_ingestion.max_files_per_batch, 5);
        assert_eq!(config.auto_ingestion.debounce_seconds(), 10);
    }

    #[test]
    fn test_queue_processor_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.queue_processor.batch_size, 10);
        assert_eq!(config.queue_processor.poll_interval_ms, 500);
        assert_eq!(config.queue_processor.max_retries, 5);
        assert_eq!(config.queue_processor.target_throughput, 1000);
        assert!(config.queue_processor.enable_metrics);
        assert_eq!(config.queue_processor.worker_count, 4);
        assert_eq!(config.queue_processor.backpressure_threshold, 1000);
    }

    #[test]
    fn test_git_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert!(config.git.track_branch_lifecycle);
        assert!(config.git.auto_delete_branch_documents);
        assert_eq!(config.git.branch_scan_interval_seconds, 5);
    }

    #[test]
    fn test_embedding_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.embedding.model, "sentence-transformers/all-MiniLM-L6-v2");
        assert!(config.embedding.enable_sparse_vectors);
        assert_eq!(config.embedding.cache_max_entries, 1000);
    }

    #[test]
    fn test_lsp_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.lsp.max_servers_per_project, 3);
        assert!(config.lsp.auto_start_on_activation);
        assert_eq!(config.lsp.deactivation_delay_secs, 60);
        assert_eq!(config.lsp.cache_ttl_secs, 300);
        assert_eq!(config.lsp.startup_timeout_secs, 30);
        assert_eq!(config.lsp.request_timeout_secs, 10);
        assert_eq!(config.lsp.max_restart_attempts, 3);
    }

    #[test]
    fn test_grammars_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.grammars.cache_dir, "~/.workspace-qdrant/grammars");
        assert!(config.grammars.required.contains(&"rust".to_string()));
        assert!(config.grammars.required.contains(&"python".to_string()));
        assert!(config.grammars.auto_download);
        assert_eq!(config.grammars.tree_sitter_version, "0.24");
    }

    #[test]
    fn test_updates_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert!(config.updates.auto_check);
        assert_eq!(config.updates.channel, "stable");
        assert!(config.updates.notify_only);
        assert_eq!(config.updates.check_interval_hours, 24);
    }

    #[test]
    fn test_performance_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.performance.max_concurrent_tasks, 4);
        assert_eq!(config.performance.default_timeout_ms(), 30_000);
        assert!(config.performance.enable_preemption);
        assert_eq!(config.performance.chunk_size, 1000);
    }

    #[test]
    fn test_observability_defaults() {
        let config = &*DEFAULT_YAML_CONFIG;
        assert_eq!(config.observability.collection_interval_secs(), 60);
        assert!(!config.observability.metrics.enabled);
        assert!(!config.observability.telemetry.enabled);
        assert_eq!(config.observability.telemetry.history_retention, 120);
    }

    #[test]
    fn test_parse_duration_to_ms() {
        assert_eq!(parse_duration_to_ms("30s"), Some(30_000));
        assert_eq!(parse_duration_to_ms("5m"), Some(300_000));
        assert_eq!(parse_duration_to_ms("1h"), Some(3_600_000));
        assert_eq!(parse_duration_to_ms("500ms"), Some(500));
        assert_eq!(parse_duration_to_ms("10"), Some(10_000)); // bare number = seconds
        assert_eq!(parse_duration_to_ms(""), None);
    }

    #[test]
    fn test_parse_size_to_bytes() {
        assert_eq!(parse_size_to_bytes("50MB"), Some(50 * 1_048_576));
        assert_eq!(parse_size_to_bytes("100KB"), Some(100 * 1024));
        assert_eq!(parse_size_to_bytes("1GB"), Some(1_073_741_824));
        assert_eq!(parse_size_to_bytes("1024B"), Some(1024));
        assert_eq!(parse_size_to_bytes(""), None);
    }

    #[test]
    fn test_grammar_download_url_is_full_template() {
        // Verify YAML defaults contain the complete URL template with all placeholders,
        // not just a base URL prefix. This prevents grammar downloads from producing
        // incomplete artifact URLs.
        let defaults = YamlGrammarsConfig::default();
        assert!(defaults.download_base_url.contains("{language}"), "Missing {{language}} placeholder");
        assert!(defaults.download_base_url.contains("{version}"), "Missing {{version}} placeholder");
        assert!(defaults.download_base_url.contains("{platform}"), "Missing {{platform}} placeholder");
        assert!(defaults.download_base_url.contains("{ext}"), "Missing {{ext}} placeholder");
    }
}
