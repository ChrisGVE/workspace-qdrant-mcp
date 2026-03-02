//! Processing and tooling configuration sections: auto-ingestion, queue processor,
//! git, embedding, LSP, grammars, updates, and tagging.

use serde::Deserialize;

use super::parse_duration_to_ms;

// ── Auto Ingestion ──────────────────────────────────────────────────────

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

// ── Queue Processor ─────────────────────────────────────────────────────

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

// ── Git ─────────────────────────────────────────────────────────────────

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

// ── Embedding ───────────────────────────────────────────────────────────

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

// ── LSP ─────────────────────────────────────────────────────────────────

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

// ── Grammars ────────────────────────────────────────────────────────────

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
    pub idle_update_check_enabled: bool,
    pub idle_update_check_delay_secs: u64,
}

impl Default for YamlGrammarsConfig {
    fn default() -> Self {
        Self {
            cache_dir: "~/.workspace-qdrant/grammars".to_string(),
            required: vec![],
            auto_download: true,
            tree_sitter_version: env!("TREE_SITTER_VERSION_MAJOR_MINOR").to_string(),
            download_base_url: "https://github.com/tree-sitter/tree-sitter-{language}/releases/download/v{version}/tree-sitter-{language}-{platform}.{ext}".to_string(),
            verify_checksums: true,
            lazy_loading: true,
            check_interval_hours: 168,
            idle_update_check_enabled: true,
            idle_update_check_delay_secs: 300,
        }
    }
}

// ── Updates ─────────────────────────────────────────────────────────────

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

// ── Tagging ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlTaggingConfig {
    pub tier3: YamlTier3Config,
}

impl Default for YamlTaggingConfig {
    fn default() -> Self {
        Self {
            tier3: YamlTier3Config::default(),
        }
    }
}

/// Provider slot config (reused for primary and fallback).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlProviderSlot {
    pub provider: String,
    pub access_mode: String,
    pub model: String,
    pub api_key_env: String,
    pub base_url: Option<String>,
}

impl Default for YamlProviderSlot {
    fn default() -> Self {
        Self {
            provider: "anthropic".to_string(),
            access_mode: "cli".to_string(),
            model: "claude-haiku-4-5-20251001".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct YamlTier3Config {
    pub enabled: bool,
    pub primary: YamlProviderSlot,
    pub fallback: Option<YamlProviderSlot>,
    pub max_chunks_per_doc: usize,
    pub max_tags_per_chunk: usize,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub rate_limit_rps: u32,
    pub temperature: f64,
    pub total_budget_secs: u64,
    pub max_consecutive_failures: u32,
}

impl Default for YamlTier3Config {
    fn default() -> Self {
        Self {
            enabled: false,
            primary: YamlProviderSlot::default(),
            fallback: None,
            max_chunks_per_doc: 10,
            max_tags_per_chunk: 5,
            timeout_secs: 15,
            max_retries: 2,
            rate_limit_rps: 10,
            temperature: 0.3,
            total_budget_secs: 60,
            max_consecutive_failures: 2,
        }
    }
}
