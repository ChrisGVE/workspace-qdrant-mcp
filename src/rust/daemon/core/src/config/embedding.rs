//! Embedding generation configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const FASTEMBED_OUTPUT_DIM: usize = 384;

fn default_cache_max_entries() -> usize {
    1000
}

fn default_provider() -> String {
    "openai_compatible".to_string()
}

fn default_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_base_url() -> String {
    "https://api.openai.com".to_string()
}

fn default_remote_batch_size() -> usize {
    128
}

fn default_api_key_env_var() -> String {
    "OPENAI_API_KEY".to_string()
}

fn default_output_dim() -> usize {
    1536
}

fn default_health_probe_cache_secs() -> u64 {
    60
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

    /// Active dense provider. Valid values: "fastembed", "openai_compatible".
    #[serde(default = "default_provider")]
    pub provider: String,

    /// Model name passed to the provider.
    #[serde(default = "default_model")]
    pub model: String,

    /// Base URL for the OpenAI-compatible endpoint (no trailing slash).
    #[serde(default = "default_base_url")]
    pub base_url: String,

    /// Number of texts per HTTP request for the remote provider.
    /// Has no effect when `provider = "fastembed"`.
    #[serde(default = "default_remote_batch_size")]
    pub remote_batch_size: usize,

    /// Name of the environment variable that holds the API key.
    /// Resolved at daemon startup — never stored in config on disk.
    #[serde(default = "default_api_key_env_var")]
    pub api_key_env_var: String,

    /// Expected output dimensionality of the configured model.
    ///
    /// Authoritative dim for the startup dim-mismatch guard and for
    /// collection recreation during reembed. The runtime
    /// `provider.output_dim()` atomic is informational only — updated by
    /// the probe to reflect confirmed actual dim and used for WARN logging
    /// on drift.
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,

    /// Seconds to cache the embedding provider health probe result.
    /// 0 = no caching (probe on every health check call).
    #[serde(default = "default_health_probe_cache_secs")]
    pub health_probe_cache_secs: u64,

    /// Prefix prepended to DOCUMENT texts before dense embedding (ingestion).
    /// Instruction-tuned retrieval models require it — multilingual-e5 expects
    /// "passage: " (trailing space included). Empty = no prefix. Dense leg
    /// only; sparse/BM25 never sees the prefix.
    #[serde(default)]
    pub document_prefix: String,

    /// Prefix prepended to QUERY texts before dense embedding (the gRPC
    /// EmbedText path used for search queries). multilingual-e5 expects
    /// "query: " (trailing space included). Empty = no prefix.
    #[serde(default)]
    pub query_prefix: String,
}

impl Default for EmbeddingSettings {
    fn default() -> Self {
        Self {
            cache_max_entries: default_cache_max_entries(),
            model_cache_dir: None,
            provider: default_provider(),
            model: default_model(),
            base_url: default_base_url(),
            remote_batch_size: default_remote_batch_size(),
            api_key_env_var: default_api_key_env_var(),
            output_dim: default_output_dim(),
            health_probe_cache_secs: default_health_probe_cache_secs(),
            document_prefix: String::new(),
            query_prefix: String::new(),
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

        match self.provider.as_str() {
            "fastembed" | "openai_compatible" => {}
            other => {
                return Err(format!(
                    "embedding.provider must be 'fastembed' or 'openai_compatible', got '{}'",
                    other
                ));
            }
        }

        if self.output_dim == 0 {
            return Err("embedding.output_dim must be greater than 0".to_string());
        }

        if self.provider == "openai_compatible" && self.remote_batch_size == 0 {
            return Err(
                "embedding.remote_batch_size must be greater than 0 for openai_compatible provider"
                    .to_string(),
            );
        }

        // model_cache_dir is created by the daemon at startup if absent; no existence check here.

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

        if let Ok(val) = env::var("WQM_EMBEDDING_PROVIDER") {
            self.provider = val;
        }

        if let Ok(val) = env::var("WQM_EMBEDDING_BASE_URL") {
            self.base_url = val;
        }

        if let Ok(val) = env::var("WQM_EMBEDDING_API_KEY_ENV_VAR") {
            self.api_key_env_var = val;
        }

        if let Ok(val) = env::var("WQM_EMBEDDING_DOCUMENT_PREFIX") {
            self.document_prefix = val;
        }

        if let Ok(val) = env::var("WQM_EMBEDDING_QUERY_PREFIX") {
            self.query_prefix = val;
        }

        if self.provider == "fastembed" {
            // FastEmbed is pinned to AllMiniLM-L6-v2 in this fork. Align the
            // reported model + dim with the actual provider and clear the
            // base_url (a local model has no endpoint) — otherwise the embedding
            // status surfaces the OpenAI-compatible config DEFAULTS
            // (text-embedding-3-small / api.openai.com), which is misleading.
            // Run LAST so a stray WQM_EMBEDDING_BASE_URL can't re-introduce an
            // endpoint for what is a local, endpoint-less provider.
            self.output_dim = FASTEMBED_OUTPUT_DIM;
            self.model = "all-MiniLM-L6-v2".to_string();
            self.base_url = String::new();
            // MiniLM is not instruction-tuned — prefixes only hurt it.
            self.document_prefix = String::new();
            self.query_prefix = String::new();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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

        // model_cache_dir is accepted regardless of whether the path exists yet;
        // the daemon creates it at startup.
        settings.model_cache_dir = Some(PathBuf::from("/tmp/test_cache/nonexistent/path"));
        assert!(settings.validate().is_ok());

        settings.model_cache_dir = None;
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_embedding_settings_new_defaults() {
        let settings = EmbeddingSettings::default();
        assert_eq!(settings.provider, "openai_compatible");
        assert_eq!(settings.model, "text-embedding-3-small");
        assert_eq!(settings.base_url, "https://api.openai.com");
        assert_eq!(settings.remote_batch_size, 128);
        assert_eq!(settings.api_key_env_var, "OPENAI_API_KEY");
        assert_eq!(settings.output_dim, 1536);
        assert_eq!(settings.health_probe_cache_secs, 60);
    }

    #[test]
    fn test_embedding_settings_existing_defaults_preserved() {
        let settings = EmbeddingSettings::default();
        assert_eq!(settings.cache_max_entries, 1000);
        assert!(settings.model_cache_dir.is_none());
    }

    #[test]
    #[serial]
    fn test_embedding_settings_env_overrides_fastembed_pins_last() {
        let mut settings = EmbeddingSettings::default();

        // Even with a stray BASE_URL set, the fastembed provider must end up
        // endpoint-less and pinned to its real model/dim — the pinning runs
        // AFTER the env overrides so it always wins for this local provider.
        std::env::set_var("WQM_EMBEDDING_PROVIDER", "fastembed");
        std::env::set_var("WQM_EMBEDDING_BASE_URL", "https://example.test");
        std::env::set_var("WQM_EMBEDDING_API_KEY_ENV_VAR", "MY_KEY");

        settings.apply_env_overrides();

        assert_eq!(settings.provider, "fastembed");
        assert_eq!(settings.output_dim, FASTEMBED_OUTPUT_DIM);
        assert_eq!(settings.model, "all-MiniLM-L6-v2");
        assert_eq!(
            settings.base_url, "",
            "fastembed base_url must be cleared even if env sets one"
        );
        // Non-endpoint overrides are still honored.
        assert_eq!(settings.api_key_env_var, "MY_KEY");

        std::env::remove_var("WQM_EMBEDDING_PROVIDER");
        std::env::remove_var("WQM_EMBEDDING_BASE_URL");
        std::env::remove_var("WQM_EMBEDDING_API_KEY_ENV_VAR");
    }

    #[test]
    #[serial]
    fn test_embedding_settings_env_overrides_openai_compatible_keeps_base_url() {
        let mut settings = EmbeddingSettings::default();

        // For a remote provider the BASE_URL override must survive — the
        // fastembed pinning only applies when provider == "fastembed".
        std::env::set_var("WQM_EMBEDDING_PROVIDER", "openai_compatible");
        std::env::set_var("WQM_EMBEDDING_BASE_URL", "https://example.test");

        settings.apply_env_overrides();

        assert_eq!(settings.provider, "openai_compatible");
        assert_eq!(settings.base_url, "https://example.test");

        std::env::remove_var("WQM_EMBEDDING_PROVIDER");
        std::env::remove_var("WQM_EMBEDDING_BASE_URL");
    }

    #[test]
    fn test_embedding_settings_validate_unknown_provider() {
        let mut settings = EmbeddingSettings::default();
        settings.provider = "unknown".to_string();
        assert!(settings.validate().is_err());
    }

    #[test]
    fn test_embedding_settings_validate_output_dim_zero() {
        let mut settings = EmbeddingSettings::default();
        settings.output_dim = 0;
        assert!(settings.validate().is_err());
    }
}
