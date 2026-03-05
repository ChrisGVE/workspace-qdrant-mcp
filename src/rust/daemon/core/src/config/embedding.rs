//! Embedding generation configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_cache_max_entries() -> usize {
    1000
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
