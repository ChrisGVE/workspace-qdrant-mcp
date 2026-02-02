//! High-level grammar cache manager.
//!
//! This module provides a unified interface for managing tree-sitter grammars,
//! combining the cache, loader, downloader, and version checker into a single
//! convenient API.
//!
//! # Usage
//!
//! ```ignore
//! let manager = GrammarManager::new(config).await?;
//!
//! // Get a grammar (loads from cache or downloads)
//! let language = manager.get_grammar("rust").await?;
//!
//! // Preload all required grammars
//! manager.preload_required().await?;
//! ```

use super::grammar_cache::{GrammarCachePaths, GrammarMetadata};
use super::grammar_downloader::{DownloadError, GrammarDownloader};
use super::grammar_loader::{GrammarLoadError, GrammarLoader};
use super::version_checker::{check_grammar_compatibility, CompatibilityStatus};
use crate::config::GrammarConfig;
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, error, info, warn};
use tree_sitter::Language;

/// Errors that can occur during grammar management.
#[derive(Debug, Error)]
pub enum GrammarError {
    #[error("Grammar not available for language: {0}")]
    NotAvailable(String),

    #[error("Grammar version incompatible: {0}")]
    VersionIncompatible(String),

    #[error("Grammar load failed: {0}")]
    LoadFailed(#[from] GrammarLoadError),

    #[error("Grammar download failed: {0}")]
    DownloadFailed(#[from] DownloadError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Auto-download disabled and grammar not cached: {0}")]
    AutoDownloadDisabled(String),
}

/// Result type for grammar manager operations.
pub type GrammarResult<T> = Result<T, GrammarError>;

/// Status of a grammar in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarStatus {
    /// Grammar is loaded and ready to use
    Loaded,
    /// Grammar is cached but not loaded
    Cached,
    /// Grammar needs to be downloaded
    NeedsDownload,
    /// Grammar exists but version is incompatible
    IncompatibleVersion,
    /// Grammar is not available
    NotAvailable,
}

/// Information about a managed grammar.
#[derive(Debug, Clone)]
pub struct GrammarInfo {
    /// Language name
    pub language: String,
    /// Current status
    pub status: GrammarStatus,
    /// Metadata if available
    pub metadata: Option<GrammarMetadata>,
    /// Compatibility status if loaded
    pub compatibility: Option<CompatibilityStatus>,
}

/// High-level grammar cache manager.
///
/// This struct provides a unified interface for managing tree-sitter grammars,
/// handling caching, loading, downloading, and version checking.
pub struct GrammarManager {
    /// Grammar configuration
    config: GrammarConfig,
    /// Grammar loader
    loader: GrammarLoader,
    /// Grammar downloader (optional, only if auto_download enabled)
    downloader: Option<GrammarDownloader>,
    /// Loaded grammars cache (language -> Language)
    loaded_grammars: HashMap<String, Language>,
    /// Default grammar version to use
    default_version: String,
}

impl GrammarManager {
    /// Create a new grammar manager with the given configuration.
    pub fn new(config: GrammarConfig) -> Self {
        let cache_paths = GrammarCachePaths::with_root(
            config.expanded_cache_dir(),
            &config.tree_sitter_version,
        );

        let loader = GrammarLoader::new(cache_paths.clone());

        let downloader = if config.auto_download {
            Some(GrammarDownloader::new(
                cache_paths,
                &config.download_base_url,
                config.verify_checksums,
            ))
        } else {
            None
        };

        Self {
            config,
            loader,
            downloader,
            loaded_grammars: HashMap::new(),
            default_version: "0.23.0".to_string(), // Default grammar version
        }
    }

    /// Get a grammar for the specified language.
    ///
    /// This method:
    /// 1. Returns the grammar if already loaded
    /// 2. Loads from cache if available
    /// 3. Downloads if auto_download is enabled and not cached
    /// 4. Returns error if unavailable
    pub async fn get_grammar(&mut self, language: &str) -> GrammarResult<Language> {
        // Check if already loaded
        if let Some(lang) = self.loaded_grammars.get(language) {
            debug!(language = language, "Grammar already loaded");
            return Ok(lang.clone());
        }

        // Try to load from cache
        match self.loader.load_grammar(language) {
            Ok(loaded) => {
                // Check version compatibility
                let compat = check_grammar_compatibility(&loaded.language);
                if !compat.is_compatible() {
                    warn!(
                        language = language,
                        "Grammar version incompatible, will try to download new version"
                    );

                    // Try to download a new version if auto_download enabled
                    if let Some(ref downloader) = self.downloader {
                        return self
                            .download_and_load(language, &self.default_version.clone())
                            .await;
                    } else {
                        return Err(GrammarError::VersionIncompatible(format!(
                            "Grammar {} has incompatible version and auto_download is disabled",
                            language
                        )));
                    }
                }

                // Cache and return
                self.loaded_grammars
                    .insert(language.to_string(), loaded.language.clone());
                info!(language = language, "Grammar loaded from cache");
                Ok(loaded.language)
            }
            Err(GrammarLoadError::NotFound(_)) => {
                // Try to download if enabled
                if let Some(ref downloader) = self.downloader {
                    self.download_and_load(language, &self.default_version.clone())
                        .await
                } else {
                    Err(GrammarError::AutoDownloadDisabled(language.to_string()))
                }
            }
            Err(e) => Err(GrammarError::LoadFailed(e)),
        }
    }

    /// Get a grammar if it's already loaded, without attempting to load or download.
    pub fn get_loaded_grammar(&self, language: &str) -> Option<&Language> {
        self.loaded_grammars.get(language)
    }

    /// Check the status of a grammar.
    pub fn grammar_status(&self, language: &str) -> GrammarStatus {
        if self.loaded_grammars.contains_key(language) {
            return GrammarStatus::Loaded;
        }

        let cache_paths = self.loader.cache_paths();
        if cache_paths.grammar_exists(language) {
            // Check if metadata exists and version is compatible
            if let Ok(Some(metadata)) = cache_paths.load_metadata(language) {
                // Simple version check - could be enhanced
                if metadata.tree_sitter_version == self.config.tree_sitter_version {
                    return GrammarStatus::Cached;
                } else {
                    return GrammarStatus::IncompatibleVersion;
                }
            }
            GrammarStatus::Cached
        } else if self.downloader.is_some() {
            GrammarStatus::NeedsDownload
        } else {
            GrammarStatus::NotAvailable
        }
    }

    /// Get information about a grammar.
    pub fn grammar_info(&self, language: &str) -> GrammarInfo {
        let status = self.grammar_status(language);
        let metadata = self
            .loader
            .cache_paths()
            .load_metadata(language)
            .ok()
            .flatten();

        let compatibility = self
            .loaded_grammars
            .get(language)
            .map(|lang| check_grammar_compatibility(lang));

        GrammarInfo {
            language: language.to_string(),
            status,
            metadata,
            compatibility,
        }
    }

    /// Preload all required grammars from configuration.
    ///
    /// Returns a map of language -> result for each required grammar.
    pub async fn preload_required(&mut self) -> HashMap<String, GrammarResult<()>> {
        let required = self.config.required.clone();
        let mut results = HashMap::new();

        for language in required {
            let result = self.get_grammar(&language).await.map(|_| ());
            results.insert(language, result);
        }

        results
    }

    /// Check which required grammars are missing.
    pub fn missing_required(&self) -> Vec<String> {
        self.config
            .required
            .iter()
            .filter(|lang| {
                let status = self.grammar_status(lang);
                !matches!(status, GrammarStatus::Loaded | GrammarStatus::Cached)
            })
            .cloned()
            .collect()
    }

    /// List all loaded grammars.
    pub fn loaded_languages(&self) -> Vec<&str> {
        self.loaded_grammars.keys().map(|s| s.as_str()).collect()
    }

    /// List all cached grammars (loaded or not).
    pub fn cached_languages(&self) -> std::io::Result<Vec<String>> {
        self.loader.cache_paths().list_cached_languages()
    }

    /// Unload a grammar from memory.
    ///
    /// The grammar remains in the cache on disk.
    pub fn unload_grammar(&mut self, language: &str) -> bool {
        self.loaded_grammars.remove(language).is_some() && self.loader.unload_grammar(language)
    }

    /// Unload all grammars from memory.
    pub fn unload_all(&mut self) {
        self.loaded_grammars.clear();
        self.loader.unload_all();
    }

    /// Get the grammar configuration.
    pub fn config(&self) -> &GrammarConfig {
        &self.config
    }

    /// Set the default grammar version for downloads.
    pub fn set_default_version(&mut self, version: impl Into<String>) {
        self.default_version = version.into();
    }

    /// Download and load a grammar.
    async fn download_and_load(
        &mut self,
        language: &str,
        version: &str,
    ) -> GrammarResult<Language> {
        let downloader = self
            .downloader
            .as_ref()
            .ok_or_else(|| GrammarError::AutoDownloadDisabled(language.to_string()))?;

        info!(
            language = language,
            version = version,
            "Downloading grammar"
        );

        // Download the grammar
        downloader.download_grammar(language, version).await?;

        // Load the downloaded grammar
        let loaded = self.loader.load_grammar(language)?;

        // Verify compatibility
        let compat = check_grammar_compatibility(&loaded.language);
        if !compat.is_compatible() {
            error!(
                language = language,
                "Downloaded grammar is still incompatible"
            );
            return Err(GrammarError::VersionIncompatible(format!(
                "Downloaded grammar {} is incompatible with runtime",
                language
            )));
        }

        // Cache and return
        self.loaded_grammars
            .insert(language.to_string(), loaded.language.clone());
        info!(language = language, "Grammar downloaded and loaded");

        Ok(loaded.language)
    }
}

impl std::fmt::Debug for GrammarManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarManager")
            .field("config", &self.config)
            .field("loaded_languages", &self.loaded_languages())
            .field("auto_download", &self.downloader.is_some())
            .finish()
    }
}

/// Create a grammar manager from configuration with default settings.
pub fn create_grammar_manager(config: GrammarConfig) -> GrammarManager {
    GrammarManager::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(temp_dir: &TempDir, auto_download: bool) -> GrammarConfig {
        GrammarConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            required: vec!["rust".to_string(), "python".to_string()],
            auto_download,
            tree_sitter_version: "0.24".to_string(),
            download_base_url: "https://example.com/{language}/v{version}/{platform}.{ext}"
                .to_string(),
            verify_checksums: false,
            lazy_loading: true,
        }
    }

    #[test]
    fn test_grammar_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let manager = GrammarManager::new(config);

        assert!(manager.loaded_languages().is_empty());
    }

    #[test]
    fn test_grammar_manager_no_auto_download() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, false);
        let manager = GrammarManager::new(config);

        // Downloader should be None
        assert!(manager.downloader.is_none());
    }

    #[test]
    fn test_grammar_status_not_cached() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let manager = GrammarManager::new(config);

        // With auto_download, uncached grammars should need download
        assert_eq!(
            manager.grammar_status("rust"),
            GrammarStatus::NeedsDownload
        );

        // Without auto_download, uncached grammars should be unavailable
        let temp_dir2 = TempDir::new().unwrap();
        let config2 = test_config(&temp_dir2, false);
        let manager2 = GrammarManager::new(config2);
        assert_eq!(
            manager2.grammar_status("rust"),
            GrammarStatus::NotAvailable
        );
    }

    #[test]
    fn test_grammar_info() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let manager = GrammarManager::new(config);

        let info = manager.grammar_info("rust");
        assert_eq!(info.language, "rust");
        assert_eq!(info.status, GrammarStatus::NeedsDownload);
        assert!(info.metadata.is_none());
        assert!(info.compatibility.is_none());
    }

    #[test]
    fn test_missing_required() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let manager = GrammarManager::new(config);

        let missing = manager.missing_required();
        assert!(missing.contains(&"rust".to_string()));
        assert!(missing.contains(&"python".to_string()));
    }

    #[test]
    fn test_unload_grammar() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let mut manager = GrammarManager::new(config);

        // Unloading a non-loaded grammar should return false
        assert!(!manager.unload_grammar("rust"));
    }

    #[test]
    fn test_unload_all() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let mut manager = GrammarManager::new(config);

        manager.unload_all();
        assert!(manager.loaded_languages().is_empty());
    }

    #[test]
    fn test_set_default_version() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let mut manager = GrammarManager::new(config);

        manager.set_default_version("0.24.0");
        assert_eq!(manager.default_version, "0.24.0");
    }

    #[test]
    fn test_grammar_manager_debug() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let manager = GrammarManager::new(config);

        let debug_str = format!("{:?}", manager);
        assert!(debug_str.contains("GrammarManager"));
        assert!(debug_str.contains("auto_download"));
    }

    // Note: Testing actual grammar loading requires real grammar files.
    // Integration tests should cover the full workflow with real grammars.
}
