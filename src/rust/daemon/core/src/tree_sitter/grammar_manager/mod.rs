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

mod cache;
mod loading;
mod provider;
mod validation;

#[cfg(test)]
mod tests;

use super::grammar_cache::GrammarCachePaths;
use super::grammar_downloader::GrammarDownloader;
use super::grammar_loader::GrammarLoader;
use crate::config::GrammarConfig;
use std::collections::HashMap;
use thiserror::Error;
use tree_sitter::Language;

// Re-export all public types
pub use provider::LoadedGrammarsProvider;
pub use validation::GrammarValidationResult;

/// Errors that can occur during grammar management.
#[derive(Debug, Error)]
pub enum GrammarError {
    #[error("Grammar not available for language: {0}")]
    NotAvailable(String),

    #[error("Grammar version incompatible: {0}")]
    VersionIncompatible(String),

    #[error("Grammar load failed: {0}")]
    LoadFailed(#[from] super::grammar_loader::GrammarLoadError),

    #[error("Grammar download failed: {0}")]
    DownloadFailed(#[from] super::grammar_downloader::DownloadError),

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
    pub metadata: Option<super::grammar_cache::GrammarMetadata>,
    /// Compatibility status if loaded
    pub compatibility: Option<super::version_checker::CompatibilityStatus>,
}

/// High-level grammar cache manager.
///
/// This struct provides a unified interface for managing tree-sitter grammars,
/// handling caching, loading, downloading, and version checking.
pub struct GrammarManager {
    /// Grammar configuration
    pub(crate) config: GrammarConfig,
    /// Grammar loader
    pub(crate) loader: GrammarLoader,
    /// Grammar downloader (optional, only if auto_download enabled)
    pub(crate) downloader: Option<GrammarDownloader>,
    /// Loaded grammars cache (language -> Language)
    pub(crate) loaded_grammars: HashMap<String, Language>,
    /// Default grammar version to use
    pub(crate) default_version: String,
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

        let default_version = config.tree_sitter_version.clone();

        Self {
            config,
            loader,
            downloader,
            loaded_grammars: HashMap::new(),
            default_version,
        }
    }

    /// Get the grammar configuration.
    pub fn config(&self) -> &GrammarConfig {
        &self.config
    }

    /// Get the grammar cache paths.
    pub fn cache_paths(&self) -> &GrammarCachePaths {
        self.loader.cache_paths()
    }

    /// Check if auto-download is enabled (downloader is available).
    pub fn has_downloader(&self) -> bool {
        self.downloader.is_some()
    }

    /// Set the default grammar version for downloads.
    pub fn set_default_version(&mut self, version: impl Into<String>) {
        self.default_version = version.into();
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
