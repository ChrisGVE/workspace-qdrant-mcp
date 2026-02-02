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

    /// Reload a grammar from the cache (unload then load).
    ///
    /// This is useful when you want to refresh a grammar after updating
    /// the cache, for example after downloading a new version.
    pub async fn reload_grammar(&mut self, language: &str) -> GrammarResult<Language> {
        info!(language = language, "Reloading grammar");

        // Unload the existing grammar
        self.unload_grammar(language);

        // Load fresh from cache or download
        self.get_grammar(language).await
    }

    /// Reload all loaded grammars.
    ///
    /// This unloads all grammars and reloads them from the cache.
    /// Useful after bulk grammar updates.
    pub async fn reload_all(&mut self) -> HashMap<String, GrammarResult<()>> {
        let languages: Vec<String> = self.loaded_languages().iter().map(|s| s.to_string()).collect();

        info!("Reloading {} grammars", languages.len());

        // Unload all
        self.unload_all();

        // Reload each
        let mut results = HashMap::new();
        for language in languages {
            let result = self.get_grammar(&language).await.map(|_| ());
            results.insert(language, result);
        }

        results
    }

    /// Clear the grammar cache for a specific language.
    ///
    /// This removes the cached grammar file and metadata, forcing a re-download
    /// on the next request if auto_download is enabled.
    pub fn clear_cache(&self, language: &str) -> std::io::Result<bool> {
        let grammar_path = self.loader.cache_paths().grammar_path(language);
        let metadata_path = self.loader.cache_paths().metadata_path(language);

        let mut cleared = false;

        if grammar_path.exists() {
            std::fs::remove_file(&grammar_path)?;
            cleared = true;
            info!(language = language, "Cleared cached grammar file");
        }

        if metadata_path.exists() {
            std::fs::remove_file(&metadata_path)?;
            info!(language = language, "Cleared cached grammar metadata");
        }

        Ok(cleared)
    }

    /// Clear all cached grammars.
    pub fn clear_all_cache(&self) -> std::io::Result<usize> {
        let languages = self.cached_languages()?;
        let mut cleared = 0;

        for language in languages {
            if self.clear_cache(&language)? {
                cleared += 1;
            }
        }

        info!("Cleared {} cached grammars", cleared);
        Ok(cleared)
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

/// Validation result for grammar availability checks.
#[derive(Debug, Clone)]
pub struct GrammarValidationResult {
    /// Grammars that are fully available (loaded or can be loaded)
    pub available: Vec<String>,
    /// Grammars that need to be downloaded
    pub needs_download: Vec<String>,
    /// Grammars that are not available and cannot be obtained
    pub unavailable: Vec<String>,
    /// Whether all required grammars are available
    pub all_required_available: bool,
    /// Validation messages for logging/reporting
    pub messages: Vec<String>,
}

impl GrammarValidationResult {
    /// Check if validation passed (all required grammars available).
    pub fn is_valid(&self) -> bool {
        self.all_required_available
    }

    /// Get a summary of the validation result.
    pub fn summary(&self) -> String {
        format!(
            "Grammars: {} available, {} need download, {} unavailable",
            self.available.len(),
            self.needs_download.len(),
            self.unavailable.len()
        )
    }
}

impl GrammarManager {
    /// Validate grammar availability at startup.
    ///
    /// This checks all required grammars and returns a validation result
    /// that can be used for logging and decision making.
    ///
    /// Returns a GrammarValidationResult indicating which grammars are
    /// available, which need downloading, and which are unavailable.
    pub fn validate_grammars(&self) -> GrammarValidationResult {
        let mut available = Vec::new();
        let mut needs_download = Vec::new();
        let mut unavailable = Vec::new();
        let mut messages = Vec::new();

        for language in &self.config.required {
            match self.grammar_status(language) {
                GrammarStatus::Loaded => {
                    available.push(language.clone());
                    messages.push(format!("{}: loaded", language));
                }
                GrammarStatus::Cached => {
                    available.push(language.clone());
                    messages.push(format!("{}: cached", language));
                }
                GrammarStatus::NeedsDownload => {
                    needs_download.push(language.clone());
                    messages.push(format!("{}: needs download", language));
                }
                GrammarStatus::IncompatibleVersion => {
                    needs_download.push(language.clone());
                    messages.push(format!("{}: incompatible version, needs re-download", language));
                }
                GrammarStatus::NotAvailable => {
                    unavailable.push(language.clone());
                    messages.push(format!(
                        "{}: not available (auto_download disabled)",
                        language
                    ));
                }
            }
        }

        // All required are available if none are in unavailable
        // (needs_download is OK if auto_download is enabled)
        let all_required_available = unavailable.is_empty();

        GrammarValidationResult {
            available,
            needs_download,
            unavailable,
            all_required_available,
            messages,
        }
    }

    /// Validate and optionally preload required grammars.
    ///
    /// This performs validation and, if auto_download is enabled, attempts
    /// to download and load any missing grammars.
    ///
    /// Returns the validation result after any download attempts.
    pub async fn validate_and_preload(&mut self) -> GrammarValidationResult {
        // First, run initial validation
        let initial = self.validate_grammars();

        // Log initial status
        for msg in &initial.messages {
            debug!("{}", msg);
        }

        // If all required are available or auto_download is disabled, return
        if initial.is_valid() || self.downloader.is_none() {
            return initial;
        }

        // Try to preload required grammars
        info!("Preloading required grammars...");
        let results = self.preload_required().await;

        // Log results
        for (language, result) in &results {
            match result {
                Ok(_) => info!("Grammar '{}' loaded successfully", language),
                Err(e) => warn!("Failed to load grammar '{}': {}", language, e),
            }
        }

        // Re-validate after preloading
        self.validate_grammars()
    }
}

/// A synchronous LanguageProvider backed by pre-loaded grammars.
///
/// This provider is created from a GrammarManager's loaded grammars and can be
/// used with SemanticChunker. Since loading grammars is an async operation,
/// this provider only returns grammars that were already loaded.
///
/// # Example
///
/// ```ignore
/// // Preload grammars (async)
/// let manager = GrammarManager::new(config);
/// manager.preload_required().await;
///
/// // Create sync provider from loaded grammars
/// let provider = manager.create_language_provider();
///
/// // Use with chunker
/// let chunker = SemanticChunker::with_provider(8000, Arc::new(provider));
/// ```
#[derive(Debug, Clone)]
pub struct LoadedGrammarsProvider {
    grammars: HashMap<String, Language>,
}

impl LoadedGrammarsProvider {
    /// Create a new provider with no loaded grammars.
    pub fn new() -> Self {
        Self {
            grammars: HashMap::new(),
        }
    }

    /// Create a provider from a map of loaded grammars.
    pub fn from_loaded(grammars: HashMap<String, Language>) -> Self {
        Self { grammars }
    }

    /// Add a grammar to the provider.
    pub fn add_grammar(&mut self, language: &str, grammar: Language) {
        self.grammars.insert(language.to_string(), grammar);
    }

    /// Get the number of loaded grammars.
    pub fn len(&self) -> usize {
        self.grammars.len()
    }

    /// Check if the provider has no loaded grammars.
    pub fn is_empty(&self) -> bool {
        self.grammars.is_empty()
    }
}

impl Default for LoadedGrammarsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl super::parser::LanguageProvider for LoadedGrammarsProvider {
    fn get_language(&self, name: &str) -> Option<Language> {
        self.grammars.get(name).cloned()
    }

    fn supports_language(&self, name: &str) -> bool {
        self.grammars.contains_key(name)
    }

    fn available_languages(&self) -> Vec<&str> {
        self.grammars.keys().map(|s| s.as_str()).collect()
    }
}

impl GrammarManager {
    /// Create a synchronous LanguageProvider from the currently loaded grammars.
    ///
    /// This takes a snapshot of the loaded grammars. Any grammars loaded after
    /// calling this method won't be available in the provider.
    pub fn create_language_provider(&self) -> LoadedGrammarsProvider {
        LoadedGrammarsProvider::from_loaded(self.loaded_grammars.clone())
    }
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

    // Tests for LoadedGrammarsProvider
    #[test]
    fn test_loaded_grammars_provider_new() {
        use crate::tree_sitter::parser::LanguageProvider;

        let provider = LoadedGrammarsProvider::new();
        assert!(provider.is_empty());
        assert_eq!(provider.len(), 0);
        assert!(provider.available_languages().is_empty());
    }

    #[test]
    fn test_loaded_grammars_provider_with_grammar() {
        use crate::tree_sitter::parser::{get_static_language, LanguageProvider};

        let mut provider = LoadedGrammarsProvider::new();

        // Add a real grammar from static loading
        if let Some(lang) = get_static_language("rust") {
            provider.add_grammar("rust", lang);
        }

        assert!(!provider.is_empty());
        assert_eq!(provider.len(), 1);
        assert!(provider.supports_language("rust"));
        assert!(!provider.supports_language("python"));

        let languages = provider.available_languages();
        assert!(languages.contains(&"rust"));
    }

    #[test]
    fn test_loaded_grammars_provider_get_language() {
        use crate::tree_sitter::parser::{get_static_language, LanguageProvider};

        let mut provider = LoadedGrammarsProvider::new();

        if let Some(lang) = get_static_language("rust") {
            provider.add_grammar("rust", lang);
        }

        // Should return the language
        assert!(provider.get_language("rust").is_some());

        // Should return None for unknown language
        assert!(provider.get_language("unknown").is_none());
    }

    #[test]
    fn test_loaded_grammars_provider_from_map() {
        use crate::tree_sitter::parser::{get_static_language, LanguageProvider};
        use std::collections::HashMap;

        let mut map = HashMap::new();
        if let Some(lang) = get_static_language("rust") {
            map.insert("rust".to_string(), lang);
        }
        if let Some(lang) = get_static_language("python") {
            map.insert("python".to_string(), lang);
        }

        let provider = LoadedGrammarsProvider::from_loaded(map);

        assert_eq!(provider.len(), 2);
        assert!(provider.supports_language("rust"));
        assert!(provider.supports_language("python"));
    }

    #[test]
    fn test_grammar_manager_create_language_provider() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, false);
        let manager = GrammarManager::new(config);

        // Manager has no loaded grammars yet
        let provider = manager.create_language_provider();
        assert!(provider.is_empty());
    }

    // Tests for grammar validation
    #[test]
    fn test_validate_grammars_with_auto_download() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true); // auto_download enabled
        let manager = GrammarManager::new(config);

        let result = manager.validate_grammars();

        // With auto_download, missing grammars should be in needs_download
        assert!(!result.needs_download.is_empty());
        assert!(result.unavailable.is_empty());
        // All required are "available" because they can be downloaded
        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_grammars_without_auto_download() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, false); // auto_download disabled
        let manager = GrammarManager::new(config);

        let result = manager.validate_grammars();

        // Without auto_download, missing grammars should be unavailable
        assert!(!result.unavailable.is_empty());
        // Not all required are available
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validation_result_summary() {
        let result = GrammarValidationResult {
            available: vec!["rust".to_string()],
            needs_download: vec!["python".to_string()],
            unavailable: vec!["go".to_string()],
            all_required_available: false,
            messages: vec![],
        };

        let summary = result.summary();
        assert!(summary.contains("1 available"));
        assert!(summary.contains("1 need download"));
        assert!(summary.contains("1 unavailable"));
    }

    #[test]
    fn test_validation_result_is_valid() {
        // Valid: all required available
        let valid_result = GrammarValidationResult {
            available: vec!["rust".to_string()],
            needs_download: vec![],
            unavailable: vec![],
            all_required_available: true,
            messages: vec![],
        };
        assert!(valid_result.is_valid());

        // Invalid: some unavailable
        let invalid_result = GrammarValidationResult {
            available: vec![],
            needs_download: vec![],
            unavailable: vec!["rust".to_string()],
            all_required_available: false,
            messages: vec![],
        };
        assert!(!invalid_result.is_valid());
    }

    // Tests for reload and cache clearing methods
    #[test]
    fn test_clear_cache_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, false);
        let manager = GrammarManager::new(config);

        // Clearing cache for a language that doesn't exist should return Ok(false)
        let result = manager.clear_cache("nonexistent");
        assert!(result.is_ok());
        assert!(!result.unwrap()); // false = nothing was cleared
    }

    #[test]
    fn test_clear_cache_with_cached_grammar() {
        use crate::tree_sitter::grammar_cache::grammar_filename;

        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let manager = GrammarManager::new(config);

        // Create a fake cache in the correct directory structure
        // The path is: cache_dir/<platform>/<tree_sitter_version>/<language>/grammar.<ext>
        let grammar_path = manager.loader.cache_paths().grammar_path("rust");
        std::fs::create_dir_all(grammar_path.parent().unwrap()).unwrap();
        std::fs::write(&grammar_path, b"fake grammar").unwrap();

        assert!(grammar_path.exists());

        // Clear should succeed and return true
        let result = manager.clear_cache("rust");
        assert!(result.is_ok());
        assert!(result.unwrap()); // true = something was cleared

        // Grammar file should no longer exist
        assert!(!grammar_path.exists());
    }

    #[test]
    fn test_clear_all_cache_empty() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, false);
        let manager = GrammarManager::new(config);

        // Clearing all cache when empty should return 0
        let result = manager.clear_all_cache();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_clear_all_cache_with_multiple_languages() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, true);
        let manager = GrammarManager::new(config);

        // Create fake cache directories for multiple languages with correct structure
        for lang in &["rust", "python", "go"] {
            let grammar_path = manager.loader.cache_paths().grammar_path(lang);
            std::fs::create_dir_all(grammar_path.parent().unwrap()).unwrap();
            std::fs::write(&grammar_path, b"fake grammar").unwrap();
        }

        // Clear all should succeed
        let result = manager.clear_all_cache();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3); // 3 languages cleared

        // Verify grammar files no longer exist
        for lang in &["rust", "python", "go"] {
            let grammar_path = manager.loader.cache_paths().grammar_path(lang);
            assert!(!grammar_path.exists(), "Grammar for {} should be cleared", lang);
        }
    }

    #[tokio::test]
    async fn test_reload_grammar_not_loaded() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, false);
        let mut manager = GrammarManager::new(config);

        // Reloading a grammar that was never loaded and can't be downloaded
        // should fail with AutoDownloadDisabled (since auto_download is false)
        let result = manager.reload_grammar("nonexistent").await;
        assert!(result.is_err());
        match result {
            Err(GrammarError::AutoDownloadDisabled(lang)) => assert_eq!(lang, "nonexistent"),
            other => panic!("Expected AutoDownloadDisabled error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_reload_all_empty() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir, false);
        let mut manager = GrammarManager::new(config);

        // Reloading all when nothing is loaded should return empty map
        let results = manager.reload_all().await;
        assert!(results.is_empty());
    }
}
