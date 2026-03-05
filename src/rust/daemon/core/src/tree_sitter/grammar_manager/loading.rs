//! Grammar loading, downloading, preloading, and reloading operations.

use super::{GrammarError, GrammarManager, GrammarResult};
use crate::tree_sitter::grammar_loader::GrammarLoadError;
use crate::tree_sitter::version_checker::check_grammar_compatibility;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};
use tree_sitter::Language;

impl GrammarManager {
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
            Ok(loaded) => self.handle_loaded_grammar(language, loaded).await,
            Err(GrammarLoadError::NotFound(_)) => self.handle_not_found(language).await,
            Err(e) => Err(GrammarError::LoadFailed(e)),
        }
    }

    /// Handle a grammar that was successfully loaded from cache.
    ///
    /// Checks ABI compatibility and metadata version, potentially re-downloading
    /// if the cached version is incompatible or outdated.
    async fn handle_loaded_grammar(
        &mut self,
        language: &str,
        loaded: crate::tree_sitter::grammar_loader::LoadedGrammar,
    ) -> GrammarResult<Language> {
        // Check ABI compatibility first
        let compat = check_grammar_compatibility(&loaded.language);
        if !compat.is_compatible() {
            warn!(
                language = language,
                "Grammar ABI incompatible, will try to download new version"
            );

            if self.downloader.is_some() {
                return self
                    .download_and_load(language, &self.default_version.clone())
                    .await;
            }
            return Err(GrammarError::VersionIncompatible(format!(
                "Grammar {} has incompatible ABI version and auto_download is disabled",
                language
            )));
        }

        // Check metadata version matches configured version
        if self.should_redownload_for_version(language) {
            if self.downloader.is_some() {
                return self
                    .download_and_load(language, &self.default_version.clone())
                    .await;
            }
            // If auto_download is disabled, use the cached version anyway
            debug!(
                language = language,
                "Using cached grammar despite version mismatch (auto_download disabled)"
            );
        }

        // Cache and return
        self.loaded_grammars
            .insert(language.to_string(), loaded.language.clone());
        info!(language = language, "Grammar loaded from cache");
        Ok(loaded.language)
    }

    /// Check whether a cached grammar should be re-downloaded due to version mismatch.
    fn should_redownload_for_version(&self, language: &str) -> bool {
        let cache_paths = self.loader.cache_paths();
        if let Ok(Some(metadata)) = cache_paths.load_metadata(language) {
            if metadata.tree_sitter_version != self.config.tree_sitter_version {
                warn!(
                    language = language,
                    cached_version = %metadata.tree_sitter_version,
                    expected_version = %self.config.tree_sitter_version,
                    "Grammar metadata version mismatch, will try to re-download"
                );
                return true;
            }
        }
        false
    }

    /// Handle a grammar that was not found in the cache.
    async fn handle_not_found(&mut self, language: &str) -> GrammarResult<Language> {
        if self.downloader.is_some() {
            self.download_and_load(language, &self.default_version.clone())
                .await
        } else {
            Err(GrammarError::AutoDownloadDisabled(language.to_string()))
        }
    }

    /// Get a grammar if it's already loaded, without attempting to load or download.
    pub fn get_loaded_grammar(&self, language: &str) -> Option<&Language> {
        self.loaded_grammars.get(language)
    }

    /// Download and load a grammar.
    pub(super) async fn download_and_load(
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
        let languages: Vec<String> = self
            .loaded_languages()
            .iter()
            .map(|s| s.to_string())
            .collect();

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
}
