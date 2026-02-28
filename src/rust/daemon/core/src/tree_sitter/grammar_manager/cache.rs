//! Grammar cache status, inspection, and lifecycle management.

use super::{GrammarInfo, GrammarManager, GrammarStatus};
use crate::tree_sitter::version_checker::check_grammar_compatibility;
use tracing::info;

impl GrammarManager {
    /// Check the status of a grammar.
    pub fn grammar_status(&self, language: &str) -> GrammarStatus {
        if self.loaded_grammars.contains_key(language) {
            return GrammarStatus::Loaded;
        }

        let cache_paths = self.loader.cache_paths();
        if cache_paths.grammar_exists(language) {
            // Check if metadata exists and version is compatible
            if let Ok(Some(metadata)) = cache_paths.load_metadata(language) {
                if metadata.tree_sitter_version == self.config.tree_sitter_version {
                    return GrammarStatus::Cached;
                }
                return GrammarStatus::IncompatibleVersion;
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
}
