//! Grammar validation, periodic version checks, and preload orchestration.

use super::{GrammarManager, GrammarResult, GrammarStatus};
use std::collections::HashMap;
use tracing::{debug, info, warn};

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
    /// Returns a `GrammarValidationResult` indicating which grammars are
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
                    messages.push(format!(
                        "{}: incompatible version, needs re-download",
                        language
                    ));
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

        // If nothing needs downloading or auto_download is disabled, return
        if initial.needs_download.is_empty() || self.downloader.is_none() {
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

    /// Check if a periodic grammar version check is due.
    ///
    /// Compares the `last_checked_at` timestamp in grammar metadata against
    /// the configured `check_interval_hours`. Returns true if any required
    /// grammar needs checking.
    pub fn needs_periodic_check(&self) -> bool {
        if self.config.check_interval_hours == 0 {
            return false;
        }

        let interval = chrono::Duration::hours(self.config.check_interval_hours as i64);
        let now = chrono::Utc::now();
        let cache_paths = self.loader.cache_paths();

        for language in &self.config.required {
            if let Ok(Some(metadata)) = cache_paths.load_metadata(language) {
                let last_check = metadata
                    .last_checked_at
                    .as_deref()
                    .or(Some(metadata.cached_at.as_str()))
                    .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                    .map(|dt| dt.with_timezone(&chrono::Utc));

                match last_check {
                    Some(checked_at) if now - checked_at < interval => continue,
                    _ => return true,
                }
            } else {
                // No metadata means no check has been done
                return true;
            }
        }

        false
    }

    /// Perform a periodic grammar version check.
    ///
    /// For each required grammar, checks if the cached version matches the
    /// configured `tree_sitter_version`. Re-downloads if mismatched and
    /// auto_download is enabled. Updates `last_checked_at` timestamps.
    ///
    /// This method is designed to be called by the daemon's main loop at
    /// the interval specified by `check_interval_hours`.
    pub async fn periodic_version_check(&mut self) -> HashMap<String, GrammarResult<()>> {
        let mut results = HashMap::new();

        if self.downloader.is_none() {
            debug!("Periodic grammar check skipped: auto_download disabled");
            return results;
        }

        info!("Running periodic grammar version check");

        let required = self.config.required.clone();
        let expected_version = self.config.tree_sitter_version.clone();
        let cache_paths = self.loader.cache_paths().clone();

        for language in &required {
            let needs_update = if let Ok(Some(metadata)) = cache_paths.load_metadata(language) {
                metadata.tree_sitter_version != expected_version
            } else {
                true // No metadata means grammar needs download
            };

            if needs_update {
                info!(
                    language = language.as_str(),
                    expected_version = expected_version.as_str(),
                    "Grammar version mismatch, re-downloading"
                );
                let result = self.get_grammar(language).await.map(|_| ());
                results.insert(language.clone(), result);
            } else {
                // Update last_checked_at timestamp
                if let Ok(Some(mut metadata)) = cache_paths.load_metadata(language) {
                    metadata.mark_checked();
                    if let Err(e) = cache_paths.save_metadata(language, &metadata) {
                        warn!(
                            language = language.as_str(),
                            "Failed to update grammar check timestamp: {}", e
                        );
                    }
                }
                results.insert(language.clone(), Ok(()));
            }
        }

        results
    }
}
