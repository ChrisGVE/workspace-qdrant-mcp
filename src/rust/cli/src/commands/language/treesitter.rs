//! `language ts-install` and `language ts-remove` subcommands

use anyhow::{anyhow, Result};

use crate::output;

/// Install a Tree-sitter grammar.
pub async fn ts_install(language: &str, force: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    output::section(format!("Installing Tree-sitter Grammar: {}", language));

    let mut config = GrammarConfig::default();
    config.auto_download = true;

    let mut manager = GrammarManager::new(config);

    // Check if already cached
    if !force && manager.cache_paths().grammar_exists(language) {
        output::warning("Grammar already cached. Use --force to reinstall.");
        return Ok(());
    }

    // If force, clear existing cache first
    if force {
        match manager.clear_cache(language) {
            Ok(true) => output::info("Cleared existing cache"),
            Ok(false) => {}
            Err(e) => output::warning(format!("Could not clear cache: {}", e)),
        }
    }

    // Attempt to download and load
    output::info("Downloading grammar...");
    match manager.get_grammar(language).await {
        Ok(_) => {
            output::success("Grammar installed successfully");

            // Verify compatibility
            let info = manager.grammar_info(language);
            if let Some(compat) = info.compatibility {
                if compat.is_compatible() {
                    output::success("Version compatible");
                } else {
                    output::warning("Version may have compatibility issues");
                }
            }
        }
        Err(e) => {
            return Err(anyhow!("Failed to install grammar: {}", e));
        }
    }

    Ok(())
}

/// Remove a Tree-sitter grammar.
pub async fn ts_remove(language: &str) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    output::section(format!("Removing Tree-sitter Grammar: {}", language));

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);

    if language == "all" {
        match manager.clear_all_cache() {
            Ok(count) => {
                output::success(format!("Removed {} grammar(s) from cache", count));
            }
            Err(e) => {
                return Err(anyhow!("Failed to clear cache: {}", e));
            }
        }
    } else {
        match manager.clear_cache(language) {
            Ok(true) => {
                output::success("Grammar removed from cache");
            }
            Ok(false) => {
                output::warning("Grammar not found in cache");
            }
            Err(e) => {
                return Err(anyhow!("Failed to remove grammar: {}", e));
            }
        }
    }

    Ok(())
}
