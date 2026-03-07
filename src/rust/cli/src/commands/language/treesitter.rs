//! `language ts-install`, `language ts-remove`, and `language ts-search` subcommands

use anyhow::{anyhow, Result};
use colored::Colorize;

use crate::output;

use super::helpers::{find_language, load_definitions};

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

/// List Tree-sitter grammars with cache and registry status.
pub async fn ts_list(show_all: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus};

    output::section("Tree-sitter Grammars");

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);
    let cached = manager.cached_languages().unwrap_or_default();

    let mut defs = load_definitions();
    defs.sort_by(|a, b| a.language.to_lowercase().cmp(&b.language.to_lowercase()));

    println!(
        "  {:<16} {:<42} {:<10} {}",
        "Language".bold(),
        "Repository".bold(),
        "Status".bold(),
        "Patterns".bold()
    );
    println!("  {}", "─".repeat(78).dimmed());

    let mut shown = 0;
    let mut cached_count = 0;

    for def in &defs {
        if !def.has_grammar() {
            continue;
        }

        let lang_id = def.id();
        let is_cached = cached.contains(&lang_id);
        let status = manager.grammar_status(&lang_id);

        if !show_all && !is_cached {
            continue;
        }

        let status_str = match status {
            GrammarStatus::Loaded => "Loaded".green(),
            GrammarStatus::Cached => "Cached".blue(),
            GrammarStatus::NeedsDownload => "Available".yellow(),
            GrammarStatus::NotAvailable => "N/A".dimmed(),
            GrammarStatus::IncompatibleVersion => "Incompat".red(),
        };

        let repo = def
            .grammar
            .sources
            .first()
            .map(|s| s.repo.as_str())
            .unwrap_or("—");

        let has_patterns = if def.has_semantic_patterns() {
            "✓".green().to_string()
        } else {
            "—".dimmed().to_string()
        };

        println!(
            "  {:<16} {:<42} {:<10} {}",
            def.language, repo, status_str, has_patterns
        );

        shown += 1;
        if is_cached {
            cached_count += 1;
        }
    }

    println!();
    if show_all {
        output::kv("Cached", format!("{cached_count}/{shown}"));
    } else {
        output::kv("Cached", format!("{cached_count}"));
        output::info("Use --all to show available but uncached grammars.");
    }

    Ok(())
}

/// Search available Tree-sitter grammars for a language.
pub async fn ts_search(language: &str) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::language_registry::types::GrammarQuality;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    output::section(format!("Grammar Search: {}", language));

    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("Unknown language: {language}"));
            output::info("Use 'wqm language list' to see all known languages.");
            return Ok(());
        }
    };

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);
    let status = manager.grammar_status(&def.id());

    output::kv("Language", &def.language);
    output::kv("Status", &format!("{:?}", status));
    output::separator();

    if def.grammar.sources.is_empty() {
        output::info("No grammar sources found in registry.");
        return Ok(());
    }

    println!(
        "  {:<45} {:<10} {}",
        "Repository".bold(),
        "Quality".bold(),
        "Origin".bold()
    );
    println!("  {}", "─".repeat(70).dimmed());

    for src in &def.grammar.sources {
        let quality = format!("{:?}", src.quality);
        let quality_colored = match src.quality {
            GrammarQuality::Curated => quality.green(),
            GrammarQuality::Official => quality.blue(),
            GrammarQuality::Community => quality.yellow(),
        };
        let origin = src.origin.as_deref().unwrap_or("unknown");
        println!("  {:<45} {:<10} {}", src.repo, quality_colored, origin);
    }

    if def.grammar.has_cpp_scanner {
        println!();
        output::info("Note: This grammar requires a C++ compiler for its scanner.");
    }

    println!();
    output::info(format!("Install with: wqm language ts-install {language}"));

    Ok(())
}
