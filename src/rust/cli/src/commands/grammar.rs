//! Grammar command - tree-sitter grammar management
//!
//! Phase 1 HIGH priority command for managing tree-sitter grammars.
//! Subcommands: list, install, remove, verify, reload, clear-cache

use anyhow::{anyhow, Result};
use clap::{Args, Subcommand};
use colored::Colorize;

/// Grammar command arguments
#[derive(Args)]
pub struct GrammarArgs {
    #[command(subcommand)]
    command: GrammarCommand,
}

/// Grammar subcommands
#[derive(Subcommand)]
enum GrammarCommand {
    /// List available and loaded grammars
    List {
        /// Show only loaded grammars
        #[arg(short, long)]
        loaded: bool,

        /// Show only cached grammars
        #[arg(short, long)]
        cached: bool,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Install (download) a grammar for a language
    Install {
        /// Language to install grammar for (e.g., rust, python)
        language: String,

        /// Specific version to install
        #[arg(short, long)]
        version: Option<String>,

        /// Force reinstall even if already cached
        #[arg(short, long)]
        force: bool,
    },

    /// Remove a grammar from the cache
    Remove {
        /// Language to remove (or 'all' for all grammars)
        language: String,
    },

    /// Verify grammar compatibility
    Verify {
        /// Language to verify (or 'all' for all grammars)
        language: String,
    },

    /// Reload a grammar (unload and load fresh from cache)
    Reload {
        /// Language to reload (or 'all' for all grammars)
        language: String,
    },

    /// Clear the grammar cache
    ClearCache {
        /// Language to clear (or 'all' for all grammars)
        language: String,
    },

    /// Show grammar cache location and info
    Info,
}

/// Execute the grammar command
pub async fn execute(args: GrammarArgs) -> Result<()> {
    match args.command {
        GrammarCommand::List { loaded, cached, verbose } => {
            list_grammars(loaded, cached, verbose).await
        }
        GrammarCommand::Install { language, version, force } => {
            install_grammar(&language, version.as_deref(), force).await
        }
        GrammarCommand::Remove { language } => {
            remove_grammar(&language).await
        }
        GrammarCommand::Verify { language } => {
            verify_grammar(&language).await
        }
        GrammarCommand::Reload { language } => {
            reload_grammar(&language).await
        }
        GrammarCommand::ClearCache { language } => {
            clear_cache(&language).await
        }
        GrammarCommand::Info => {
            show_grammar_info().await
        }
    }
}

/// List available and loaded grammars
async fn list_grammars(loaded_only: bool, cached_only: bool, verbose: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus, StaticLanguageProvider};

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);

    println!("{}", "Tree-sitter Grammars".bold());
    println!();

    // Show static grammars
    if !cached_only {
        println!("{}", "Static (Bundled) Grammars".cyan());
        let static_langs = StaticLanguageProvider::SUPPORTED_LANGUAGES;
        if static_langs.is_empty() {
            println!("  (none - static-grammars feature disabled)");
        } else {
            for lang in static_langs {
                println!("  {} {}", "✓".green(), lang);
            }
        }
        println!();
    }

    // Show cached grammars
    if !loaded_only {
        println!("{}", "Cached Grammars".cyan());
        match manager.cached_languages() {
            Ok(cached) if !cached.is_empty() => {
                for lang in &cached {
                    let status = manager.grammar_status(lang);
                    let status_icon = match status {
                        GrammarStatus::Loaded => "✓".green(),
                        GrammarStatus::Cached => "●".blue(),
                        GrammarStatus::NeedsDownload => "↓".yellow(),
                        GrammarStatus::NotAvailable => "✗".red(),
                        GrammarStatus::IncompatibleVersion => "!".yellow(),
                    };
                    print!("  {} {}", status_icon, lang);

                    if verbose {
                        let info = manager.grammar_info(lang);
                        if let Some(meta) = info.metadata {
                            print!(" (v{}, ts {})", meta.grammar_version, meta.tree_sitter_version);
                        }
                    }
                    println!();
                }
            }
            Ok(_) => println!("  (none)"),
            Err(e) => println!("  Error reading cache: {}", e),
        }
        println!();
    }

    // Show loaded grammars
    println!("{}", "Loaded Grammars".cyan());
    let loaded = manager.loaded_languages();
    if loaded.is_empty() {
        println!("  (none - grammars are loaded on demand)");
    } else {
        for lang in loaded {
            println!("  {} {}", "✓".green(), lang);
        }
    }

    Ok(())
}

/// Install a grammar for a language
async fn install_grammar(language: &str, version: Option<&str>, force: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    println!("{}", format!("Installing Grammar: {}", language).bold());
    println!();

    let mut config = GrammarConfig::default();
    config.auto_download = true;

    let mut manager = GrammarManager::new(config);

    if let Some(v) = version {
        manager.set_default_version(v);
        println!("  Using version: {}", v);
    }

    // Check if already cached
    if !force && manager.cache_paths().grammar_exists(language) {
        println!("  {} Grammar already cached. Use --force to reinstall.", "!".yellow());
        return Ok(());
    }

    // If force, clear existing cache first
    if force {
        match manager.clear_cache(language) {
            Ok(true) => println!("  {} Cleared existing cache", "i".blue()),
            Ok(false) => {}
            Err(e) => println!("  {} Warning: failed to clear cache: {}", "!".yellow(), e),
        }
    }

    // Attempt to download and load
    println!("  Downloading grammar...");
    match manager.get_grammar(language).await {
        Ok(_) => {
            println!("  {} Grammar installed successfully", "✓".green());

            // Verify compatibility
            let info = manager.grammar_info(language);
            if let Some(compat) = info.compatibility {
                if compat.is_compatible() {
                    println!("  {} Version compatible", "✓".green());
                } else {
                    println!("  {} Version may have compatibility issues", "!".yellow());
                }
            }
        }
        Err(e) => {
            return Err(anyhow!("Failed to install grammar: {}", e));
        }
    }

    Ok(())
}

/// Remove a grammar from the cache
async fn remove_grammar(language: &str) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);

    println!("{}", format!("Removing Grammar: {}", language).bold());
    println!();

    if language == "all" {
        match manager.clear_all_cache() {
            Ok(count) => {
                println!("  {} Removed {} grammar(s) from cache", "✓".green(), count);
            }
            Err(e) => {
                return Err(anyhow!("Failed to clear cache: {}", e));
            }
        }
    } else {
        match manager.clear_cache(language) {
            Ok(true) => {
                println!("  {} Grammar removed from cache", "✓".green());
            }
            Ok(false) => {
                println!("  {} Grammar not found in cache", "!".yellow());
            }
            Err(e) => {
                return Err(anyhow!("Failed to remove grammar: {}", e));
            }
        }
    }

    Ok(())
}

/// Verify grammar compatibility
async fn verify_grammar(language: &str) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::{GrammarManager, check_grammar_compatibility};

    let config = GrammarConfig::default();
    let mut manager = GrammarManager::new(config);

    println!("{}", format!("Verifying Grammar: {}", language).bold());
    println!();

    if language == "all" {
        let result = manager.validate_grammars();
        println!("{}", result.summary());
        println!();

        if !result.available.is_empty() {
            println!("{}", "Available".cyan());
            for lang in &result.available {
                println!("  {} {}", "✓".green(), lang);
            }
        }

        if !result.needs_download.is_empty() {
            println!("{}", "Need Download".cyan());
            for lang in &result.needs_download {
                println!("  {} {}", "↓".yellow(), lang);
            }
        }

        if !result.unavailable.is_empty() {
            println!("{}", "Unavailable".cyan());
            for lang in &result.unavailable {
                println!("  {} {}", "✗".red(), lang);
            }
        }

        if !result.is_valid() {
            return Err(anyhow!("Some required grammars are unavailable"));
        }
    } else {
        // Try to load and verify
        match manager.get_grammar(language).await {
            Ok(lang) => {
                let compat = check_grammar_compatibility(&lang);
                println!("  Language: {}", language);
                println!("  ABI Version: {}", lang.version());

                if compat.is_fully_compatible() {
                    println!("  {} Fully compatible", "✓".green());
                } else if compat.is_compatible() {
                    println!("  {} Compatible (minor version difference)", "!".yellow());
                } else {
                    println!("  {} Incompatible - grammar may not work correctly", "✗".red());
                    return Err(anyhow!("Grammar version incompatible"));
                }
            }
            Err(e) => {
                println!("  {} Could not load grammar: {}", "✗".red(), e);
                return Err(anyhow!("Failed to verify grammar: {}", e));
            }
        }
    }

    Ok(())
}

/// Reload a grammar
async fn reload_grammar(language: &str) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    let config = GrammarConfig::default();
    let mut manager = GrammarManager::new(config);

    println!("{}", format!("Reloading Grammar: {}", language).bold());
    println!();

    if language == "all" {
        let results = manager.reload_all().await;
        let mut success = 0;
        let mut failed = 0;

        for (lang, result) in results {
            match result {
                Ok(_) => {
                    println!("  {} {}", "✓".green(), lang);
                    success += 1;
                }
                Err(e) => {
                    println!("  {} {} - {}", "✗".red(), lang, e);
                    failed += 1;
                }
            }
        }

        println!();
        println!("  Reloaded: {}, Failed: {}", success, failed);

        if failed > 0 {
            return Err(anyhow!("Some grammars failed to reload"));
        }
    } else {
        match manager.reload_grammar(language).await {
            Ok(_) => {
                println!("  {} Grammar reloaded successfully", "✓".green());
            }
            Err(e) => {
                return Err(anyhow!("Failed to reload grammar: {}", e));
            }
        }
    }

    Ok(())
}

/// Clear the grammar cache
async fn clear_cache(language: &str) -> Result<()> {
    // Same as remove_grammar
    remove_grammar(language).await
}

/// Show grammar info
async fn show_grammar_info() -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus, RuntimeInfo};

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config.clone());

    println!("{}", "Grammar Configuration".bold());
    println!();

    // Runtime info
    let runtime = RuntimeInfo::current();
    println!("{}", "Tree-sitter Runtime".cyan());
    println!("  Version: {}", runtime.version_string);
    println!("  ABI Version: {}", runtime.abi_version);
    println!("  Min Compatible ABI: {}", runtime.min_compatible_abi);
    println!();

    // Cache info
    println!("{}", "Cache Configuration".cyan());
    println!("  Cache Directory: {}", config.cache_dir.display());
    println!("  Auto Download: {}", if config.auto_download { "enabled" } else { "disabled" });
    println!("  Verify Checksums: {}", if config.verify_checksums { "enabled" } else { "disabled" });
    println!("  Lazy Loading: {}", if config.lazy_loading { "enabled" } else { "disabled" });
    println!();

    // Cache paths
    let cache_paths = manager.cache_paths();
    println!("{}", "Cache Paths".cyan());
    println!("  Root: {}", cache_paths.root.display());
    println!("  Platform: {}", cache_paths.platform_dir().display());
    println!("  Version: {}", cache_paths.version_dir().display());
    println!();

    // Required languages
    println!("{}", "Required Languages".cyan());
    if config.required.is_empty() {
        println!("  (none configured)");
    } else {
        for lang in &config.required {
            let status = manager.grammar_status(lang);
            let icon = match status {
                GrammarStatus::Loaded => "✓".green(),
                GrammarStatus::Cached => "●".blue(),
                GrammarStatus::NeedsDownload => "↓".yellow(),
                GrammarStatus::NotAvailable => "✗".red(),
                GrammarStatus::IncompatibleVersion => "!".yellow(),
            };
            println!("  {} {} ({:?})", icon, lang, status);
        }
    }

    Ok(())
}
