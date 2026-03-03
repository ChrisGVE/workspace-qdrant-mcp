//! `language list` subcommand

use anyhow::Result;
use colored::Colorize;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

use super::helpers::detect_available_servers;

/// List available languages with support status.
pub async fn list_languages(installed: bool, category: Option<String>, verbose: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::known_grammar_languages;
    use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus, StaticLanguageProvider};

    output::section("Language Support");

    if installed {
        output::info("Showing installed components only");
    }
    if let Some(cat) = &category {
        output::kv("Category", cat);
    }
    output::separator();

    // Check daemon connection
    let daemon_connected = DaemonClient::connect_default().await.is_ok();
    if daemon_connected {
        output::status_line("Daemon", ServiceStatus::Healthy);
    } else {
        output::status_line("Daemon", ServiceStatus::Unhealthy);
    }
    output::separator();

    // Show LSP servers available
    println!("{}", "LSP Servers".cyan().bold());
    let servers = detect_available_servers();
    if servers.is_empty() {
        if !installed {
            println!("  (none detected on PATH)");
        }
    } else {
        for (language, server_name, path) in servers {
            if verbose {
                println!("  {} {} - {} ({})", "✓".green(), language, server_name, path);
            } else {
                println!("  {} {} - {}", "✓".green(), language, server_name);
            }
        }
    }
    println!();

    // Show Tree-sitter grammars
    println!("{}", "Tree-sitter Grammars".cyan().bold());

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config.clone());
    let cached = manager.cached_languages().unwrap_or_default();

    // Static grammars (only shown when feature is enabled)
    let static_langs = StaticLanguageProvider::SUPPORTED_LANGUAGES;
    if !static_langs.is_empty() {
        println!("  {}", "Static (bundled):".dimmed());
        for lang in static_langs {
            println!("    {} {}", "✓".green(), lang);
        }
    }

    // Cached (dynamically downloaded) grammars
    if !cached.is_empty() {
        println!("  {}", "Cached (dynamic):".dimmed());
        for lang in &cached {
            let status = manager.grammar_status(lang);
            let status_icon = match status {
                GrammarStatus::Loaded => "✓".green(),
                GrammarStatus::Cached => "●".blue(),
                GrammarStatus::NeedsDownload => "↓".yellow(),
                GrammarStatus::NotAvailable => "✗".red(),
                GrammarStatus::IncompatibleVersion => "!".yellow(),
            };
            print!("    {} {}", status_icon, lang);
            if verbose {
                let info = manager.grammar_info(lang);
                if let Some(meta) = info.metadata {
                    print!(" (v{}, ts {})", meta.grammar_version, meta.tree_sitter_version);
                }
            }
            println!();
        }
    } else if !installed {
        println!("  Cached: (none)");
    }

    // Available for download
    if !installed {
        let known = known_grammar_languages();
        let downloadable: Vec<&&str> = known
            .iter()
            .filter(|l| !cached.contains(&l.to_string()) && !static_langs.contains(l))
            .collect();
        if !downloadable.is_empty() && config.auto_download {
            println!("  {}", "Available (auto-download on first use):".dimmed());
            for lang in downloadable {
                println!("    {} {}", "↓".yellow(), lang);
            }
        }
    }

    if !installed {
        output::separator();
        output::info("Install components with:");
        output::info("  wqm language lsp-install <language>  # LSP server");
        output::info("  wqm language ts-install <language>   # Tree-sitter grammar");
    }

    Ok(())
}
