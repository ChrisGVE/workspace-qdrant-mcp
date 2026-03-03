//! `language list` subcommand

use anyhow::Result;
use colored::Colorize;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

use super::helpers::detect_available_servers;

/// List available languages with support status.
pub async fn list_languages(installed: bool, category: Option<String>, verbose: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    output::section("Language Support");

    if installed {
        output::info("Showing installed components only");
    }
    if let Some(cat) = &category {
        output::kv("Category", cat);
    }
    output::separator();

    let daemon_connected = DaemonClient::connect_default().await.is_ok();
    if daemon_connected {
        output::status_line("Daemon", ServiceStatus::Healthy);
    } else {
        output::status_line("Daemon", ServiceStatus::Unhealthy);
    }
    output::separator();

    show_lsp_servers(installed, verbose);

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config.clone());
    show_grammar_status(&manager, &config, installed, verbose);

    if !installed {
        show_install_hints();
    }

    Ok(())
}

/// Print the LSP servers section.
fn show_lsp_servers(installed: bool, verbose: bool) {
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
}

/// Print the Tree-sitter grammar section (static, cached, downloadable).
fn show_grammar_status(
    manager: &workspace_qdrant_core::tree_sitter::GrammarManager,
    config: &workspace_qdrant_core::config::GrammarConfig,
    installed: bool,
    verbose: bool,
) {
    use workspace_qdrant_core::known_grammar_languages;
    use workspace_qdrant_core::tree_sitter::{GrammarStatus, StaticLanguageProvider};

    println!("{}", "Tree-sitter Grammars".cyan().bold());

    let cached = manager.cached_languages().unwrap_or_default();
    let static_langs = StaticLanguageProvider::SUPPORTED_LANGUAGES;

    if !static_langs.is_empty() {
        println!("  {}", "Static (bundled):".dimmed());
        for lang in static_langs {
            println!("    {} {}", "✓".green(), lang);
        }
    }

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
}

/// Print installation hint footer.
fn show_install_hints() {
    output::separator();
    output::info("Install components with:");
    output::info("  wqm language lsp-install <language>  # LSP server");
    output::info("  wqm language ts-install <language>   # Tree-sitter grammar");
}
