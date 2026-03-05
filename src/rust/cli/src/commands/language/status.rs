//! `language status` subcommand

use anyhow::Result;
use colored::Colorize;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

use super::helpers::{get_lsp_server_info, which_cmd};

const ALL_LANGUAGES: &[&str] = &[
    "rust",
    "python",
    "typescript",
    "go",
    "java",
    "c",
    "cpp",
    "ruby",
    "php",
    "shell",
    "html",
];

/// Show language support status (LSP + grammar availability).
pub async fn language_status(language: Option<String>, verbose: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::known_grammar_languages;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    output::section("Language Support Status");

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config.clone());
    let cached = manager.cached_languages().unwrap_or_default();
    let known = known_grammar_languages();

    print_grammar_config_summary(&config, cached.len(), known.len());

    if let Some(ref lang) = language {
        output::kv("Language", lang);
        output::separator();
        check_single_language(lang.as_str(), &manager, verbose);
    } else {
        output::info("Checking all languages...");
        output::separator();
        for lang in ALL_LANGUAGES {
            check_single_language(lang, &manager, verbose);
        }
    }

    print_daemon_language_components(verbose).await;

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn print_grammar_config_summary(
    config: &workspace_qdrant_core::config::GrammarConfig,
    cached_count: usize,
    known_count: usize,
) {
    println!("{}", "Tree-sitter Grammar Settings".cyan().bold());
    output::kv(
        "  Auto-download",
        if config.auto_download {
            "enabled"
        } else {
            "disabled"
        },
    );
    output::kv(
        "  Cached grammars",
        &format!("{}/{}", cached_count, known_count),
    );
    let idle_status = if config.idle_update_check_enabled {
        format!(
            "enabled (every {} hours, after {}s idle)",
            config.check_interval_hours, config.idle_update_check_delay_secs
        )
    } else {
        "disabled".to_string()
    };
    output::kv("  Idle update checks", &idle_status);
    output::separator();
}

fn check_single_language(
    lang: &str,
    manager: &workspace_qdrant_core::tree_sitter::GrammarManager,
    verbose: bool,
) {
    use workspace_qdrant_core::tree_sitter::GrammarStatus;

    println!("{}", format!("  {}", lang).cyan().bold());

    let lsp_info = get_lsp_server_info(lang);
    if let Some((server_name, executables)) = lsp_info {
        let lsp_path = executables.iter().find_map(|exe| which_cmd(exe));
        match lsp_path {
            Some(path) => {
                print!("    LSP: {} {}", "✓".green(), server_name);
                if verbose {
                    print!(" ({})", path);
                }
                println!();
            }
            None => println!("    LSP: {} not installed", "✗".red()),
        }
    } else {
        println!("    LSP: {} not supported", "?".yellow());
    }

    match manager.grammar_status(lang) {
        GrammarStatus::Loaded => println!("    Grammar: {} loaded", "✓".green()),
        GrammarStatus::Cached => println!("    Grammar: {} cached", "●".blue()),
        GrammarStatus::NeedsDownload => {
            println!("    Grammar: {} available (needs download)", "↓".yellow())
        }
        GrammarStatus::NotAvailable => println!("    Grammar: {} not available", "✗".red()),
        GrammarStatus::IncompatibleVersion => {
            println!("    Grammar: {} incompatible version", "!".yellow())
        }
    }

    println!();
}

async fn print_daemon_language_components(verbose: bool) {
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::separator();
            output::info("Daemon Language Components:");
            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    for comp in &health.components {
                        if comp.component_name.contains("lsp")
                            || comp.component_name.contains("grammar")
                            || comp.component_name.contains("language")
                        {
                            let status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&format!("  {}", comp.component_name), status);
                            if !comp.message.is_empty() && verbose {
                                output::kv("    Message", &comp.message);
                            }
                        }
                    }
                }
                Err(e) => output::warning(format!("Could not get daemon status: {}", e)),
            }
        }
        Err(_) => {
            output::separator();
            output::warning("Daemon not running - some status info unavailable");
        }
    }
}
