//! `language list` subcommand

use anyhow::Result;
use colored::Colorize;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

use super::helpers::{detect_available_servers, load_definitions, which_cmd};

/// List available languages with support status.
pub async fn list_languages(
    installed: bool,
    category: Option<String>,
    verbose: bool,
) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::language_registry::types::LanguageType;
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

    // Load registry definitions
    let mut defs = load_definitions();

    // Filter by category/type
    if let Some(ref cat) = category {
        let type_filter = match cat.to_lowercase().as_str() {
            "programming" | "prog" => Some(LanguageType::Programming),
            "markup" | "mark" => Some(LanguageType::Markup),
            "data" | "config" => Some(LanguageType::Data),
            "prose" => Some(LanguageType::Prose),
            _ => {
                output::warning(format!("Unknown category '{cat}'. Valid: programming, markup, data, prose"));
                None
            }
        };
        if let Some(lt) = type_filter {
            defs.retain(|d| d.language_type == lt);
        }
    }

    defs.sort_by(|a, b| a.language.to_lowercase().cmp(&b.language.to_lowercase()));

    // Detect installed components
    let detected_servers = detect_available_servers();
    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config.clone());
    let cached_grammars = manager.cached_languages().unwrap_or_default();

    // Print table header
    println!(
        "  {:<16} {:<12} {:<8} {:<10} {}",
        "Language".bold(),
        "Extensions".bold(),
        "Grammar".bold(),
        "LSP".bold(),
        "Type".bold()
    );
    println!("  {}", "─".repeat(64).dimmed());

    let mut shown = 0;

    for def in &defs {
        let lang_id = def.id();

        // Grammar status
        let grammar_status = if cached_grammars.contains(&lang_id) {
            "Cached".blue()
        } else if def.has_grammar() {
            "Available".yellow()
        } else {
            "None".dimmed()
        };

        // LSP status
        let has_lsp_detected = detected_servers
            .iter()
            .any(|(l, _, _, _)| l.to_lowercase() == def.language.to_lowercase());
        let lsp_status = if has_lsp_detected {
            "Detected".green()
        } else if def.has_lsp() {
            "Available".yellow()
        } else {
            "None".dimmed()
        };

        // Filter for installed-only mode
        if installed {
            let has_cached = cached_grammars.contains(&lang_id);
            if !has_cached && !has_lsp_detected {
                continue;
            }
        }

        // Extensions (show first 3)
        let exts: String = def
            .extensions
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let exts_display = if def.extensions.len() > 3 {
            format!("{}…", exts)
        } else {
            exts
        };

        let type_label = match def.language_type {
            LanguageType::Programming => "prog",
            LanguageType::Markup => "markup",
            LanguageType::Data => "data",
            LanguageType::Prose => "prose",
        };

        println!(
            "  {:<16} {:<12} {:<8} {:<10} {}",
            def.language, exts_display, grammar_status, lsp_status, type_label.dimmed()
        );

        if verbose {
            if !def.aliases.is_empty() {
                println!("    {}: {}", "aliases".dimmed(), def.aliases.join(", "));
            }
            if def.has_semantic_patterns() {
                println!("    {}", "has semantic patterns".dimmed());
            }
            for server in &def.lsp_servers {
                let detected = which_cmd(&server.binary);
                let status = match detected {
                    Some(ref p) => format!("{} at {}", "✓".green(), p),
                    None => format!("{}", "not found".dimmed()),
                };
                println!("    LSP: {} ({}) - {}", server.name, server.binary, status);
            }
        }

        shown += 1;
    }

    println!();
    output::kv("Total", &format!("{shown} languages"));

    if !installed {
        show_install_hints();
    }

    Ok(())
}

/// Print installation hint footer.
fn show_install_hints() {
    output::separator();
    output::info("Install components with:");
    output::info("  wqm language lsp-install <language>  # LSP server");
    output::info("  wqm language ts-install <language>   # Tree-sitter grammar");
}
