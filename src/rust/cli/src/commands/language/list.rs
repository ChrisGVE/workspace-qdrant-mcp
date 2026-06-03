//! `language list` subcommand

use anyhow::Result;
use colored::Colorize;
use serde::Serialize;
use tabled::Tabled;

use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::language_registry::types::{LanguageDefinition, LanguageType};
use workspace_qdrant_core::lsp::detection::editor_paths::DetectionSource;
use workspace_qdrant_core::tree_sitter::GrammarManager;

use crate::output::{self, ColumnHints, ServiceStatus};

use super::helpers::{detect_available_servers, load_definitions, which_cmd};

/// Row for the language list table.
#[derive(Tabled, Serialize)]
struct LanguageRow {
    #[tabled(rename = "Language")]
    language: String,
    #[tabled(rename = "Extensions")]
    extensions: String,
    #[tabled(rename = "Grammar")]
    grammar: String,
    #[tabled(rename = "LSP")]
    lsp: String,
    #[tabled(rename = "Type")]
    lang_type: String,
}

impl ColumnHints for LanguageRow {
    fn content_columns() -> &'static [usize] {
        &[1] // Extensions column is variable-length
    }
}

/// List available languages with support status.
pub async fn list_languages(
    installed: bool,
    category: Option<String>,
    verbose: bool,
) -> Result<()> {
    output::section("Language Support");

    if installed {
        output::info("Showing installed components only");
    }
    if let Some(cat) = &category {
        output::kv("Category", cat);
    }
    output::separator();

    let daemon_connected = crate::grpc::connect_default().await.is_ok();
    let status = if daemon_connected {
        ServiceStatus::Healthy
    } else {
        ServiceStatus::Unhealthy
    };
    output::status_line("Daemon", status);
    output::separator();

    let mut defs = load_definitions();
    filter_by_category(&mut defs, &category);
    defs.sort_by_key(|d| d.language.to_lowercase());

    let detected_servers = detect_available_servers();
    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config.clone());
    let cached_grammars = manager.cached_languages().unwrap_or_default();

    let rows = build_language_rows(&defs, &cached_grammars, &detected_servers, installed);

    if verbose {
        output::print_table_auto(&rows);
        println!();
        print_verbose_details(&defs, &cached_grammars, &detected_servers, installed);
    } else {
        output::print_table_auto(&rows);
    }

    println!();
    output::kv("Total", &format!("{} languages", rows.len()));

    if !installed {
        show_install_hints();
    }

    Ok(())
}

/// Filter definitions by language category/type.
fn filter_by_category(defs: &mut Vec<LanguageDefinition>, category: &Option<String>) {
    let Some(cat) = category else { return };
    let type_filter = match cat.to_lowercase().as_str() {
        "programming" | "prog" => Some(LanguageType::Programming),
        "markup" | "mark" => Some(LanguageType::Markup),
        "data" | "config" => Some(LanguageType::Data),
        "prose" => Some(LanguageType::Prose),
        _ => {
            output::warning(format!(
                "Unknown category '{cat}'. Valid: programming, markup, data, prose"
            ));
            None
        }
    };
    if let Some(lt) = type_filter {
        defs.retain(|d| d.language_type == lt);
    }
}

/// Build table rows from language definitions.
fn build_language_rows(
    defs: &[LanguageDefinition],
    cached_grammars: &[String],
    detected_servers: &[(String, String, String, DetectionSource)],
    installed: bool,
) -> Vec<LanguageRow> {
    let mut rows = Vec::new();

    for def in defs {
        let lang_id = def.id();

        let grammar = format_grammar_cell(def, cached_grammars, &lang_id);
        let (lsp, has_lsp_detected) = format_lsp_cell(def, detected_servers);

        if installed {
            let has_cached = cached_grammars.contains(&lang_id);
            if !has_cached && !has_lsp_detected {
                continue;
            }
        }

        let extensions = format_extensions(&def.extensions);
        let lang_type = format_language_type(&def.language_type);

        rows.push(LanguageRow {
            language: def.language.clone(),
            extensions,
            grammar,
            lsp,
            lang_type,
        });
    }

    rows
}

fn format_grammar_cell(def: &LanguageDefinition, cached: &[String], lang_id: &str) -> String {
    if cached.contains(&lang_id.to_string()) {
        format!("{}", "Cached".blue())
    } else if def.has_grammar() {
        format!("{}", "Available".yellow())
    } else {
        format!("{}", "None".dimmed())
    }
}

fn format_lsp_cell(
    def: &LanguageDefinition,
    detected_servers: &[(String, String, String, DetectionSource)],
) -> (String, bool) {
    let has_lsp_detected = detected_servers
        .iter()
        .any(|(l, _, _, _)| l.to_lowercase() == def.language.to_lowercase());
    let lsp = if has_lsp_detected {
        format!("{}", "Detected".green())
    } else if def.has_lsp() {
        format!("{}", "Available".yellow())
    } else {
        format!("{}", "None".dimmed())
    };
    (lsp, has_lsp_detected)
}

fn format_extensions(extensions: &[String]) -> String {
    let exts: String = extensions
        .iter()
        .take(3)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if extensions.len() > 3 {
        format!("{exts}…")
    } else {
        exts
    }
}

fn format_language_type(lang_type: &LanguageType) -> String {
    match lang_type {
        LanguageType::Programming => "prog".to_string(),
        LanguageType::Markup => "markup".to_string(),
        LanguageType::Data => "data".to_string(),
        LanguageType::Prose => "prose".to_string(),
    }
}

/// Print verbose details for each language (aliases, semantic patterns, LSP paths).
fn print_verbose_details(
    defs: &[LanguageDefinition],
    cached_grammars: &[String],
    detected_servers: &[(String, String, String, DetectionSource)],
    installed: bool,
) {
    for def in defs {
        let lang_id = def.id();

        // Skip non-installed in installed-only mode
        if installed {
            let has_cached = cached_grammars.contains(&lang_id);
            let has_lsp_detected = detected_servers
                .iter()
                .any(|(l, _, _, _)| l.to_lowercase() == def.language.to_lowercase());
            if !has_cached && !has_lsp_detected {
                continue;
            }
        }

        let mut has_detail = false;

        if !def.aliases.is_empty() {
            if !has_detail {
                println!("  {}:", def.language.bold());
            }
            println!("    {}: {}", "aliases".dimmed(), def.aliases.join(", "));
            has_detail = true;
        }
        if def.has_semantic_patterns() {
            if !has_detail {
                println!("  {}:", def.language.bold());
            }
            println!("    {}", "has semantic patterns".dimmed());
            has_detail = true;
        }
        for server in &def.lsp_servers {
            let detected = which_cmd(&server.binary);
            let status = match detected {
                Some(ref p) => format!("{} at {}", "✓".green(), p),
                None => format!("{}", "not found".dimmed()),
            };
            if !has_detail {
                println!("  {}:", def.language.bold());
                has_detail = true;
            }
            println!("    LSP: {} ({}) - {}", server.name, server.binary, status);
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
