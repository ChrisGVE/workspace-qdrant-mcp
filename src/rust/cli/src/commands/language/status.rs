//! `language status`, `language info`, and `language refresh` subcommands
//!
//! Grammar cache status comes from the daemon's `LanguageService` (best-effort —
//! registry data still renders offline); the registry refresh is delegated to
//! the daemon's `RefreshLanguageRegistry` RPC instead of building an in-process
//! multi-source provider stack. No `workspace-qdrant-core` link (WI-e2, #82).

use anyhow::Result;
use colored::Colorize;

use wqm_common::language_registry::types::{LanguageDefinition, LanguageType};

use crate::output::{self, ServiceStatus};

use super::helpers::{
    find_language, load_definitions, try_grammar_cached_and_status, try_grammar_status, which_cmd,
};

/// Show language support status (LSP + grammar availability).
pub async fn language_status(language: Option<String>, verbose: bool) -> Result<()> {
    output::section("Language Support Status");

    let defs = load_definitions();
    let known_count = defs.iter().filter(|d| d.has_grammar()).count();
    // Best-effort live cache status from the daemon (empty when down).
    let (cached, status_map) = try_grammar_cached_and_status().await;

    print_grammar_status_summary(cached.len(), known_count);

    if let Some(ref lang) = language {
        output::kv("Language", lang);
        output::separator();
        check_single_language(lang.as_str(), &status_map, verbose);
    } else {
        output::info("Checking all languages with LSP support...");
        output::separator();
        for def in &defs {
            if def.has_lsp() {
                check_single_language(&def.id(), &status_map, verbose);
            }
        }
    }

    print_daemon_language_components(verbose).await;

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn print_grammar_status_summary(cached_count: usize, known_count: usize) {
    println!("{}", "Tree-sitter Grammar Settings".cyan().bold());
    output::kv(
        "  Cached grammars",
        format!("{}/{}", cached_count, known_count),
    );
    output::separator();
}

/// Render a single language's LSP + grammar status. `status_map` is the daemon's
/// `language id → grammar-status` map (empty when the daemon is unreachable).
fn check_single_language(
    lang: &str,
    status_map: &std::collections::HashMap<String, String>,
    verbose: bool,
) {
    println!("{}", format!("  {}", lang).cyan().bold());

    let def = find_language(lang);
    if let Some(ref def) = def {
        for server in &def.lsp_servers {
            let lsp_path = which_cmd(&server.binary);
            match lsp_path {
                Some(path) => {
                    print!("    LSP: {} {}", "✓".green(), server.name);
                    if verbose {
                        print!(" ({})", path);
                    }
                    println!();
                }
                None => println!("    LSP: {} {} not installed", "✗".red(), server.name),
            }
        }
        if def.lsp_servers.is_empty() {
            println!("    LSP: {} not configured", "?".yellow());
        }
    } else {
        println!("    LSP: {} unknown language", "?".yellow());
    }

    let status = status_map
        .get(lang)
        .map(String::as_str)
        .unwrap_or("not_available");
    match status {
        "loaded" => println!("    Grammar: {} loaded", "✓".green()),
        "cached" => println!("    Grammar: {} cached", "●".blue()),
        "needs_download" => println!("    Grammar: {} available (needs download)", "↓".yellow()),
        "incompatible_version" => println!("    Grammar: {} incompatible version", "!".yellow()),
        _ => println!("    Grammar: {} not available", "✗".red()),
    }

    println!();
}

/// Show detailed information about a specific language.
pub async fn language_info(language: &str) -> Result<()> {
    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("Unknown language: {language}"));
            output::info("Use 'wqm language list' to see all known languages.");
            return Ok(());
        }
    };

    output::section(format!("Language: {}", def.language));

    let grammar_status = try_grammar_status(&def.id()).await;
    print_identity_section(&def);
    print_grammar_section(&def, &grammar_status);
    print_semantic_patterns_section(&def);
    print_lsp_servers_section(&def);

    Ok(())
}

fn print_identity_section(def: &LanguageDefinition) {
    println!("{}", "Identity".cyan().bold());
    output::kv("  Name", &def.language);
    output::kv("  ID", &def.id());
    if !def.aliases.is_empty() {
        output::kv("  Aliases", &def.aliases.join(", "));
    }
    if !def.extensions.is_empty() {
        output::kv("  Extensions", &def.extensions.join(", "));
    }
    let type_label = match def.language_type {
        LanguageType::Programming => "Programming",
        LanguageType::Markup => "Markup",
        LanguageType::Data => "Data",
        LanguageType::Prose => "Prose",
    };
    output::kv("  Type", type_label);
    println!();
}

fn print_grammar_section(def: &LanguageDefinition, grammar_status: &str) {
    println!("{}", "Grammar".cyan().bold());
    if def.grammar.sources.is_empty() {
        println!("  No grammar sources configured");
    } else {
        output::kv("  Status", grammar_status);

        for (i, src) in def.grammar.sources.iter().enumerate() {
            let quality = format!("{:?}", src.quality);
            let origin = src.origin.as_deref().unwrap_or("unknown");
            println!(
                "  {}. {} ({}, from {})",
                i + 1,
                src.repo,
                quality.to_lowercase(),
                origin
            );
        }
        if def.grammar.has_cpp_scanner {
            println!("  {}", "Requires C++ compiler for scanner".dimmed());
        }
        if let Some(ref sub) = def.grammar.src_subdir {
            output::kv("  Source subdir", sub);
        }
    }
    println!();
}

fn print_semantic_patterns_section(def: &LanguageDefinition) {
    println!("{}", "Semantic Patterns".cyan().bold());
    if let Some(ref patterns) = def.semantic_patterns {
        let docstyle = format!("{:?}", patterns.docstring_style);
        output::kv("  Docstring style", &docstyle);
        if let Some(ref name_node) = patterns.name_node {
            output::kv("  Name node", name_node);
        }
        if let Some(ref body_node) = patterns.body_node {
            output::kv("  Body node", body_node);
        }
        print_pattern_line("  Functions", &patterns.function.node_types);
        print_pattern_line("  Async fns", &patterns.function.async_node_types);
        print_pattern_line("  Classes", &patterns.class.node_types);
        print_pattern_line("  Methods", &patterns.method.node_types);
        print_pattern_line("  Structs", &patterns.struct_def.node_types);
        print_pattern_line("  Enums", &patterns.enum_def.node_types);
        print_pattern_line("  Traits", &patterns.trait_def.node_types);
        print_pattern_line("  Interfaces", &patterns.interface.node_types);
        print_pattern_line("  Modules", &patterns.module.node_types);
        print_pattern_line("  Constants", &patterns.constant.node_types);
        print_pattern_line("  Macros", &patterns.macro_def.node_types);
        print_pattern_line("  Type aliases", &patterns.type_alias.node_types);
        print_pattern_line("  Impl blocks", &patterns.impl_block.node_types);
        print_pattern_line("  Preamble", &patterns.preamble.node_types);
    } else {
        println!("  Not configured (text chunking fallback)");
    }
    println!();
}

fn print_lsp_servers_section(def: &LanguageDefinition) {
    println!("{}", "LSP Servers".cyan().bold());
    if def.lsp_servers.is_empty() {
        println!("  No LSP servers configured");
    } else {
        for server in &def.lsp_servers {
            let detected = which_cmd(&server.binary);
            let status_str = match &detected {
                Some(p) => format!("{} at {}", "✓".green(), p),
                None => format!("{}", "not found".red()),
            };
            println!(
                "  {} ({}) - {} [priority: {}]",
                server.name, server.binary, status_str, server.priority
            );
            if !server.install_methods.is_empty() {
                for method in &server.install_methods {
                    let available = which_cmd(&method.manager).is_some();
                    let marker = if available {
                        "▸".green().to_string()
                    } else {
                        "▸".dimmed().to_string()
                    };
                    println!(
                        "    {} {} install: {}",
                        marker, method.manager, method.command
                    );
                }
            }
        }
    }
}

/// Refresh the language registry via the daemon's `RefreshLanguageRegistry` RPC.
///
/// The multi-source upstream provider stack (Linguist / Mason / nvim-treesitter
/// / tree-sitter-grammars / bundled) lives daemon-side now (the daemon owns the
/// registry + grammar cache); the CLI no longer builds it in-process.
pub async fn language_refresh() -> Result<()> {
    output::section("Refreshing Language Registry");

    let mut client = crate::grpc::ensure_daemon_available().await?;
    output::info("Requesting registry refresh from the daemon...");

    match client.refresh_language_registry().await {
        Ok(s) => {
            output::separator();
            output::success("Registry refreshed successfully");
            output::kv("Total languages", format!("{}", s.total));
            output::kv("With grammars", format!("{}", s.with_grammars));
            output::kv("With LSP servers", format!("{}", s.with_lsp));
            output::kv(
                "With semantic patterns",
                format!("{}", s.with_semantic_patterns),
            );
        }
        Err(e) => {
            output::warning(format!("Registry refresh failed: {}", e.message()));
        }
    }

    Ok(())
}

fn print_pattern_line(label: &str, types: &[String]) {
    if !types.is_empty() {
        output::kv(label, &types.join(", "));
    }
}

async fn print_daemon_language_components(verbose: bool) {
    match crate::grpc::connect_default().await {
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
                            output::status_line(format!("  {}", comp.component_name), status);
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
