//! `language status` subcommand

use anyhow::Result;
use colored::Colorize;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

use super::helpers::{find_language, load_definitions, which_cmd};

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
        output::info("Checking all languages with LSP support...");
        output::separator();
        let defs = load_definitions();
        for def in &defs {
            if def.has_lsp() {
                check_single_language(&def.id(), &manager, verbose);
            }
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
        format!("{}/{}", cached_count, known_count),
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

/// Show detailed information about a specific language.
pub async fn language_info(language: &str) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::language_registry::types::LanguageType;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("Unknown language: {language}"));
            output::info("Use 'wqm language list' to see all known languages.");
            return Ok(());
        }
    };

    output::section(format!("Language: {}", def.language));

    // Identity
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

    // Grammar
    println!("{}", "Grammar".cyan().bold());
    if def.grammar.sources.is_empty() {
        println!("  No grammar sources configured");
    } else {
        let config = GrammarConfig::default();
        let manager = GrammarManager::new(config);
        let status = manager.grammar_status(&def.id());
        output::kv("  Status", &format!("{:?}", status));

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

    // Semantic Patterns
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

    // LSP Servers
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
                    println!("    {} {} install: {}", marker, method.manager, method.command);
                }
            }
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
