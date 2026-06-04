//! Language diagnostics subcommand
//!
//! Checks LSP server availability, tree-sitter grammar presence,
//! daemon language support status, and file extension mappings.
//!
//! All language metadata is sourced from the bundled language registry via
//! `wqm-common`; grammar cache status comes from the daemon's LanguageService
//! (best-effort) — no `workspace-qdrant-core` link, no hardcoded language lists.

use anyhow::Result;
use std::process::Command;

use wqm_common::language_registry::types::LanguageDefinition;
use wqm_common::lsp_detection::find_lsp_binary;

use crate::commands::language::helpers::{find_language, load_definitions, try_grammar_status};
use crate::output::{self, ServiceStatus};

/// Diagnose language support issues.
pub async fn diagnose_language(language: &str, verbose: bool) -> Result<()> {
    output::section(format!("Language Diagnostics: {}", language));

    output::info("Running diagnostic checks...");
    output::separator();

    // 1. Check LSP server
    let lsp_found = check_lsp_server(language, verbose);

    output::separator();

    // 2. Check Tree-sitter grammar
    let grammar_found = check_tree_sitter_grammar(language, verbose).await;

    output::separator();

    // 3. Check daemon language support status
    check_daemon_language_support(language, verbose).await;

    output::separator();

    // 4. File extension mapping
    show_extension_mapping(language);

    output::separator();
    show_diagnostic_summary(language, lsp_found, grammar_found);

    Ok(())
}

/// Check for LSP server binaries on PATH using the language registry.
fn check_lsp_server(language: &str, verbose: bool) -> bool {
    output::info("1. LSP Server Check");

    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("  Unknown language: {}", language));
            return false;
        }
    };

    if def.lsp_servers.is_empty() {
        output::warning(format!("  No LSP servers defined for {}", def.language));
        return false;
    }

    let mut lsp_found = false;
    for server in &def.lsp_servers {
        match find_lsp_binary(&server.binary) {
            Some(result) => {
                output::success(format!(
                    "  Found: {} ({}) at {}",
                    server.name,
                    server.binary,
                    result.path.display()
                ));
                lsp_found = true;

                if verbose {
                    output::kv("    Source", format!("{:?}", result.source));
                    if let Ok(ver_output) = Command::new(&server.binary).arg("--version").output() {
                        if ver_output.status.success() {
                            let version = String::from_utf8_lossy(&ver_output.stdout);
                            output::kv("    Version", version.trim());
                        }
                    }
                }
                break;
            }
            None => {
                if verbose {
                    output::info(format!("  Not found: {} ({})", server.name, server.binary));
                }
            }
        }
    }

    if !lsp_found {
        output::warning(format!("  No LSP server found for {}", def.language));
        show_lsp_install_suggestions(&def);
    }

    lsp_found
}

/// Print LSP install suggestions from the language registry.
fn show_lsp_install_suggestions(def: &LanguageDefinition) {
    let mut has_suggestions = false;
    for server in &def.lsp_servers {
        if !server.install_methods.is_empty() {
            if !has_suggestions {
                output::info("  Install suggestions:");
                has_suggestions = true;
            }
            for method in &server.install_methods {
                output::info(format!("    {}", method.command));
            }
        }
    }
    if !has_suggestions {
        output::info(format!(
            "  Search for a language server for {}",
            def.language
        ));
    }
}

/// Check tree-sitter grammar availability via the daemon's LanguageService
/// (best-effort; reports not-available when the daemon is unreachable).
async fn check_tree_sitter_grammar(language: &str, verbose: bool) -> bool {
    output::info("2. Tree-sitter Grammar Check");

    let lang_id = resolve_grammar_language_id(language);
    let status = try_grammar_status(&lang_id).await;

    match status.as_str() {
        "loaded" => {
            output::success(format!("  Grammar loaded: {}", lang_id));
            true
        }
        "cached" => {
            output::success(format!("  Grammar cached: {}", lang_id));
            true
        }
        "needs_download" => {
            output::warning(format!("  Grammar not cached: {}", lang_id));
            output::info("  Install with: wqm language ts-install <language>");
            false
        }
        "incompatible_version" => {
            output::warning(format!(
                "  Grammar cached but incompatible version: {}",
                lang_id
            ));
            output::info("  Reinstall with: wqm language ts-install --force <language>");
            false
        }
        _ => {
            output::warning(format!(
                "  No tree-sitter grammar available for {}",
                lang_id
            ));
            if verbose {
                output::info("  This language may not have tree-sitter support");
            }
            false
        }
    }
}

/// Resolve user-supplied language name to a grammar language ID.
///
/// Uses the language registry for alias resolution, falling back to
/// lowercase normalization for unknown languages.
fn resolve_grammar_language_id(language: &str) -> String {
    if let Some(def) = find_language(language) {
        def.id().to_string()
    } else {
        language.to_lowercase()
    }
}

/// Check daemon language support status via gRPC health endpoint.
async fn check_daemon_language_support(language: &str, verbose: bool) {
    output::info("3. Daemon Language Support");

    let lang_lower = language.to_lowercase();

    match crate::grpc::connect_default().await {
        Ok(mut client) => match client.system().health(()).await {
            Ok(response) => {
                let health = response.into_inner();

                for comp in &health.components {
                    if comp.component_name.to_lowercase().contains(&lang_lower)
                        || comp.component_name.contains("lsp")
                        || comp.component_name.contains("grammar")
                    {
                        let status = ServiceStatus::from_proto(comp.status);
                        output::status_line(format!("  {}", comp.component_name), status);
                        if !comp.message.is_empty() && verbose {
                            output::kv("    Details", &comp.message);
                        }
                    }
                }
            }
            Err(e) => {
                output::warning(format!("  Could not get daemon status: {}", e));
            }
        },
        Err(_) => {
            output::warning("  Daemon not running - cannot check language support status");
        }
    }
}

/// Show file extension mapping from the language registry.
fn show_extension_mapping(language: &str) {
    output::info("4. File Extension Mapping");

    match find_language(language) {
        Some(def) => {
            if !def.extensions.is_empty() {
                output::kv("  Extensions", def.extensions.join(", "));
            } else {
                output::info(format!("  No file extensions defined for {}", def.language));
            }
        }
        None => {
            // Fallback: scan all definitions for any that match as alias
            let defs = load_definitions();
            let normalized = language.to_lowercase();
            let matching: Vec<_> = defs
                .iter()
                .filter(|d| {
                    d.extensions
                        .iter()
                        .any(|ext| ext.trim_start_matches('.') == normalized)
                })
                .collect();

            if matching.is_empty() {
                output::info(format!("  Unknown language: {}", language));
            } else {
                for def in matching {
                    output::kv(
                        format!("  {} extensions", def.language),
                        def.extensions.join(", "),
                    );
                }
            }
        }
    }
}

/// Show diagnostic summary based on what was found.
fn show_diagnostic_summary(language: &str, lsp_found: bool, grammar_found: bool) {
    output::info("Diagnostic Summary:");

    if lsp_found && grammar_found {
        output::success(format!("  {} support appears fully configured", language));
    } else if lsp_found {
        output::warning(format!("  {} has LSP but missing grammar", language));
    } else if grammar_found {
        output::warning(format!("  {} has grammar but missing LSP", language));
    } else {
        output::error(format!("  {} support not configured", language));
    }
}
