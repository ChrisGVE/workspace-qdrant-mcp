//! `language query` subcommand — explore the language registry.
//!
//! Without args: compact table of all languages with grammar/LSP status.
//! With `<lang>` arg: detailed view including grammar sources, LSP servers,
//! install instructions, and current user preference.

use anyhow::Result;
use colored::Colorize;
use serde::Serialize;
use tabled::Tabled;

use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus};

use crate::output::{self, ColumnHints};

use super::helpers::{find_language, load_definitions, which_cmd};
use super::preferences;

/// List all languages with grammar/LSP status and user preferences.
pub async fn query_all() -> Result<()> {
    let defs = load_definitions();
    if defs.is_empty() {
        output::warning("No language definitions loaded");
        return Ok(());
    }

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);
    let prefs = preferences::load_preferences().unwrap_or_default();

    let mut rows: Vec<QueryRow> = Vec::with_capacity(defs.len());

    for def in &defs {
        let lang_id = def.id();
        let grammar_str = format_grammar_status(manager.grammar_status(&lang_id));
        let lsp_str = format_lsp_status(def);
        let pref_str = format_preference(&lang_id, &prefs);

        let exts = if def.extensions.is_empty() {
            "-".to_string()
        } else {
            def.extensions.join(", ")
        };

        rows.push(QueryRow {
            language: def.language.clone(),
            lang_type: format!("{}", def.language_type),
            grammar: grammar_str,
            lsp: lsp_str,
            preference: pref_str,
            extensions: exts,
        });
    }

    output::section("Language Registry");
    output::print_table_auto(&rows);

    println!();
    print_summary(&rows);
    Ok(())
}

/// Show detailed information about a specific language.
pub async fn query_language(language: &str) -> Result<()> {
    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("Unknown language: {language}"));
            output::info("Use 'wqm language query' to see all known languages.");
            return Ok(());
        }
    };

    let lang_id = def.id();
    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);
    let prefs = preferences::load_preferences().unwrap_or_default();

    output::section(format!("Language: {}", def.language));

    // Identity
    println!("{}", "Identity".cyan().bold());
    output::kv("  Name", &def.language);
    output::kv("  ID", &lang_id);
    if !def.aliases.is_empty() {
        output::kv("  Aliases", &def.aliases.join(", "));
    }
    if !def.extensions.is_empty() {
        output::kv("  Extensions", &def.extensions.join(", "));
    }
    output::kv("  Type", format!("{}", def.language_type));
    println!();

    // Grammar sources
    println!("{}", "Grammar Sources".cyan().bold());
    let grammar_status = manager.grammar_status(&lang_id);
    output::kv("  Status", format_grammar_verbose(grammar_status));

    let resolved_grammar = preferences::resolve_grammar(&lang_id);
    if let Some(ref repo) = resolved_grammar {
        output::kv("  Active", repo);
    }

    if def.grammar.sources.is_empty() {
        println!("  No grammar sources configured");
    } else {
        for (i, src) in def.grammar.sources.iter().enumerate() {
            let quality = format!("{}", src.quality);
            let origin = src.origin.as_deref().unwrap_or("unknown");
            let active = resolved_grammar
                .as_deref()
                .map_or(i == 0, |r| r == src.repo);
            let marker = if active {
                "▸".green().to_string()
            } else {
                " ".to_string()
            };
            println!(
                "  {} {}. {} ({}, from {})",
                marker,
                i + 1,
                src.repo,
                quality,
                origin
            );
        }
    }
    if def.grammar.has_cpp_scanner {
        println!("  {}", "Requires C++ compiler for scanner".dimmed());
    }
    println!();

    // LSP servers
    println!("{}", "LSP Servers".cyan().bold());
    let resolved_lsp = preferences::resolve_lsp(&lang_id);

    if def.lsp_servers.is_empty() {
        println!("  No LSP servers configured");
    } else {
        for server in &def.lsp_servers {
            let detected = which_cmd(&server.binary);
            let status_str = match &detected {
                Some(p) => format!("{} at {}", "installed".green(), p),
                None => "not found".red().to_string(),
            };
            let active = resolved_lsp
                .as_deref()
                .map_or(false, |r| r.eq_ignore_ascii_case(&server.name));
            let marker = if active {
                "▸".green().to_string()
            } else {
                " ".to_string()
            };
            println!(
                "  {} {} ({}) — {} [priority: {}]",
                marker, server.name, server.binary, status_str, server.priority
            );
            if !server.install_methods.is_empty() {
                for method in &server.install_methods {
                    let available = which_cmd(&method.manager).is_some();
                    let icon = if available {
                        "▸".green().to_string()
                    } else {
                        "▸".dimmed().to_string()
                    };
                    println!("      {} {}: {}", icon, method.manager, method.command);
                }
            }
        }
    }
    println!();

    // User preference
    println!("{}", "User Preference".cyan().bold());
    if let Some(pref) = prefs.languages.get(&lang_id) {
        if let Some(ref l) = pref.lsp {
            output::kv("  LSP", l);
        }
        if let Some(ref g) = pref.grammar {
            output::kv("  Grammar", g);
        }
    } else {
        println!("  None (using registry defaults)");
        output::info(
            "  Set with: wqm language preferences set <lang> --lsp <server> --grammar <repo>",
        );
    }

    Ok(())
}

// ── Formatting helpers ───────────────────────────────────────────────

fn format_grammar_status(status: GrammarStatus) -> String {
    match status {
        GrammarStatus::Loaded => format!("{} loaded", "~".green()),
        GrammarStatus::Cached => format!("{} cached", "~".blue()),
        GrammarStatus::NeedsDownload => format!("{} download", "v".yellow()),
        GrammarStatus::IncompatibleVersion => format!("{} compat", "!".yellow()),
        GrammarStatus::NotAvailable => format!("{} none", "x".red()),
    }
}

fn format_grammar_verbose(status: GrammarStatus) -> String {
    match status {
        GrammarStatus::Loaded => "Loaded (in memory)".green().to_string(),
        GrammarStatus::Cached => "Cached (on disk)".blue().to_string(),
        GrammarStatus::NeedsDownload => "Available (needs download)".yellow().to_string(),
        GrammarStatus::IncompatibleVersion => "Incompatible version".yellow().to_string(),
        GrammarStatus::NotAvailable => "Not available".red().to_string(),
    }
}

fn format_lsp_status(
    def: &workspace_qdrant_core::language_registry::types::LanguageDefinition,
) -> String {
    if def.lsp_servers.is_empty() {
        return format!("{} n/a", "-".dimmed());
    }
    for server in &def.lsp_servers {
        if which_cmd(&server.binary).is_some() {
            return format!("{} {}", "~".green(), server.name);
        }
    }
    let first_name = &def.lsp_servers[0].name;
    format!("{} {}", "x".red(), first_name)
}

fn format_preference(lang_id: &str, prefs: &preferences::LanguagePreferences) -> String {
    match prefs.languages.get(lang_id) {
        Some(pref) => {
            let mut parts = Vec::new();
            if let Some(ref l) = pref.lsp {
                parts.push(format!("lsp:{l}"));
            }
            if let Some(ref g) = pref.grammar {
                parts.push(format!("grammar:{g}"));
            }
            if parts.is_empty() {
                "-".to_string()
            } else {
                parts.join(", ")
            }
        }
        None => "-".to_string(),
    }
}

fn print_summary(rows: &[QueryRow]) {
    let total = rows.len();
    let with_prefs = rows.iter().filter(|r| r.preference != "-").count();
    if with_prefs > 0 {
        println!(
            "  {} languages | {} with user preferences",
            total.to_string().bold(),
            with_prefs.to_string().cyan(),
        );
    } else {
        println!("  {} languages", total.to_string().bold());
    }
}

// ── Table row ────────────────────────────────────────────────────────

#[derive(Tabled, Serialize)]
struct QueryRow {
    #[tabled(rename = "Language")]
    language: String,
    #[tabled(rename = "Type")]
    lang_type: String,
    #[tabled(rename = "Grammar")]
    grammar: String,
    #[tabled(rename = "LSP")]
    lsp: String,
    #[tabled(rename = "Preference")]
    preference: String,
    #[tabled(rename = "Extensions")]
    extensions: String,
}

impl ColumnHints for QueryRow {
    fn content_columns() -> &'static [usize] {
        &[4, 5] // Preference and Extensions columns are variable-length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_grammar_status_variants() {
        assert!(!format_grammar_status(GrammarStatus::Loaded).is_empty());
        assert!(!format_grammar_status(GrammarStatus::Cached).is_empty());
        assert!(!format_grammar_status(GrammarStatus::NeedsDownload).is_empty());
        assert!(!format_grammar_status(GrammarStatus::IncompatibleVersion).is_empty());
        assert!(!format_grammar_status(GrammarStatus::NotAvailable).is_empty());
    }

    #[test]
    fn format_preference_empty() {
        let prefs = preferences::LanguagePreferences::default();
        assert_eq!(format_preference("rust", &prefs), "-");
    }

    #[test]
    fn format_preference_with_values() {
        let mut prefs = preferences::LanguagePreferences::default();
        prefs.languages.insert(
            "rust".into(),
            preferences::LanguagePreference {
                lsp: Some("rust-analyzer".into()),
                grammar: None,
            },
        );
        let result = format_preference("rust", &prefs);
        assert!(result.contains("lsp:rust-analyzer"));
    }
}
