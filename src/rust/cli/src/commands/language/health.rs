//! `language health` subcommand — compact overview of all language support.

use anyhow::Result;
use colored::Colorize;
use serde::Serialize;
use tabled::Tabled;

use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus};

use crate::output::{self, ColumnHints};

use super::helpers::{load_definitions, which_cmd};

/// Row for the language health table.
#[derive(Tabled, Serialize)]
struct HealthRow {
    #[tabled(rename = "Language")]
    language: String,
    #[tabled(rename = "Type")]
    lang_type: String,
    #[tabled(rename = "Grammar")]
    grammar: String,
    #[tabled(rename = "LSP")]
    lsp: String,
    #[tabled(rename = "Extensions")]
    extensions: String,
}

impl ColumnHints for HealthRow {
    fn content_columns() -> &'static [usize] {
        &[4] // Extensions column is variable-length
    }
}

/// Show a compact table of all known languages with grammar and LSP status.
pub async fn language_health() -> Result<()> {
    let defs = load_definitions();
    if defs.is_empty() {
        output::warning("No language definitions loaded");
        return Ok(());
    }

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);

    let mut rows: Vec<HealthRow> = Vec::with_capacity(defs.len());
    let mut stats = HealthStats::default();

    for def in &defs {
        let lang_id = def.id();

        let grammar_str = format_grammar_status(manager.grammar_status(&lang_id));
        let lsp_str = format_lsp_status(def);

        // Track stats
        match manager.grammar_status(&lang_id) {
            GrammarStatus::Loaded | GrammarStatus::Cached => stats.grammar_ok += 1,
            _ => stats.grammar_missing += 1,
        }
        if def
            .lsp_servers
            .iter()
            .any(|s| which_cmd(&s.binary).is_some())
        {
            stats.lsp_ok += 1;
        } else if def.has_lsp() {
            stats.lsp_missing += 1;
        } else {
            stats.lsp_none += 1;
        }

        let type_label = format!("{:?}", def.language_type);
        let exts = if def.extensions.is_empty() {
            "-".to_string()
        } else {
            def.extensions.join(", ")
        };

        rows.push(HealthRow {
            language: def.language.clone(),
            lang_type: type_label,
            grammar: grammar_str,
            lsp: lsp_str,
            extensions: exts,
        });
    }

    output::section("Language Health");
    output::print_table_auto(&rows);
    println!();
    print_health_summary(&stats, rows.len());

    Ok(())
}

fn format_grammar_status(status: GrammarStatus) -> String {
    match status {
        GrammarStatus::Loaded => format!("{} loaded", "~".green()),
        GrammarStatus::Cached => format!("{} cached", "~".blue()),
        GrammarStatus::NeedsDownload => format!("{} download", "v".yellow()),
        GrammarStatus::IncompatibleVersion => format!("{} compat", "!".yellow()),
        GrammarStatus::NotAvailable => format!("{} none", "x".red()),
    }
}

fn format_lsp_status(
    def: &workspace_qdrant_core::language_registry::types::LanguageDefinition,
) -> String {
    if def.lsp_servers.is_empty() {
        return format!("{} n/a", "-".dimmed());
    }

    // Check each server, report first found
    for server in &def.lsp_servers {
        if which_cmd(&server.binary).is_some() {
            return format!("{} {}", "~".green(), server.name);
        }
    }

    // Has servers configured but none found
    let first_name = &def.lsp_servers[0].name;
    format!("{} {}", "x".red(), first_name)
}

#[derive(Default)]
struct HealthStats {
    grammar_ok: usize,
    grammar_missing: usize,
    lsp_ok: usize,
    lsp_missing: usize,
    lsp_none: usize,
}

fn print_health_summary(stats: &HealthStats, total: usize) {
    println!(
        "  {} languages | Grammar: {} ok, {} missing | LSP: {} ok, {} missing, {} n/a",
        total.to_string().bold(),
        stats.grammar_ok.to_string().green(),
        stats.grammar_missing.to_string().yellow(),
        stats.lsp_ok.to_string().green(),
        stats.lsp_missing.to_string().yellow(),
        stats.lsp_none.to_string().dimmed(),
    );
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
}
