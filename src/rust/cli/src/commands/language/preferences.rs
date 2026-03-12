//! `language preferences` subcommand — manage per-language overrides.
//!
//! Preferences are stored in `~/.workspace-qdrant/language_preferences.yaml`.
//! Resolution order: user preference → registry default (highest tier) → fallback.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use crate::output::{self, ColumnHints};

use super::helpers::find_language;

// ── Data types ───────────────────────────────────────────────────────

/// Per-language user preference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguagePreference {
    /// Preferred LSP server name (must match a registry server name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lsp: Option<String>,
    /// Preferred grammar repo (e.g., "tree-sitter/tree-sitter-rust").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
}

/// All user language preferences, keyed by language ID.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LanguagePreferences {
    #[serde(default, flatten)]
    pub languages: BTreeMap<String, LanguagePreference>,
}

// ── File I/O ─────────────────────────────────────────────────────────

fn preferences_path() -> Result<PathBuf> {
    let dir = wqm_common::paths::get_config_dir().context("cannot determine config directory")?;
    Ok(dir.join("language_preferences.yaml"))
}

pub fn load_preferences() -> Result<LanguagePreferences> {
    let path = preferences_path()?;
    if !path.exists() {
        return Ok(LanguagePreferences::default());
    }
    let content =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    if content.trim().is_empty() {
        return Ok(LanguagePreferences::default());
    }
    let prefs: LanguagePreferences =
        serde_yaml_ng::from_str(&content).with_context(|| format!("parsing {}", path.display()))?;
    Ok(prefs)
}

fn save_preferences(prefs: &LanguagePreferences) -> Result<()> {
    let path = preferences_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let yaml = serde_yaml_ng::to_string(prefs).context("serializing preferences")?;
    std::fs::write(&path, yaml).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

// ── Subcommands ──────────────────────────────────────────────────────

/// Set a preference for a language.
pub async fn preferences_set(
    language: &str,
    lsp: Option<String>,
    grammar: Option<String>,
) -> Result<()> {
    if lsp.is_none() && grammar.is_none() {
        output::warning("Specify --lsp <server> and/or --grammar <repo>");
        return Ok(());
    }

    let normalized = language.to_lowercase();

    // Validate language exists in registry
    if find_language(&normalized).is_none() {
        output::warning(format!("Unknown language: {language}"));
        output::info("Use 'wqm language list' to see known languages.");
        return Ok(());
    }

    // Validate LSP server name if provided
    if let Some(ref lsp_name) = lsp {
        if let Some(def) = find_language(&normalized) {
            let known: Vec<&str> = def.lsp_servers.iter().map(|s| s.name.as_str()).collect();
            if !known.iter().any(|n| n.eq_ignore_ascii_case(lsp_name)) {
                output::warning(format!(
                    "LSP server '{lsp_name}' not in registry for {language}"
                ));
                if known.is_empty() {
                    output::info("No LSP servers configured for this language.");
                } else {
                    output::info(format!("Available: {}", known.join(", ")));
                }
                return Ok(());
            }
        }
    }

    // Validate grammar repo if provided
    if let Some(ref grammar_repo) = grammar {
        if let Some(def) = find_language(&normalized) {
            let known: Vec<&str> = def
                .grammar
                .sources
                .iter()
                .map(|s| s.repo.as_str())
                .collect();
            if !known.iter().any(|r| r == grammar_repo) {
                output::warning(format!(
                    "Grammar repo '{grammar_repo}' not in registry for {language}"
                ));
                if known.is_empty() {
                    output::info("No grammar sources configured for this language.");
                } else {
                    output::info(format!("Available: {}", known.join(", ")));
                }
                return Ok(());
            }
        }
    }

    let mut prefs = load_preferences()?;
    let entry = prefs
        .languages
        .entry(normalized.clone())
        .or_insert_with(|| LanguagePreference {
            lsp: None,
            grammar: None,
        });
    if let Some(ref l) = lsp {
        entry.lsp = Some(l.clone());
    }
    if let Some(ref g) = grammar {
        entry.grammar = Some(g.clone());
    }
    save_preferences(&prefs)?;

    output::success(format!("Preference set for {normalized}"));
    if let Some(ref l) = lsp {
        output::kv("  LSP", l);
    }
    if let Some(ref g) = grammar {
        output::kv("  Grammar", g);
    }
    Ok(())
}

/// List all user preferences.
pub async fn preferences_list() -> Result<()> {
    let prefs = load_preferences()?;
    if prefs.languages.is_empty() {
        output::info("No language preferences set.");
        output::info("Use 'wqm language preferences set <lang> --lsp <server>' to set one.");
        return Ok(());
    }

    output::section("Language Preferences");

    let rows: Vec<PrefRow> = prefs
        .languages
        .iter()
        .map(|(lang, pref)| PrefRow {
            language: lang.clone(),
            lsp: pref.lsp.clone().unwrap_or_else(|| "-".into()),
            grammar: pref.grammar.clone().unwrap_or_else(|| "-".into()),
        })
        .collect();

    output::print_table_auto(&rows);

    let path = preferences_path()?;
    println!();
    output::kv("File", path.display());
    Ok(())
}

/// Reset preferences for a language.
pub async fn preferences_reset(language: &str) -> Result<()> {
    let normalized = language.to_lowercase();
    let mut prefs = load_preferences()?;

    if prefs.languages.remove(&normalized).is_some() {
        save_preferences(&prefs)?;
        output::success(format!("Preferences cleared for {normalized}"));
    } else {
        output::info(format!("No preferences set for {normalized}"));
    }
    Ok(())
}

// ── Table row ────────────────────────────────────────────────────────

#[derive(Tabled, Serialize)]
struct PrefRow {
    #[tabled(rename = "Language")]
    language: String,
    #[tabled(rename = "Preferred LSP")]
    lsp: String,
    #[tabled(rename = "Preferred Grammar")]
    grammar: String,
}

impl ColumnHints for PrefRow {
    fn content_columns() -> &'static [usize] {
        &[2] // Grammar repo can be long
    }
}

// ── Resolution logic ─────────────────────────────────────────────────

/// Resolve the preferred LSP server for a language.
///
/// Resolution: user preference → registry default (lowest priority number).
pub fn resolve_lsp(language: &str) -> Option<String> {
    let normalized = language.to_lowercase();

    // Check user preference
    if let Ok(prefs) = load_preferences() {
        if let Some(pref) = prefs.languages.get(&normalized) {
            if let Some(ref lsp) = pref.lsp {
                return Some(lsp.clone());
            }
        }
    }

    // Fall back to registry default
    if let Some(def) = find_language(&normalized) {
        return def.preferred_lsp().map(|s| s.name.clone());
    }

    None
}

/// Resolve the preferred grammar repo for a language.
///
/// Resolution: user preference → registry default (first source).
pub fn resolve_grammar(language: &str) -> Option<String> {
    let normalized = language.to_lowercase();

    // Check user preference
    if let Ok(prefs) = load_preferences() {
        if let Some(pref) = prefs.languages.get(&normalized) {
            if let Some(ref g) = pref.grammar {
                return Some(g.clone());
            }
        }
    }

    // Fall back to registry default
    if let Some(def) = find_language(&normalized) {
        return def.preferred_grammar().map(|s| s.repo.clone());
    }

    None
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_preferences_empty() {
        let prefs = LanguagePreferences::default();
        assert!(prefs.languages.is_empty());
    }

    #[test]
    fn roundtrip_yaml() {
        let mut prefs = LanguagePreferences::default();
        prefs.languages.insert(
            "rust".into(),
            LanguagePreference {
                lsp: Some("rust-analyzer".into()),
                grammar: None,
            },
        );
        let yaml = serde_yaml_ng::to_string(&prefs).unwrap();
        let parsed: LanguagePreferences = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.languages.len(), 1);
        assert_eq!(
            parsed.languages["rust"].lsp.as_deref(),
            Some("rust-analyzer")
        );
        assert!(parsed.languages["rust"].grammar.is_none());
    }

    #[test]
    fn resolve_lsp_falls_back_to_registry() {
        // With no user prefs set, should return registry default
        let result = resolve_lsp("rust");
        // rust-analyzer is the default for Rust in the registry
        assert!(result.is_some(), "Rust should have a default LSP");
    }

    #[test]
    fn resolve_grammar_falls_back_to_registry() {
        let result = resolve_grammar("rust");
        assert!(result.is_some(), "Rust should have a default grammar");
        let repo = result.unwrap();
        assert!(
            repo.contains("tree-sitter-rust"),
            "Rust grammar should be tree-sitter-rust, got: {repo}"
        );
    }
}
