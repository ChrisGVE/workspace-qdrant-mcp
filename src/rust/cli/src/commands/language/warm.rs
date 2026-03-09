//! `language warm` subcommand — pre-warm tree-sitter grammars for a project.
//!
//! Scans a project directory for file extensions, determines which languages
//! are present, and downloads all needed grammars. Eliminates cold-start
//! grammar download stalls in the daemon.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;
use colored::Colorize;
use walkdir::WalkDir;

use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::patterns::exclusion::{should_exclude_directory, should_exclude_file};
use workspace_qdrant_core::tree_sitter::{detect_language, GrammarManager};

use crate::output;

/// Pre-warm tree-sitter grammars for a project or explicit language list.
pub async fn warm(
    project_path: Option<&str>,
    languages: Option<&str>,
    force: bool,
) -> Result<()> {
    output::section("Grammar Pre-warming");

    let target_languages = if let Some(lang_list) = languages {
        parse_language_list(lang_list)
    } else if let Some(path) = project_path {
        scan_project_languages(path)
    } else {
        // Default: current directory
        scan_project_languages(".")
    };

    if target_languages.is_empty() {
        output::info("No languages detected. Nothing to warm.");
        return Ok(());
    }

    output::info(format!(
        "Languages to warm: {}",
        target_languages
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    ));
    output::separator();

    let mut config = GrammarConfig::default();
    config.auto_download = true;
    let mut manager = GrammarManager::new(config);

    let total = target_languages.len();
    let mut downloaded = 0u32;
    let mut already_cached = 0u32;
    let mut failed = 0u32;

    for language in &target_languages {
        if !force && manager.cache_paths().grammar_exists(language) {
            output::success(format!("  {}: cached", language));
            already_cached += 1;
            continue;
        }

        output::info(format!("  {}: downloading...", language));
        match manager.get_grammar(language).await {
            Ok(_) => {
                output::success(format!("  {}: installed", language));
                downloaded += 1;
            }
            Err(e) => {
                output::warning(format!("  {}: failed ({})", language, e));
                failed += 1;
            }
        }
    }

    output::separator();
    output::kv("Total", total);
    output::kv("Downloaded", format!("{}", downloaded).green());
    if already_cached > 0 {
        output::kv("Already cached", format!("{}", already_cached).blue());
    }
    if failed > 0 {
        output::kv("Failed", format!("{}", failed).red());
    }

    Ok(())
}

/// Parse a comma-separated language list.
fn parse_language_list(input: &str) -> BTreeSet<String> {
    input
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Scan a project directory for file extensions and determine languages.
fn scan_project_languages(project_path: &str) -> BTreeSet<String> {
    let path = Path::new(project_path);
    if !path.is_dir() {
        output::warning(format!("Not a directory: {}", project_path));
        return BTreeSet::new();
    }

    output::info(format!("Scanning {} for languages...", project_path));

    let mut languages = BTreeSet::new();

    let walker = WalkDir::new(path)
        .follow_links(false)
        .max_depth(10)
        .into_iter()
        .filter_entry(|entry| {
            if entry.file_type().is_dir() {
                let dir_name = entry.file_name().to_string_lossy();
                !should_exclude_directory(&dir_name)
            } else {
                true
            }
        });

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }

        let file_path = entry.path();

        // Check file exclusion
        if let Some(path_str) = file_path.to_str() {
            if should_exclude_file(path_str) {
                continue;
            }
        }

        if let Some(lang) = detect_language(file_path) {
            languages.insert(lang.to_string());
        }
    }

    output::info(format!("Detected {} language(s)", languages.len()));
    languages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_language_list_basic() {
        let result = parse_language_list("rust,python,typescript");
        assert_eq!(result.len(), 3);
        assert!(result.contains("rust"));
        assert!(result.contains("python"));
        assert!(result.contains("typescript"));
    }

    #[test]
    fn parse_language_list_with_spaces() {
        let result = parse_language_list("rust , python , go");
        assert_eq!(result.len(), 3);
        assert!(result.contains("rust"));
        assert!(result.contains("python"));
        assert!(result.contains("go"));
    }

    #[test]
    fn parse_language_list_deduplicates() {
        let result = parse_language_list("rust,Rust,RUST");
        assert_eq!(result.len(), 1);
        assert!(result.contains("rust"));
    }

    #[test]
    fn parse_language_list_empty() {
        let result = parse_language_list("");
        assert!(result.is_empty());
    }

    #[test]
    fn scan_nonexistent_dir_returns_empty() {
        let result = scan_project_languages("/nonexistent/path/12345");
        assert!(result.is_empty());
    }
}
