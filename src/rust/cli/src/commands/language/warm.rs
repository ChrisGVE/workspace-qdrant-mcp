//! `language warm` subcommand — pre-warm tree-sitter grammars for a project.
//!
//! Scans a project directory for file extensions, determines which languages
//! are present, and downloads all needed grammars. Eliminates cold-start
//! grammar download stalls in the daemon.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::Result;
use colored::Colorize;
use walkdir::WalkDir;

use wqm_common::exclusion::{should_exclude_directory, should_exclude_file};

use crate::output;

use super::helpers::{build_extension_map, load_definitions};

/// Pre-warm tree-sitter grammars for a project or explicit language list.
pub async fn warm(project_path: Option<&str>, languages: Option<&str>, force: bool) -> Result<()> {
    output::section("Grammar Pre-warming");

    let ext_map = build_extension_map(&load_definitions());

    let target_languages = if let Some(lang_list) = languages {
        parse_language_list(lang_list)
    } else if let Some(path) = project_path {
        scan_project_languages(path, &ext_map)
    } else {
        // Default: current directory
        scan_project_languages(".", &ext_map)
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

    // Warming installs grammars into the daemon's cache → daemon required.
    let mut client = crate::grpc::ensure_daemon_available().await?;

    let total = target_languages.len();
    let mut downloaded = 0u32;
    let mut already_cached = 0u32;
    let mut failed = 0u32;

    for language in &target_languages {
        if !force {
            if let Ok(q) = client.query_language(language.clone()).await {
                if q.found && matches!(q.grammar_status.as_str(), "cached" | "loaded") {
                    output::success(format!("  {}: cached", language));
                    already_cached += 1;
                    continue;
                }
            }
        }

        output::info(format!("  {}: downloading...", language));
        match client.install_grammar(language.clone(), force).await {
            Ok(_) => {
                output::success(format!("  {}: installed", language));
                downloaded += 1;
            }
            Err(e) => {
                output::warning(format!("  {}: failed ({})", language, e.message()));
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

/// Scan a project directory for file extensions and determine languages via the
/// registry-derived `ext_map`.
fn scan_project_languages(
    project_path: &str,
    ext_map: &HashMap<String, String>,
) -> BTreeSet<String> {
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

        let ext = file_path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase());
        if let Some(lang) = ext.and_then(|e| ext_map.get(&e)) {
            languages.insert(lang.clone());
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
        let ext_map = build_extension_map(&load_definitions());
        let result = scan_project_languages("/nonexistent/path/12345", &ext_map);
        assert!(result.is_empty());
    }
}
