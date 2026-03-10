//! `language projects` subcommand — per-project language support gaps.
//!
//! Scans each registered project's directory for file extensions, detects
//! languages, and cross-references with grammar/LSP availability. Works
//! without the daemon by scanning the filesystem directly.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use colored::Colorize;
use serde::Serialize;
use tabled::Tabled;
use walkdir::WalkDir;

use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::patterns::exclusion::should_exclude_directory;
use workspace_qdrant_core::tree_sitter::{detect_language, GrammarManager, GrammarStatus};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::ListProjectsRequest;
use crate::output::{self, ColumnHints};

use super::helpers::{load_definitions, which_cmd};

/// Row for the per-project language gap table.
#[derive(Tabled, Serialize)]
struct ProjectLangRow {
    #[tabled(rename = "Project")]
    project: String,
    #[tabled(rename = "Language")]
    language: String,
    #[tabled(rename = "Files")]
    file_count: String,
    #[tabled(rename = "Grammar")]
    grammar: String,
    #[tabled(rename = "LSP")]
    lsp: String,
}

impl ColumnHints for ProjectLangRow {
    fn content_columns() -> &'static [usize] {
        &[0] // Project path is variable-length
    }
}

/// Show per-project language support gaps.
pub async fn language_projects(gaps_only: bool) -> Result<()> {
    output::section("Project Language Support");

    // Get projects from daemon, fall back to warning
    let projects = match fetch_projects().await {
        Some(p) if !p.is_empty() => p,
        Some(_) => {
            output::info("No registered projects found.");
            return Ok(());
        }
        None => {
            output::warning("Daemon not running — cannot list registered projects.");
            output::info("Start the daemon or use 'wqm language health' for system-wide status.");
            return Ok(());
        }
    };

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);
    let defs = load_definitions();

    let mut rows: Vec<ProjectLangRow> = Vec::new();
    let mut total_gaps = 0usize;

    for (name, root) in &projects {
        let languages = scan_project_languages(root);
        if languages.is_empty() {
            continue;
        }

        let display_name = abbreviate_project(name, root);

        for (lang_id, count) in &languages {
            let grammar_status = manager.grammar_status(lang_id);
            let grammar_ok = matches!(
                grammar_status,
                GrammarStatus::Loaded | GrammarStatus::Cached
            );

            let lsp_ok = defs
                .iter()
                .find(|d| d.id() == *lang_id)
                .map(|d| d.lsp_servers.iter().any(|s| which_cmd(&s.binary).is_some()))
                .unwrap_or(false);

            let has_gap = !grammar_ok || !lsp_ok;
            if gaps_only && !has_gap {
                continue;
            }
            if has_gap {
                total_gaps += 1;
            }

            let grammar_str = if grammar_ok {
                format!("{} ok", "~".green())
            } else {
                format!("{} missing", "x".red())
            };

            let lsp_str = if lsp_ok {
                format!("{} ok", "~".green())
            } else {
                let has_config = defs.iter().any(|d| d.id() == *lang_id && d.has_lsp());
                if has_config {
                    format!("{} missing", "x".red())
                } else {
                    format!("{} n/a", "-".dimmed())
                }
            };

            rows.push(ProjectLangRow {
                project: display_name.clone(),
                language: lang_id.clone(),
                file_count: count.to_string(),
                grammar: grammar_str,
                lsp: lsp_str,
            });
        }
    }

    if rows.is_empty() {
        if gaps_only {
            output::success("No language support gaps found across projects.");
        } else {
            output::info("No languages detected in any registered project.");
        }
    } else {
        output::print_table_auto(&rows);
        if total_gaps > 0 {
            println!();
            output::warning(format!(
                "{} language support gap(s) found. Use 'wqm language ts-install <lang>' or 'wqm language lsp-install <lang>' to fix.",
                total_gaps
            ));
        }
    }

    Ok(())
}

/// Fetch project list from the daemon via gRPC.
/// Returns None if daemon is unreachable.
async fn fetch_projects() -> Option<Vec<(String, String)>> {
    let mut client = DaemonClient::connect_default().await.ok()?;
    let request = ListProjectsRequest {
        priority_filter: None,
        active_only: false,
    };
    let response = client.project().list_projects(request).await.ok()?;
    let list = response.into_inner();

    Some(
        list.projects
            .into_iter()
            .map(|p| (p.project_name, p.project_root))
            .collect(),
    )
}

/// Scan a project directory for file extensions and count files per language.
fn scan_project_languages(project_root: &str) -> BTreeMap<String, usize> {
    let path = Path::new(project_root);
    if !path.is_dir() {
        return BTreeMap::new();
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();

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
        if let Some(lang) = detect_language(entry.path()) {
            *counts.entry(lang.to_string()).or_insert(0) += 1;
        }
    }

    counts
}

/// Abbreviate project display: use name if available, otherwise last path component.
fn abbreviate_project(name: &str, root: &str) -> String {
    if !name.is_empty() {
        return name.to_string();
    }
    Path::new(root)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| root.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abbreviate_uses_name_when_present() {
        assert_eq!(abbreviate_project("myapp", "/home/user/myapp"), "myapp");
    }

    #[test]
    fn abbreviate_falls_back_to_last_component() {
        assert_eq!(abbreviate_project("", "/home/user/myapp"), "myapp");
    }

    #[test]
    fn scan_nonexistent_dir_returns_empty() {
        let result = scan_project_languages("/nonexistent/path/12345");
        assert!(result.is_empty());
    }

    #[test]
    fn scan_detects_languages_with_counts() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("testproject");
        std::fs::create_dir_all(&project).unwrap();

        std::fs::write(project.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(project.join("lib.rs"), "pub fn x() {}").unwrap();
        std::fs::write(project.join("app.py"), "print('hi')").unwrap();

        let result = scan_project_languages(project.to_str().unwrap());
        assert_eq!(result.get("rust"), Some(&2));
        assert_eq!(result.get("python"), Some(&1));
    }
}
