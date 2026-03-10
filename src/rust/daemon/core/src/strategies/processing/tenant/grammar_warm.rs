//! Background grammar pre-warming for newly registered projects.
//!
//! Scans a project directory for file extensions, determines which languages
//! are present, and downloads needed tree-sitter grammars asynchronously.
//! Called after project registration (Tenant/Add) to eliminate cold-start
//! grammar download stalls during file ingestion.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, info};
use walkdir::WalkDir;

use crate::patterns::exclusion::should_exclude_directory;
use crate::tree_sitter::{detect_language, GrammarManager};

/// Spawn a background task that scans a project for file extensions and
/// downloads needed tree-sitter grammars.
///
/// Non-blocking: returns immediately. Grammar downloads happen asynchronously.
/// When individual files encounter missing grammars during ingestion, they
/// will find them already cached (or in-flight) thanks to this pre-warming.
pub(crate) fn spawn_grammar_warming(
    grammar_mgr: Arc<RwLock<GrammarManager>>,
    project_root: String,
) {
    tokio::spawn(async move {
        let languages = scan_project_for_languages(&project_root);
        if languages.is_empty() {
            return;
        }

        info!(
            "Grammar pre-warming: {} language(s) detected in {}",
            languages.len(),
            project_root
        );

        let mut downloaded = 0u32;
        let mut already_cached = 0u32;

        for language in &languages {
            // Check if already loaded or cached (read lock only)
            {
                let mgr = grammar_mgr.read().await;
                if mgr.is_loaded(language) || mgr.cache_paths().grammar_exists(language) {
                    already_cached += 1;
                    continue;
                }
            }

            // Download (write lock)
            let mut mgr = grammar_mgr.write().await;
            match mgr.get_grammar(language).await {
                Ok(_) => {
                    downloaded += 1;
                    debug!(language = %language, "Grammar pre-warmed");
                }
                Err(e) => {
                    debug!(language = %language, error = %e, "Grammar pre-warming failed");
                }
            }
        }

        if downloaded > 0 || already_cached > 0 {
            info!(
                "Grammar pre-warming complete for {}: {} downloaded, {} already cached",
                project_root, downloaded, already_cached
            );
        }
    });
}

/// Scan a project directory for file extensions and return detected language IDs.
///
/// Respects directory exclusion patterns. Limits depth to 10 levels.
fn scan_project_for_languages(project_root: &str) -> BTreeSet<String> {
    let path = Path::new(project_root);
    if !path.is_dir() {
        return BTreeSet::new();
    }

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
        if let Some(lang) = detect_language(entry.path()) {
            languages.insert(lang.to_string());
        }
    }

    languages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_nonexistent_dir_returns_empty() {
        let result = scan_project_for_languages("/nonexistent/path/12345");
        assert!(result.is_empty());
    }

    #[test]
    fn scan_empty_tempdir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let result = scan_project_for_languages(dir.path().to_str().unwrap());
        assert!(result.is_empty());
    }

    /// Create a project-like directory inside a tempdir with a non-hidden name.
    /// Tempfile creates dirs starting with `.tmp` which are excluded as hidden.
    fn make_project_dir() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("myproject");
        std::fs::create_dir_all(&project).unwrap();
        (dir, project)
    }

    #[test]
    fn scan_detects_languages_from_extensions() {
        let (_dir, project) = make_project_dir();
        std::fs::write(project.join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(project.join("app.py"), "print('hello')").unwrap();
        std::fs::write(project.join("readme.txt"), "hello").unwrap();
        let result = scan_project_for_languages(project.to_str().unwrap());
        assert!(result.contains("rust"), "expected rust in {:?}", result);
        assert!(result.contains("python"), "expected python in {:?}", result);
        assert!(!result.contains("txt"));
    }

    #[test]
    fn scan_skips_hidden_directories() {
        let (_dir, project) = make_project_dir();
        std::fs::create_dir_all(project.join(".hidden")).unwrap();
        std::fs::write(project.join(".hidden/secret.js"), "var x = 1;").unwrap();
        std::fs::write(project.join("app.py"), "print('hello')").unwrap();
        let result = scan_project_for_languages(project.to_str().unwrap());
        assert!(result.contains("python"));
        assert!(!result.contains("javascript"));
    }
}
