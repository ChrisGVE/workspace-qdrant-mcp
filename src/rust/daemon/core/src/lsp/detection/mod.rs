//! LSP Server Detection Module
//!
//! This module handles automatic detection of LSP servers available on the system
//! through PATH scanning and capability discovery, as well as project language
//! detection via marker files and file extension scanning.

pub mod editor_paths;
mod language_detection;
mod path_scanner;
mod registry;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::lsp::Language;

pub use language_detection::{LanguageMarker, ProjectLanguageDetector, ProjectLanguageResult};
pub use path_scanner::LspServerDetector;

/// Information about a detected LSP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedServer {
    /// Server executable name
    pub name: String,
    /// Full path to the executable
    pub path: PathBuf,
    /// Languages supported by this server
    pub languages: Vec<Language>,
    /// Server version if detectable
    pub version: Option<String>,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Priority ranking for this server type
    pub priority: u8,
}

/// LSP server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Supports text document synchronization
    pub text_document_sync: bool,
    /// Supports completion
    pub completion: bool,
    /// Supports hover information
    pub hover: bool,
    /// Supports signature help
    pub signature_help: bool,
    /// Supports go to definition
    pub definition: bool,
    /// Supports find references
    pub references: bool,
    /// Supports document highlighting
    pub document_highlight: bool,
    /// Supports document symbols
    pub document_symbol: bool,
    /// Supports workspace symbols
    pub workspace_symbol: bool,
    /// Supports code actions
    pub code_action: bool,
    /// Supports code lens
    pub code_lens: bool,
    /// Supports document formatting
    pub document_formatting: bool,
    /// Supports document range formatting
    pub document_range_formatting: bool,
    /// Supports document on type formatting
    pub document_on_type_formatting: bool,
    /// Supports renaming
    pub rename: bool,
    /// Supports folding ranges
    pub folding_range: bool,
    /// Supports selection ranges
    pub selection_range: bool,
    /// Supports semantic tokens
    pub semantic_tokens: bool,
    /// Supports diagnostics
    pub diagnostics: bool,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            text_document_sync: true,
            completion: true,
            hover: true,
            signature_help: false,
            definition: true,
            references: true,
            document_highlight: false,
            document_symbol: true,
            workspace_symbol: true,
            code_action: false,
            code_lens: false,
            document_formatting: false,
            document_range_formatting: false,
            document_on_type_formatting: false,
            rename: false,
            folding_range: false,
            selection_range: false,
            semantic_tokens: false,
            diagnostics: true,
        }
    }
}

/// Template for known LSP server configurations
#[derive(Debug, Clone)]
pub(crate) struct ServerTemplate {
    /// Executable name to look for
    pub(crate) executable: &'static str,
    /// Languages this server supports
    pub(crate) languages: &'static [Language],
    /// Default capabilities
    pub(crate) capabilities: ServerCapabilities,
    /// Priority (lower = higher priority)
    pub(crate) priority: u8,
    /// Version detection command arguments
    pub(crate) version_args: &'static [&'static str],
}

#[cfg(test)]
mod tests {
    use super::*;
    use path_scanner::extract_version_from_text;

    #[test]
    fn test_server_detector_creation() {
        let detector = LspServerDetector::new();
        assert!(!detector.known_servers.is_empty());
    }

    #[test]
    fn test_get_servers_for_language() {
        let detector = LspServerDetector::new();
        let python_servers = detector.get_servers_for_language(&Language::Python);
        assert!(!python_servers.is_empty());

        // Should be sorted by priority
        let priorities: Vec<u8> = python_servers
            .iter()
            .map(|name| detector.known_servers.get(name).unwrap().priority)
            .collect();

        for i in 1..priorities.len() {
            assert!(priorities[i - 1] <= priorities[i]);
        }
    }

    #[test]
    fn test_extract_version_number() {
        assert_eq!(extract_version_from_text("rust-analyzer 0.3.1"), "0.3.1");
        assert_eq!(extract_version_from_text("version 2.1.0"), "2.1.0");
        assert_eq!(extract_version_from_text("v1.2.3"), "1.2.3");
        assert_eq!(extract_version_from_text("Some program 1.0"), "1.0");
    }

    #[test]
    fn test_is_known_server() {
        let detector = LspServerDetector::new();
        assert!(detector.is_known_server("rust-analyzer"));
        assert!(detector.is_known_server("ruff-lsp"));
        assert!(!detector.is_known_server("unknown-server"));
    }

    #[tokio::test]
    async fn test_detect_servers() {
        let detector = LspServerDetector::new();
        // This test will depend on what's installed on the system
        let result = detector.detect_servers().await;
        assert!(result.is_ok());
    }

    // ProjectLanguageDetector tests

    #[test]
    fn test_project_language_detector_creation() {
        let detector = ProjectLanguageDetector::new();
        assert!(!detector.markers.is_empty());
    }

    #[test]
    fn test_language_markers_include_common_files() {
        let detector = ProjectLanguageDetector::new();
        let marker_filenames: Vec<&str> = detector.markers.iter().map(|m| m.filename).collect();

        // Verify common marker files are present
        assert!(marker_filenames.contains(&"Cargo.toml"));
        assert!(marker_filenames.contains(&"pyproject.toml"));
        assert!(marker_filenames.contains(&"package.json"));
        assert!(marker_filenames.contains(&"go.mod"));
        assert!(marker_filenames.contains(&"pom.xml"));
    }

    #[tokio::test]
    async fn test_project_language_detector_caching() {
        let detector = ProjectLanguageDetector::new();

        // Initially cache should be empty
        assert!(detector.get_cached("test-project").await.is_none());

        // After detection, result should be cached
        let temp_dir = std::env::temp_dir().join("test_project_lang_detect");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Create a Cargo.toml to detect Rust
        let cargo_path = temp_dir.join("Cargo.toml");
        std::fs::write(&cargo_path, "[package]\nname = \"test\"").unwrap();

        // Detect languages
        let result = detector.detect("test-project", &temp_dir).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.from_cache);

        // Second detection should be from cache
        let cached = detector.detect("test-project", &temp_dir).await;
        assert!(cached.is_ok());
        let cached = cached.unwrap();
        assert!(cached.from_cache);

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_detector_cache_invalidation() {
        let detector = ProjectLanguageDetector::new();

        let temp_dir = std::env::temp_dir().join("test_cache_invalidation");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Detect once to populate cache
        let _ = detector.detect("test-project-inv", &temp_dir).await;
        assert!(detector.get_cached("test-project-inv").await.is_some());

        // Invalidate cache
        detector.invalidate_cache("test-project-inv").await;
        assert!(detector.get_cached("test-project-inv").await.is_none());

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_detector_rust_project() {
        let detector = ProjectLanguageDetector::new();

        let temp_dir = std::env::temp_dir().join("test_rust_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Create a Cargo.toml
        let cargo_path = temp_dir.join("Cargo.toml");
        std::fs::write(
            &cargo_path,
            "[package]\nname = \"test\"\nversion = \"0.1.0\"",
        )
        .unwrap();

        // Create a src/main.rs
        let src_dir = temp_dir.join("src");
        let _ = std::fs::create_dir_all(&src_dir);
        std::fs::write(src_dir.join("main.rs"), "fn main() {}").unwrap();

        // Detect languages
        let result = detector.detect("rust-test", &temp_dir).await.unwrap();

        // Should detect Rust from marker
        assert!(result
            .marker_languages
            .iter()
            .any(|l| matches!(l, Language::Rust)));

        // All languages should include Rust
        assert!(result
            .all_languages
            .iter()
            .any(|l| matches!(l, Language::Rust)));

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_detector_python_project() {
        let detector = ProjectLanguageDetector::new();

        let temp_dir = std::env::temp_dir().join("test_python_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Create a pyproject.toml
        let pyproject_path = temp_dir.join("pyproject.toml");
        std::fs::write(
            &pyproject_path,
            "[project]\nname = \"test\"\nversion = \"0.1.0\"",
        )
        .unwrap();

        // Create a main.py
        std::fs::write(temp_dir.join("main.py"), "def main(): pass").unwrap();

        // Detect languages
        let result = detector.detect("python-test", &temp_dir).await.unwrap();

        // Should detect Python from marker
        assert!(result
            .marker_languages
            .iter()
            .any(|l| matches!(l, Language::Python)));

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_result_serialization() {
        let result = ProjectLanguageResult {
            marker_languages: vec![Language::Rust],
            extension_languages: vec![Language::Python, Language::JavaScript],
            all_languages: vec![Language::Rust, Language::Python, Language::JavaScript],
            from_cache: false,
        };

        // Test serialization
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Rust"));

        // Test deserialization
        let deserialized: ProjectLanguageResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.marker_languages.len(), 1);
        assert_eq!(deserialized.all_languages.len(), 3);
    }
}
