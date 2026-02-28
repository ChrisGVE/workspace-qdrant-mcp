//! Tests for WatchType, collection constants, and library watch configuration.

use super::super::*;
use tempfile::tempdir;

#[test]
fn test_get_current_branch_non_git() {
    let temp_dir = tempdir().unwrap();
    let branch = get_current_branch(temp_dir.path());
    assert_eq!(branch, "main");
}

// Multi-tenant routing tests
#[test]
fn test_watch_type_default() {
    assert_eq!(WatchType::default(), WatchType::Project);
}

#[test]
fn test_watch_type_from_str() {
    assert_eq!(WatchType::from_str("project"), Some(WatchType::Project));
    assert_eq!(WatchType::from_str("library"), Some(WatchType::Library));
    assert_eq!(WatchType::from_str("PROJECT"), Some(WatchType::Project));
    assert_eq!(WatchType::from_str("LIBRARY"), Some(WatchType::Library));
    assert_eq!(WatchType::from_str("invalid"), None);
}

#[test]
fn test_watch_type_as_str() {
    assert_eq!(WatchType::Project.as_str(), "project");
    assert_eq!(WatchType::Library.as_str(), "library");
}

#[test]
fn test_unified_collection_constants() {
    use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS};
    // Canonical collection names (without underscore prefix)
    assert_eq!(COLLECTION_PROJECTS, "projects");
    assert_eq!(COLLECTION_LIBRARIES, "libraries");
}

// Library watch ID format tests
#[test]
fn test_library_watch_id_format() {
    let library_name = "langchain";
    let id = format!("lib_{}", library_name);

    assert!(id.starts_with("lib_"));
    assert_eq!(id, "lib_langchain");

    // Test stripping prefix
    let extracted = id.strip_prefix("lib_").unwrap_or(&id);
    assert_eq!(extracted, "langchain");
}

#[test]
fn test_library_watch_config_creation() {
    use std::path::PathBuf;
    let library_name = "my_docs";
    let id = format!("lib_{}", library_name);

    let config = WatchConfig {
        id: id.clone(),
        path: PathBuf::from("/path/to/docs"),
        tenant_id: library_name.to_string(),
        collection: format!("_{}", library_name),
        patterns: vec!["*.pdf".to_string(), "*.md".to_string()],
        ignore_patterns: vec![".git/*".to_string()],
        recursive: true,
        debounce_ms: 2000,
        enabled: true,
        watch_type: WatchType::Library,
        library_name: Some(library_name.to_string()),
    };

    assert_eq!(config.watch_type, WatchType::Library);
    assert_eq!(config.library_name, Some("my_docs".to_string()));
    assert_eq!(config.collection, "_my_docs");
}
