//! Integration tests for grammar lifecycle — cache clearing, edge cases,
//! configuration, status checks, and periodic version checks.

use std::sync::Arc;
use tempfile::TempDir;
use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::{
    extract_chunks, extract_chunks_with_provider, known_grammar_languages, GrammarManager,
    GrammarMetadata, GrammarStatus, LanguageProvider, StaticLanguageProvider, TreeSitterParser,
};

/// Create a test grammar configuration with temporary cache directory.
fn test_config(temp_dir: &TempDir, auto_download: bool) -> GrammarConfig {
    GrammarConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        required: vec!["rust".to_string(), "python".to_string()],
        auto_download,
        tree_sitter_version: "0.24".to_string(),
        download_base_url: "https://example.com/{language}/v{version}/{platform}.{ext}"
            .to_string(),
        verify_checksums: false,
        lazy_loading: true,
        check_interval_hours: 168,
        ..Default::default()
    }
}

// =============================================================================
// Cache Clear Tests
// =============================================================================

#[test]
fn test_clear_cache_nonexistent_language() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    let result = manager.clear_cache("nonexistent");
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_clear_all_cache_empty() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    let result = manager.clear_all_cache();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_source_parsing() {
    use workspace_qdrant_core::tree_sitter::get_static_language;
    if get_static_language("rust").is_none() {
        return;
    }
    let source = "";
    let path = std::path::Path::new("test.rs");
    let result = extract_chunks(source, path, 1024);
    assert!(result.is_ok());
}

#[test]
fn test_large_source_chunking() {
    use workspace_qdrant_core::tree_sitter::get_static_language;
    if get_static_language("rust").is_none() {
        return;
    }
    let mut source = String::new();
    for i in 0..100 {
        source.push_str(&format!("fn function_{}() {{ let x = {}; }}\n", i, i));
    }

    let path = std::path::Path::new("test.rs");
    let result = extract_chunks(&source, path, 512);
    assert!(result.is_ok());
    let chunks = result.unwrap();
    assert!(chunks.len() > 1, "Large file should be split into multiple chunks");
}

#[test]
fn test_incremental_parsing() {
    use workspace_qdrant_core::tree_sitter::get_static_language;
    if get_static_language("rust").is_none() {
        return;
    }
    let mut parser = TreeSitterParser::new("rust").expect("Should create parser");

    let source1 = "fn main() { }";
    let tree1 = parser.parse(source1).expect("Should parse");

    let source2 = "fn main() { let x = 1; }";
    let tree2 = parser
        .parse_incremental(source2, Some(&tree1))
        .expect("Should parse incrementally");

    assert!(!tree1.root_node().has_error());
    assert!(!tree2.root_node().has_error());
}

// =============================================================================
// Configuration and Status Tests
// =============================================================================

#[test]
fn test_grammar_manager_config_accessor() {
    let temp_dir = TempDir::new().unwrap();
    let config = GrammarConfig {
        auto_download: true,
        idle_update_check_enabled: true,
        idle_update_check_delay_secs: 120,
        check_interval_hours: 24,
        ..test_config(&temp_dir, true)
    };
    let manager = GrammarManager::new(config);

    let cfg = manager.config();
    assert!(cfg.auto_download);
    assert!(cfg.idle_update_check_enabled);
    assert_eq!(cfg.idle_update_check_delay_secs, 120);
    assert_eq!(cfg.check_interval_hours, 24);
}

#[test]
fn test_grammar_manager_idle_check_disabled() {
    let temp_dir = TempDir::new().unwrap();
    let config = GrammarConfig {
        idle_update_check_enabled: false,
        ..test_config(&temp_dir, true)
    };
    let manager = GrammarManager::new(config);

    let cfg = manager.config();
    assert!(!cfg.idle_update_check_enabled);
    assert!(cfg.auto_download);
}

#[test]
fn test_auto_download_disabled_provider_empty() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    let provider = manager.create_language_provider();
    assert!(provider.is_empty());
    assert!(provider.available_languages().is_empty());

    let source = "some content here";
    let path = std::path::Path::new("test.xyz");
    let result = extract_chunks_with_provider(source, path, 1024, Some(Arc::new(provider)));
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty(), "Should produce text chunks as fallback");
}

#[test]
fn test_grammar_status_needs_download_for_known_languages() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    let known = known_grammar_languages();
    for lang in known {
        let status = manager.grammar_status(lang);
        assert!(
            matches!(
                status,
                GrammarStatus::NeedsDownload | GrammarStatus::Loaded | GrammarStatus::Cached
            ),
            "Known language '{}' should be NeedsDownload/Loaded/Cached, got: {:?}",
            lang,
            status
        );
    }
}

#[test]
fn test_grammar_status_not_available_when_auto_download_off() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    let status = manager.grammar_status("ruby");
    assert_eq!(status, GrammarStatus::NotAvailable);
}

// =============================================================================
// Periodic Version Check Tests
// =============================================================================

#[tokio::test]
async fn test_periodic_version_check_updates_timestamps() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let mut manager = GrammarManager::new(config);

    let cache_paths = manager.cache_paths().clone();
    for lang in &["rust", "python"] {
        cache_paths.create_directories(lang).unwrap();
        std::fs::write(cache_paths.grammar_path(lang), "fake").unwrap();
        let metadata = GrammarMetadata::new(
            *lang,
            "0.24",
            "0.24.0",
            &cache_paths.platform,
            "checksum",
        );
        cache_paths.save_metadata(lang, &metadata).unwrap();
    }

    let results = manager.periodic_version_check().await;
    assert_eq!(results.len(), 2, "Should check both required languages");
    for (lang, result) in &results {
        assert!(result.is_ok(), "Check for '{}' should succeed", lang);
    }

    for lang in &["rust", "python"] {
        let metadata = cache_paths.load_metadata(lang).unwrap().unwrap();
        assert!(metadata.last_checked_at.is_some(), "last_checked_at should be set for '{}'", lang);
    }
}

#[test]
fn test_check_interval_prevents_excessive_checks() {
    let temp_dir = TempDir::new().unwrap();
    let config = GrammarConfig {
        check_interval_hours: 168,
        ..test_config(&temp_dir, true)
    };
    let manager = GrammarManager::new(config);

    let cache_paths = manager.cache_paths();
    for lang in &["rust", "python"] {
        cache_paths.create_directories(lang).unwrap();
        std::fs::write(cache_paths.grammar_path(lang), "fake").unwrap();
        let mut metadata = GrammarMetadata::new(
            *lang,
            "0.24",
            "0.24.0",
            &cache_paths.platform,
            "checksum",
        );
        metadata.mark_checked();
        cache_paths.save_metadata(lang, &metadata).unwrap();
    }

    assert!(!manager.needs_periodic_check(), "Should not need check right after marking checked");
}

#[test]
fn test_check_interval_zero_disables_checks() {
    let temp_dir = TempDir::new().unwrap();
    let config = GrammarConfig {
        check_interval_hours: 0,
        ..test_config(&temp_dir, true)
    };
    let manager = GrammarManager::new(config);
    assert!(!manager.needs_periodic_check(), "check_interval_hours=0 should disable periodic checks");
}
