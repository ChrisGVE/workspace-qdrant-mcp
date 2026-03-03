//! Integration tests for dynamic grammar loading — cache, auto-download,
//! version compatibility, graceful degradation, concurrent access, and reload.

use std::sync::Arc;
use tempfile::TempDir;
use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::{
    check_grammar_compatibility, extract_chunks, GrammarManager, GrammarStatus,
    StaticLanguageProvider, TreeSitterParser,
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
        check_interval_hours: 168, // Weekly
        ..Default::default()
    }
}

// =============================================================================
// Cache Management Tests
// =============================================================================

#[test]
fn test_grammar_cache_hit_with_static_grammar() {
    use workspace_qdrant_core::tree_sitter::get_static_language;
    if get_static_language("rust").is_none() {
        return;
    }
    let source = r#"fn main() { println!("Hello"); }"#;
    let path = std::path::Path::new("test.rs");

    let chunks = extract_chunks(source, path, 1024).expect("Should parse Rust code");
    assert!(!chunks.is_empty(), "Should extract at least one chunk");
}

#[test]
fn test_grammar_cache_miss_with_unsupported_language() {
    let source = "some random text content for testing";
    let path = std::path::Path::new("test.xyz");

    let chunks = extract_chunks(source, path, 1024).expect("Should fall back to text chunking");
    assert!(!chunks.is_empty(), "Should have text chunks");
}

#[test]
fn test_grammar_status_transitions() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    assert_eq!(
        manager.grammar_status("nonexistent"),
        GrammarStatus::NotAvailable
    );

    let status = manager.grammar_status("rust");
    assert!(matches!(
        status,
        GrammarStatus::NotAvailable | GrammarStatus::NeedsDownload
    ));
}

// =============================================================================
// Auto-download Tests
// =============================================================================

#[test]
fn test_auto_download_disabled() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);
    assert!(!manager.has_downloader());
}

#[test]
fn test_auto_download_enabled() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);
    assert!(manager.has_downloader());
    assert_eq!(
        manager.grammar_status("ruby"),
        GrammarStatus::NeedsDownload
    );
}

// =============================================================================
// Version Compatibility Tests
// =============================================================================

#[test]
fn test_static_grammar_compatibility() {
    use workspace_qdrant_core::tree_sitter::get_static_language;

    if let Some(rust_lang) = get_static_language("rust") {
        let compat = check_grammar_compatibility(&rust_lang);
        assert!(compat.is_compatible(), "Static Rust grammar should be compatible");
    }

    if let Some(python_lang) = get_static_language("python") {
        let compat = check_grammar_compatibility(&python_lang);
        assert!(compat.is_compatible(), "Static Python grammar should be compatible");
    }
}

#[test]
fn test_grammar_validation_result() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    let result = manager.validate_grammars();
    let summary = result.summary();
    assert!(summary.contains("available") || summary.contains("need download"));
}

// =============================================================================
// Graceful Degradation Tests
// =============================================================================

#[test]
fn test_missing_grammar_fallback_to_text_chunking() {
    let source = "content in an unsupported language";
    let path = std::path::Path::new("test.unsupported");

    let chunks = extract_chunks(source, path, 100).expect("Should fall back to text chunking");
    assert!(!chunks.is_empty());

    for chunk in &chunks {
        assert!(
            chunk.chunk_type == workspace_qdrant_core::tree_sitter::ChunkType::Text,
            "Unsupported language should use text chunking"
        );
    }
}

#[test]
fn test_parser_with_fallback_provider() {
    let static_provider = StaticLanguageProvider::new();

    if static_provider.supports_language("rust") {
        let parser = TreeSitterParser::with_provider("rust", &static_provider)
            .expect("Should create parser");
        assert_eq!(parser.language_name(), "rust");
    }
}

#[test]
fn test_parser_with_custom_provider() {
    use workspace_qdrant_core::tree_sitter::{LoadedGrammarsProvider, LanguageProvider};

    let mut provider = LoadedGrammarsProvider::new();
    if let Some(lang) = workspace_qdrant_core::tree_sitter::get_static_language("python") {
        provider.add_grammar("python", lang);
    }

    if provider.supports_language("python") {
        let parser = TreeSitterParser::with_provider("python", &provider)
            .expect("Should create parser with custom provider");
        assert_eq!(parser.language_name(), "python");
    }
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

#[tokio::test]
async fn test_concurrent_grammar_loading() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::task;
    use workspace_qdrant_core::tree_sitter::get_static_language;

    if get_static_language("rust").is_none() {
        return;
    }

    let success_count = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    for i in 0..10 {
        let success_count = Arc::clone(&success_count);
        let handle = task::spawn(async move {
            let source = format!(r#"fn task_{}() {{ }}"#, i);
            let path = std::path::Path::new("test.rs");
            if extract_chunks(&source, path, 1024).is_ok() {
                success_count.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.expect("Task should complete");
    }
    assert_eq!(success_count.load(Ordering::SeqCst), 10);
}

#[tokio::test]
async fn test_concurrent_grammar_manager_access() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::task;

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path().to_path_buf();
    let success_count = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    for _ in 0..5 {
        let path = temp_path.clone();
        let success_count = Arc::clone(&success_count);
        let handle = task::spawn(async move {
            let config = GrammarConfig {
                cache_dir: path,
                required: vec!["rust".to_string()],
                auto_download: false,
                tree_sitter_version: "0.24".to_string(),
                download_base_url: "https://example.com".to_string(),
                verify_checksums: false,
                lazy_loading: true,
                check_interval_hours: 168,
                ..Default::default()
            };
            let manager = GrammarManager::new(config);
            let _ = manager.grammar_status("rust");
            success_count.fetch_add(1, Ordering::SeqCst);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.expect("Task should complete");
    }
    assert_eq!(success_count.load(Ordering::SeqCst), 5);
}

// =============================================================================
// Reload Tests
// =============================================================================

#[tokio::test]
async fn test_grammar_reload_empty() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let mut manager = GrammarManager::new(config);
    let results = manager.reload_all().await;
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_unload_and_reload() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let mut manager = GrammarManager::new(config);

    assert!(manager.loaded_languages().is_empty());
    assert!(!manager.unload_grammar("rust"));
    manager.unload_all();
    assert!(manager.loaded_languages().is_empty());
}
