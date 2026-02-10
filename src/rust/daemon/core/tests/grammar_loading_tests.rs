//! Integration tests for dynamic grammar loading.
//!
//! These tests cover end-to-end scenarios for the tree-sitter grammar
//! loading system including cache management, version compatibility,
//! and fallback behavior.

use std::sync::Arc;
use tempfile::TempDir;
use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::{
    check_grammar_compatibility, extract_chunks, GrammarManager, GrammarStatus,
    LanguageProvider, LoadedGrammarsProvider, SemanticChunker, StaticLanguageProvider,
    TreeSitterParser,
};

/// Create a test grammar configuration with temporary cache directory.
fn test_config(temp_dir: &TempDir, auto_download: bool) -> GrammarConfig {
    GrammarConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        required: vec!["rust".to_string(), "python".to_string()],
        auto_download,
        tree_sitter_version: "0.24".to_string(),
        download_base_url: "https://example.com/{language}/v{version}/{platform}.{ext}".to_string(),
        verify_checksums: false,
        lazy_loading: true,
        check_interval_hours: 168, // Weekly
    }
}

// =============================================================================
// Cache Management Tests
// =============================================================================

#[test]
fn test_grammar_cache_hit_with_static_grammar() {
    // When static grammars are available, parsing should succeed without dynamic loading
    let source = r#"fn main() { println!("Hello"); }"#;
    let path = std::path::Path::new("test.rs");

    let chunks = extract_chunks(source, path, 1024).expect("Should parse Rust code");
    assert!(!chunks.is_empty(), "Should extract at least one chunk");
}

#[test]
fn test_grammar_cache_miss_with_unsupported_language() {
    // When the language is not supported, should fall back to text chunking
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

    // Without auto_download, uncached grammars should be NotAvailable
    assert_eq!(
        manager.grammar_status("nonexistent"),
        GrammarStatus::NotAvailable
    );

    // Cached status for required languages without cache files
    let status = manager.grammar_status("rust");
    // Either NotAvailable (no cache) or available (if static available)
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

    // Without auto_download, uncached grammars should not be downloadable
    assert!(!manager.has_downloader());
}

#[test]
fn test_auto_download_enabled() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    // With auto_download, downloader should be available
    assert!(manager.has_downloader());

    // Grammar status should indicate NeedsDownload for uncached languages
    assert_eq!(
        manager.grammar_status("ruby"), // Not in static, not cached
        GrammarStatus::NeedsDownload
    );
}

// =============================================================================
// Version Compatibility Tests
// =============================================================================

#[test]
fn test_static_grammar_compatibility() {
    use workspace_qdrant_core::tree_sitter::get_static_language;

    // Static grammars should be compatible with the runtime
    if let Some(rust_lang) = get_static_language("rust") {
        let compat = check_grammar_compatibility(&rust_lang);
        assert!(
            compat.is_compatible(),
            "Static Rust grammar should be compatible"
        );
    }

    if let Some(python_lang) = get_static_language("python") {
        let compat = check_grammar_compatibility(&python_lang);
        assert!(
            compat.is_compatible(),
            "Static Python grammar should be compatible"
        );
    }
}

#[test]
fn test_grammar_validation_result() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, true);
    let manager = GrammarManager::new(config);

    let result = manager.validate_grammars();

    // Summary should contain counts
    let summary = result.summary();
    assert!(summary.contains("available") || summary.contains("need download"));
}

// =============================================================================
// Graceful Degradation Tests
// =============================================================================

#[test]
fn test_missing_grammar_fallback_to_text_chunking() {
    // For languages without grammar support, should fall back to text chunking
    let source = "content in an unsupported language";
    let path = std::path::Path::new("test.unsupported");

    let chunks = extract_chunks(source, path, 100).expect("Should fall back to text chunking");
    assert!(!chunks.is_empty());

    // All chunks should be text type (not semantic)
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

    // Parser with static fallback should work for supported languages
    if static_provider.supports_language("rust") {
        let parser =
            TreeSitterParser::with_provider("rust", &static_provider).expect("Should create parser");
        assert_eq!(parser.language_name(), "rust");
    }
}

#[test]
fn test_parser_with_custom_provider() {
    // Create a custom provider from static languages
    let mut provider = LoadedGrammarsProvider::new();

    if let Some(lang) = workspace_qdrant_core::tree_sitter::get_static_language("python") {
        provider.add_grammar("python", lang);
    }

    // Parser should work with custom provider
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

    let success_count = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();

    // Spawn multiple tasks trying to parse code concurrently
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

    // Wait for all tasks
    for handle in handles {
        handle.await.expect("Task should complete");
    }

    // All should succeed
    assert_eq!(success_count.load(Ordering::SeqCst), 10);
}

#[tokio::test]
async fn test_concurrent_grammar_manager_access() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::task;

    let temp_dir = TempDir::new().unwrap();
    let _config = test_config(&temp_dir, false);

    // Create manager - each task will create its own for thread safety
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
                check_interval_hours: 168, // Weekly
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

    // Reload all when nothing is loaded should return empty
    let results = manager.reload_all().await;
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_unload_and_reload() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let mut manager = GrammarManager::new(config);

    // Initially no grammars loaded
    assert!(manager.loaded_languages().is_empty());

    // Unload non-existent grammar should return false
    assert!(!manager.unload_grammar("rust"));

    // Unload all should work even when empty
    manager.unload_all();
    assert!(manager.loaded_languages().is_empty());
}

// =============================================================================
// All Supported Languages Tests
// =============================================================================

#[test]
fn test_all_supported_languages_parsing() {
    let test_cases = vec![
        ("rust", "fn main() { }", "test.rs"),
        ("python", "def main():\n    pass", "test.py"),
        ("javascript", "function main() { }", "test.js"),
        ("typescript", "function main(): void { }", "test.ts"),
        ("go", "func main() { }", "test.go"),
        ("java", "class Main { void main() { } }", "test.java"),
        ("c", "int main() { return 0; }", "test.c"),
        ("cpp", "int main() { return 0; }", "test.cpp"),
    ];

    for (lang, source, filename) in test_cases {
        let path = std::path::Path::new(filename);
        let result = extract_chunks(source, path, 1024);

        assert!(
            result.is_ok(),
            "Should successfully parse {} code (file: {})",
            lang,
            filename
        );

        let chunks = result.unwrap();
        assert!(
            !chunks.is_empty(),
            "Should extract chunks from {} code",
            lang
        );
    }
}

#[test]
fn test_language_detection() {
    use workspace_qdrant_core::tree_sitter::{detect_language, is_language_supported};

    let test_cases = vec![
        ("test.rs", Some("rust"), true),
        ("test.py", Some("python"), true),
        ("test.js", Some("javascript"), true),
        ("test.ts", Some("typescript"), true),
        ("test.tsx", Some("tsx"), true),
        ("test.jsx", Some("jsx"), true),
        ("test.go", Some("go"), true),
        ("test.java", Some("java"), true),
        ("test.c", Some("c"), true),
        ("test.cpp", Some("cpp"), true),
        ("test.json", Some("json"), false), // Detected but not supported for semantic
        ("test.xyz", None, false),
    ];

    for (filename, expected_lang, expected_supported) in test_cases {
        let path = std::path::Path::new(filename);
        let detected = detect_language(path);
        assert_eq!(
            detected, expected_lang,
            "Language detection for {} should be {:?}",
            filename, expected_lang
        );

        if let Some(lang) = detected {
            assert_eq!(
                is_language_supported(lang),
                expected_supported,
                "Language {} support should be {}",
                lang,
                expected_supported
            );
        }
    }
}

// =============================================================================
// SemanticChunker with Provider Tests
// =============================================================================

#[test]
fn test_chunker_with_static_provider() {
    let provider = Arc::new(StaticLanguageProvider::new());
    let chunker = SemanticChunker::with_provider(1024, provider);

    let source = r#"
fn hello() {
    println!("Hello");
}

fn world() {
    println!("World");
}
"#;
    let path = std::path::Path::new("test.rs");

    let result = chunker.chunk_source(source, path, "rust");
    assert!(result.is_ok());

    let chunks = result.unwrap();
    // Should have function chunks
    assert!(chunks.iter().any(|c| c.symbol_name == "hello"));
    assert!(chunks.iter().any(|c| c.symbol_name == "world"));
}

#[test]
fn test_chunker_language_provider_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    // Create provider snapshot
    let provider = manager.create_language_provider();

    // Provider should be empty since no grammars loaded
    assert!(provider.is_empty());
    assert!(provider.available_languages().is_empty());
}

// =============================================================================
// Cache Clear Tests
// =============================================================================

#[test]
fn test_clear_cache_nonexistent_language() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    // Clearing cache for non-existent language should succeed with false
    let result = manager.clear_cache("nonexistent");
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_clear_all_cache_empty() {
    let temp_dir = TempDir::new().unwrap();
    let config = test_config(&temp_dir, false);
    let manager = GrammarManager::new(config);

    // Clearing all cache when empty should return 0
    let result = manager.clear_all_cache();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_source_parsing() {
    let source = "";
    let path = std::path::Path::new("test.rs");

    let result = extract_chunks(source, path, 1024);
    assert!(result.is_ok());
    // Empty source should produce empty chunks or single empty chunk
}

#[test]
fn test_large_source_chunking() {
    // Generate a large source file
    let mut source = String::new();
    for i in 0..100 {
        source.push_str(&format!("fn function_{}() {{ let x = {}; }}\n", i, i));
    }

    let path = std::path::Path::new("test.rs");
    let result = extract_chunks(&source, path, 512);

    assert!(result.is_ok());
    let chunks = result.unwrap();
    // Should have multiple chunks due to size limit
    assert!(chunks.len() > 1, "Large file should be split into multiple chunks");
}

#[test]
fn test_incremental_parsing() {
    // Test incremental parsing support
    let mut parser = TreeSitterParser::new("rust").expect("Should create parser");

    let source1 = "fn main() { }";
    let tree1 = parser.parse(source1).expect("Should parse");

    let source2 = "fn main() { let x = 1; }";
    let tree2 = parser
        .parse_incremental(source2, Some(&tree1))
        .expect("Should parse incrementally");

    // Both trees should be valid
    assert!(!tree1.root_node().has_error());
    assert!(!tree2.root_node().has_error());
}
