//! Integration tests for grammar parsing — language detection, supported
//! languages, provider-based chunking, and SemanticChunker.

use std::sync::Arc;
use workspace_qdrant_core::tree_sitter::{
    extract_chunks, extract_chunks_with_provider, known_grammar_languages, LanguageProvider,
    LoadedGrammarsProvider, SemanticChunker, StaticLanguageProvider,
};

// =============================================================================
// All Supported Languages Tests
// =============================================================================

#[test]
fn test_all_supported_languages_parsing() {
    use workspace_qdrant_core::tree_sitter::get_static_language;
    if get_static_language("rust").is_none() {
        return;
    }

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
        ("test.json", Some("json"), true),
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
    if provider.available_languages().is_empty() {
        return;
    }
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
    assert!(chunks.iter().any(|c| c.symbol_name == "hello"));
    assert!(chunks.iter().any(|c| c.symbol_name == "world"));
}

#[test]
fn test_chunker_language_provider_snapshot() {
    use tempfile::TempDir;
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    let temp_dir = TempDir::new().unwrap();
    let config = GrammarConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        required: vec!["rust".to_string(), "python".to_string()],
        auto_download: false,
        tree_sitter_version: "0.24".to_string(),
        download_base_url: "https://example.com".to_string(),
        verify_checksums: false,
        lazy_loading: true,
        check_interval_hours: 168,
        ..Default::default()
    };
    let manager = GrammarManager::new(config);
    let provider = manager.create_language_provider();

    assert!(provider.is_empty());
    assert!(provider.available_languages().is_empty());
}

// =============================================================================
// Provider-based extract_chunks Tests
// =============================================================================

#[test]
fn test_known_grammar_languages_list() {
    let known = known_grammar_languages();
    assert!(known.contains(&"rust"), "rust should be in known grammars");
    assert!(
        known.contains(&"python"),
        "python should be in known grammars"
    );
    assert!(
        known.contains(&"javascript"),
        "javascript should be in known grammars"
    );
    assert!(
        known.contains(&"typescript"),
        "typescript should be in known grammars"
    );
    assert!(known.contains(&"tsx"), "tsx should be in known grammars");
    assert!(known.contains(&"jsx"), "jsx should be in known grammars");
    assert!(known.contains(&"go"), "go should be in known grammars");
    assert!(known.contains(&"java"), "java should be in known grammars");
    assert!(known.contains(&"c"), "c should be in known grammars");
    assert!(known.contains(&"cpp"), "cpp should be in known grammars");
    assert!(
        known.contains(&"json"),
        "json should be in known grammars"
    );
    assert!(
        !known.contains(&"unknown"),
        "unknown should not be in known grammars"
    );
}

#[test]
fn test_extract_chunks_with_provider_none_unsupported() {
    let source = "some text content for testing";
    let path = std::path::Path::new("test.xyz");
    let result = extract_chunks_with_provider(source, path, 1024, None);
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty());
}

#[test]
fn test_extract_chunks_with_provider_none_supported() {
    use workspace_qdrant_core::tree_sitter::get_static_language;
    if get_static_language("rust").is_none() {
        return;
    }
    let source = r#"fn hello() { println!("hi"); }"#;
    let path = std::path::Path::new("test.rs");
    let result = extract_chunks_with_provider(source, path, 1024, None);
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty());
}

#[test]
fn test_extract_chunks_with_provider_static() {
    let provider = Arc::new(StaticLanguageProvider::new());
    if provider.available_languages().is_empty() {
        return;
    }

    let source = r#"
fn hello() {
    println!("Hello");
}

fn world() {
    println!("World");
}
"#;
    let path = std::path::Path::new("test.rs");
    let result = extract_chunks_with_provider(source, path, 1024, Some(provider));
    assert!(result.is_ok());
    assert!(result.unwrap().len() >= 2, "Should extract function chunks");
}

#[test]
fn test_extract_chunks_with_provider_unsupported_language() {
    let provider = Arc::new(StaticLanguageProvider::new());
    let source = "some content in an unknown format";
    let path = std::path::Path::new("test.xyz");

    let result = extract_chunks_with_provider(source, path, 1024, Some(provider));
    assert!(result.is_ok());
    for chunk in result.unwrap() {
        assert_eq!(
            chunk.chunk_type,
            workspace_qdrant_core::tree_sitter::ChunkType::Text,
            "Unknown language should fall back to text chunking"
        );
    }
}

#[test]
fn test_loaded_grammars_provider_used_by_extract_chunks() {
    use workspace_qdrant_core::tree_sitter::get_static_language;

    let mut provider = LoadedGrammarsProvider::new();
    if let Some(lang) = get_static_language("python") {
        provider.add_grammar("python", lang);
    } else {
        return;
    }

    let source = r#"
def greet(name):
    print(f"Hello, {name}")

def farewell(name):
    print(f"Goodbye, {name}")
"#;
    let path = std::path::Path::new("test.py");
    let result = extract_chunks_with_provider(source, path, 1024, Some(Arc::new(provider)));
    assert!(result.is_ok());
    assert!(
        !result.unwrap().is_empty(),
        "Should produce chunks from Python source"
    );
}
