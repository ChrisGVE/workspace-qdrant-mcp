//! Tree-sitter integration for semantic code chunking.
//!
//! This module provides AST-based code chunking that extracts meaningful
//! semantic units (functions, classes, methods, structs, traits) from source code.

pub mod chunker;
pub mod grammar_cache;
pub mod grammar_downloader;
pub mod grammar_loader;
pub mod grammar_manager;
pub mod languages;
pub mod parser;
pub mod types;
pub mod version_checker;

pub use chunker::SemanticChunker;
pub use parser::{
    get_language, get_static_language, LanguageProvider, StaticLanguageProvider, TreeSitterParser,
};
pub use types::{ChunkExtractor, ChunkType, SemanticChunk};

use std::path::Path;

use crate::error::DaemonError;

/// Language registry mapping file extensions to language identifiers.
pub fn detect_language(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?;
    match extension.to_lowercase().as_str() {
        "rs" => Some("rust"),
        "py" | "pyi" | "pyw" => Some("python"),
        "js" | "mjs" | "cjs" => Some("javascript"),
        "ts" | "mts" | "cts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "jsx" => Some("jsx"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some("cpp"),
        "json" => Some("json"),
        _ => None,
    }
}

/// Check if a language is supported for semantic chunking.
pub fn is_language_supported(language: &str) -> bool {
    matches!(
        language,
        "rust" | "python" | "javascript" | "typescript" | "tsx" | "jsx" | "go" | "java" | "c" | "cpp"
    )
}

/// Extract semantic chunks from source code.
///
/// This is the main entry point for semantic code chunking.
/// Falls back to text chunking if the language is not supported or parsing fails.
pub fn extract_chunks(
    source: &str,
    path: &Path,
    max_chunk_size: usize,
) -> Result<Vec<SemanticChunk>, DaemonError> {
    let language = match detect_language(path) {
        Some(lang) if is_language_supported(lang) => lang,
        _ => {
            // Fall back to text chunking for unsupported languages
            return Ok(chunker::text_chunk_fallback(source, path, max_chunk_size));
        }
    };

    let chunker = SemanticChunker::new(max_chunk_size);
    chunker.chunk_source(source, path, language)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Path::new("foo.rs")), Some("rust"));
        assert_eq!(detect_language(Path::new("bar.py")), Some("python"));
        assert_eq!(detect_language(Path::new("baz.js")), Some("javascript"));
        assert_eq!(detect_language(Path::new("qux.ts")), Some("typescript"));
        assert_eq!(detect_language(Path::new("main.go")), Some("go"));
        assert_eq!(detect_language(Path::new("App.java")), Some("java"));
        assert_eq!(detect_language(Path::new("util.c")), Some("c"));
        assert_eq!(detect_language(Path::new("util.cpp")), Some("cpp"));
        assert_eq!(detect_language(Path::new("data.json")), Some("json"));
        assert_eq!(detect_language(Path::new("unknown.xyz")), None);
    }

    #[test]
    fn test_is_language_supported() {
        assert!(is_language_supported("rust"));
        assert!(is_language_supported("python"));
        assert!(is_language_supported("javascript"));
        assert!(is_language_supported("typescript"));
        assert!(is_language_supported("go"));
        assert!(is_language_supported("java"));
        assert!(is_language_supported("c"));
        assert!(is_language_supported("cpp"));
        assert!(!is_language_supported("json")); // JSON not supported for semantic chunking
        assert!(!is_language_supported("unknown"));
    }
}
