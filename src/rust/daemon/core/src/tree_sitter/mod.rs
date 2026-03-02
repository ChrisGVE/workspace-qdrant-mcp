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
pub use grammar_cache::{GrammarCachePaths, GrammarMetadata};
pub use grammar_downloader::{DownloadError, GrammarDownloader};
pub use grammar_loader::{GrammarLoader, LoadedGrammar};
pub use grammar_manager::{
    create_grammar_manager, GrammarError, GrammarInfo, GrammarManager, GrammarResult,
    GrammarStatus, GrammarValidationResult, LoadedGrammarsProvider,
};
pub use parser::{
    get_language, get_static_language, LanguageProvider, StaticLanguageProvider, TreeSitterParser,
};
pub use types::{ChunkExtractor, ChunkType, SemanticChunk};
pub use version_checker::{
    check_grammar_compatibility, CompatibilityStatus, RuntimeInfo, VersionError,
};

use std::path::Path;
use std::sync::Arc;

use crate::error::DaemonError;

/// Known languages with well-known tree-sitter grammars.
///
/// These languages have grammars available for download from the default
/// grammar sources. When `auto_download` is enabled, any of these languages
/// can be used for semantic chunking.
const KNOWN_GRAMMAR_LANGUAGES: &[&str] = &[
    "rust",
    "python",
    "javascript",
    "typescript",
    "tsx",
    "jsx",
    "go",
    "java",
    "c",
    "cpp",
];

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
///
/// Without a `GrammarManager`, checks against the hardcoded list of known
/// grammar languages. With a manager, also considers dynamically cached
/// or downloadable grammars.
pub fn is_language_supported(language: &str) -> bool {
    KNOWN_GRAMMAR_LANGUAGES.contains(&language)
}

/// Check if a language has an available grammar via the GrammarManager.
///
/// Returns true if the grammar is loaded, cached, or known to be downloadable.
/// This is the dynamic-aware version of `is_language_supported`.
///
/// When `auto_download` is enabled, a language is considered available if it
/// is in the known grammar list OR already cached/loaded. When `auto_download`
/// is disabled, only loaded or cached grammars count as available.
pub fn is_language_available(language: &str, manager: &GrammarManager) -> bool {
    let status = manager.grammar_status(language);
    match status {
        GrammarStatus::Loaded | GrammarStatus::Cached => true,
        GrammarStatus::NeedsDownload => {
            // The manager reports NeedsDownload for any language when auto_download
            // is on. Gate this to known grammar languages to avoid pointless downloads.
            KNOWN_GRAMMAR_LANGUAGES.contains(&language)
        }
        _ => false,
    }
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
    extract_chunks_with_provider(source, path, max_chunk_size, None)
}

/// Extract semantic chunks with an optional dynamic language provider.
///
/// When a provider is given, it supplies dynamically-loaded grammars for
/// languages beyond the statically compiled set. This enables first-use
/// grammar download: the caller ensures the grammar is loaded (async) and
/// passes a snapshot provider into this synchronous function.
pub fn extract_chunks_with_provider(
    source: &str,
    path: &Path,
    max_chunk_size: usize,
    provider: Option<Arc<dyn LanguageProvider>>,
) -> Result<Vec<SemanticChunk>, DaemonError> {
    let language = match detect_language(path) {
        Some(lang) if is_language_supported(lang) => lang,
        Some(lang) => {
            // Language detected but not in the hardcoded supported list —
            // check whether the provider has a grammar for it
            if provider
                .as_ref()
                .map_or(false, |p| p.supports_language(lang))
            {
                lang
            } else {
                return Ok(chunker::text_chunk_fallback(source, path, max_chunk_size));
            }
        }
        _ => {
            // Fall back to text chunking for unsupported languages
            return Ok(chunker::text_chunk_fallback(source, path, max_chunk_size));
        }
    };

    let chunker = match provider {
        Some(p) => SemanticChunker::with_provider(max_chunk_size, p),
        None => SemanticChunker::new(max_chunk_size),
    };
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

    #[test]
    fn test_is_language_available_with_auto_download() {
        use crate::config::GrammarConfig;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let config = GrammarConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            auto_download: true,
            ..Default::default()
        };
        let manager = GrammarManager::new(config);

        // Known languages should be available (NeedsDownload) with auto_download
        assert!(is_language_available("rust", &manager));
        assert!(is_language_available("python", &manager));

        // Unknown languages should not be available
        assert!(!is_language_available("unknown_lang", &manager));
    }

    #[test]
    fn test_is_language_available_without_auto_download() {
        use crate::config::GrammarConfig;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let config = GrammarConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            auto_download: false,
            ..Default::default()
        };
        let manager = GrammarManager::new(config);

        // Without auto_download, uncached grammars are NotAvailable
        assert!(!is_language_available("rust", &manager));
    }
}
