//! Tree-sitter integration for semantic code chunking.
//!
//! This module provides AST-based code chunking that extracts meaningful
//! semantic units (functions, classes, methods, structs, traits) from source code.

pub mod chunker;
pub mod grammar_cache;
pub mod grammar_downloader;
pub mod grammar_loader;
pub mod grammar_manager;
pub mod grammar_registry;
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
pub use grammar_registry::GrammarSource;
pub use parser::{
    get_language, get_static_language, LanguageProvider, StaticLanguageProvider, TreeSitterParser,
};
pub use types::{ChunkExtractor, ChunkType, SemanticChunk};
pub use version_checker::{
    check_grammar_compatibility, CompatibilityStatus, RuntimeInfo, VersionError,
};

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::error::DaemonError;
use crate::language_registry::providers::registry::RegistryProvider;

/// Lazily loaded language registry data derived from the YAML registry.
struct RegistryData {
    /// Set of language IDs that have grammar sources.
    known_languages: HashSet<String>,
    /// Sorted list of known language IDs (for the public API).
    known_languages_sorted: Vec<String>,
    /// Extension (without dot, lowercased) → language ID.
    extension_map: HashMap<String, String>,
}

fn registry_data() -> &'static RegistryData {
    static DATA: OnceLock<RegistryData> = OnceLock::new();
    DATA.get_or_init(|| {
        let provider = match RegistryProvider::new() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Failed to load language registry: {e}");
                return RegistryData {
                    known_languages: HashSet::new(),
                    known_languages_sorted: Vec::new(),
                    extension_map: HashMap::new(),
                };
            }
        };

        let mut known = HashSet::new();
        let mut ext_map = HashMap::new();

        for def in provider.definitions() {
            let lang_id = def.id();

            // Only add to known languages if the language has grammar sources
            if def.has_grammar() {
                known.insert(lang_id.clone());
            }

            // Build extension → language_id map (first definition wins)
            for ext in &def.extensions {
                let normalized = ext.trim_start_matches('.').to_lowercase();
                ext_map.entry(normalized).or_insert_with(|| lang_id.clone());
            }
        }

        let mut sorted: Vec<String> = known.iter().cloned().collect();
        sorted.sort();

        RegistryData {
            known_languages: known,
            known_languages_sorted: sorted,
            extension_map: ext_map,
        }
    })
}

/// Detect the language of a file from its extension using the registry.
pub fn detect_language(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?;
    let normalized = extension.to_lowercase();
    let data = registry_data();
    data.extension_map
        .get(&normalized)
        .map(|s| s.as_str())
}

/// Detect the language of a file, checking `.gitattributes` overrides first.
///
/// Detection chain: `.gitattributes` override → extension-based registry lookup.
///
/// The `relative_path` is the file's path relative to the project root,
/// used for matching against gitattributes glob patterns.
pub fn detect_language_with_overrides(
    path: &Path,
    relative_path: &str,
    overrides: &crate::patterns::GitattributesOverrides,
) -> Option<&'static str> {
    // Check gitattributes override first
    if let Some(lang) = overrides.language_override(relative_path) {
        // Map the override language to a static string if known
        let data = registry_data();
        if let Some(static_lang) = data.extension_map.values().find(|v| v.as_str() == lang) {
            return Some(static_lang.as_str());
        }
        // Also check the known_languages set directly
        if data.known_languages.contains(&lang) {
            // Find it in the sorted list for a static reference
            if let Some(found) = data
                .known_languages_sorted
                .iter()
                .find(|s| s.as_str() == lang)
            {
                return Some(found.as_str());
            }
        }
        // Language from gitattributes is not in our registry — fall through
    }

    // Fall through to extension-based detection
    detect_language(path)
}

/// Get the list of known grammar languages available for download.
pub fn known_grammar_languages() -> Vec<&'static str> {
    registry_data()
        .known_languages_sorted
        .iter()
        .map(|s| s.as_str())
        .collect()
}

/// Check if a language is supported for semantic chunking.
///
/// Checks against the registry of languages with grammar sources.
pub fn is_language_supported(language: &str) -> bool {
    registry_data().known_languages.contains(language)
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
            // Gate to known grammar languages to avoid pointless downloads.
            registry_data().known_languages.contains(language)
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
            // Language detected but not in the registry's supported list —
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
        assert!(is_language_supported("json"));
        assert!(is_language_supported("lua"));
        assert!(is_language_supported("haskell"));
        assert!(is_language_supported("swift"));
        assert!(is_language_supported("zig"));
        assert!(is_language_supported("pascal"));
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

    #[test]
    fn test_detect_language_with_overrides_no_override() {
        let overrides = crate::patterns::GitattributesOverrides::default();
        assert_eq!(
            detect_language_with_overrides(Path::new("foo.rs"), "foo.rs", &overrides),
            Some("rust")
        );
    }

    #[test]
    fn test_detect_language_with_overrides_language_override() {
        let overrides =
            crate::patterns::GitattributesOverrides::parse("*.h linguist-language=cpp\n");
        // .h normally maps to "c", but gitattributes overrides to "cpp"
        assert_eq!(
            detect_language_with_overrides(Path::new("util.h"), "util.h", &overrides),
            Some("cpp")
        );
    }

    #[test]
    fn test_detect_language_with_overrides_fallback() {
        // Override to unknown language falls back to extension
        let overrides = crate::patterns::GitattributesOverrides::parse(
            "*.rs linguist-language=unknown_lang_xyz\n",
        );
        assert_eq!(
            detect_language_with_overrides(Path::new("foo.rs"), "foo.rs", &overrides),
            Some("rust") // falls back to extension detection
        );
    }
}
