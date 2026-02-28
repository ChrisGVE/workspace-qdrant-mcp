//! Chunking strategy selection and configuration.
//!
//! Maps language identifiers to the appropriate tree-sitter extractors,
//! supporting both statically compiled and dynamically loaded grammars.

use tree_sitter::Language;

use crate::tree_sitter::languages::{
    CExtractor, CppExtractor, GoExtractor, JavaExtractor, JavaScriptExtractor, PythonExtractor,
    RustExtractor, TypeScriptExtractor,
};
use crate::tree_sitter::parser::LanguageProvider;
use crate::tree_sitter::types::ChunkExtractor;

/// Create an extractor for the given language.
///
/// If a `dynamic_lang` is provided (from a `LanguageProvider`), the extractor
/// will use it. Otherwise, it falls back to static grammars for supported
/// languages.
pub(super) fn create_extractor(
    language: &str,
    dynamic_lang: Option<Language>,
) -> Option<Box<dyn ChunkExtractor>> {
    match language {
        "rust" => Some(Box::new(
            dynamic_lang
                .map(RustExtractor::with_language)
                .unwrap_or_else(RustExtractor::new),
        )),
        "python" => Some(Box::new(
            dynamic_lang
                .map(PythonExtractor::with_language)
                .unwrap_or_else(PythonExtractor::new),
        )),
        "javascript" | "jsx" => Some(Box::new(
            dynamic_lang
                .map(JavaScriptExtractor::with_language)
                .unwrap_or_else(JavaScriptExtractor::new),
        )),
        "typescript" | "tsx" => Some(Box::new(
            dynamic_lang
                .map(|l| TypeScriptExtractor::with_language(l, language == "tsx"))
                .unwrap_or_else(|| TypeScriptExtractor::new(language == "tsx")),
        )),
        "go" => Some(Box::new(
            dynamic_lang
                .map(GoExtractor::with_language)
                .unwrap_or_else(GoExtractor::new),
        )),
        "java" => Some(Box::new(
            dynamic_lang
                .map(JavaExtractor::with_language)
                .unwrap_or_else(JavaExtractor::new),
        )),
        "c" => Some(Box::new(
            dynamic_lang
                .map(CExtractor::with_language)
                .unwrap_or_else(CExtractor::new),
        )),
        "cpp" => Some(Box::new(
            dynamic_lang
                .map(CppExtractor::with_language)
                .unwrap_or_else(CppExtractor::new),
        )),
        _ => {
            // For unknown languages, check if provider has a grammar
            // Currently we can only use known extractors, so return None
            None
        }
    }
}

/// Try to get a Language from the provider.
pub(super) fn get_language_from_provider(
    provider: Option<&dyn LanguageProvider>,
    language_name: &str,
) -> Option<Language> {
    provider.and_then(|p| p.get_language(language_name))
}
