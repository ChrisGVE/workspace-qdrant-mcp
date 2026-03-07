//! Chunking strategy selection and configuration.
//!
//! Maps language identifiers to the appropriate tree-sitter extractor.
//! Uses the generic pattern-driven extractor with YAML-defined semantic
//! patterns from the bundled language registry.

use std::collections::HashMap;
use std::sync::OnceLock;

use tree_sitter::Language;

use crate::language_registry::providers::bundled::BundledProvider;
use crate::language_registry::types::SemanticPatterns;
use crate::tree_sitter::chunker::generic_extractor::GenericExtractor;
use crate::tree_sitter::parser::LanguageProvider;
use crate::tree_sitter::types::ChunkExtractor;

/// Lazily loaded semantic patterns from the bundled YAML registry.
fn bundled_patterns() -> &'static HashMap<String, SemanticPatterns> {
    static PATTERNS: OnceLock<HashMap<String, SemanticPatterns>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let provider = match BundledProvider::new() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Failed to load bundled language definitions: {e}");
                return HashMap::new();
            }
        };
        let mut map = HashMap::new();
        for def in provider.definitions() {
            if let Some(patterns) = &def.semantic_patterns {
                map.insert(def.id(), patterns.clone());
            }
        }
        map
    })
}

/// Create an extractor for the given language.
///
/// Requires both a dynamic `Language` grammar and YAML-defined semantic
/// patterns. Returns `None` if either is missing (caller falls back to
/// text chunking).
pub(super) fn create_extractor(
    language: &str,
    dynamic_lang: Option<Language>,
) -> Option<Box<dyn ChunkExtractor>> {
    let lang = dynamic_lang?;
    let patterns = bundled_patterns().get(language)?;
    Some(Box::new(GenericExtractor::new(
        language,
        lang,
        patterns.clone(),
    )))
}

/// Try to get a Language from the provider.
pub(super) fn get_language_from_provider(
    provider: Option<&dyn LanguageProvider>,
    language_name: &str,
) -> Option<Language> {
    provider.and_then(|p| p.get_language(language_name))
}
