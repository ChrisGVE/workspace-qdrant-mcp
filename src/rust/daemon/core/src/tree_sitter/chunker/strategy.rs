//! Chunking strategy selection and configuration.
//!
//! Maps language identifiers to the appropriate tree-sitter extractors.
//! Prefers the generic pattern-driven extractor (reading patterns from the
//! bundled YAML registry) and falls back to per-language extractors when
//! patterns are not available.

use std::collections::HashMap;
use std::sync::OnceLock;

use tree_sitter::Language;

use crate::language_registry::providers::bundled::BundledProvider;
use crate::language_registry::types::SemanticPatterns;
use crate::tree_sitter::chunker::generic_extractor::GenericExtractor;
use crate::tree_sitter::languages::{
    AdaExtractor, CExtractor, ClojureExtractor, CppExtractor, ElixirExtractor, ErlangExtractor,
    FortranExtractor, GoExtractor, HaskellExtractor, JavaExtractor, JavaScriptExtractor,
    LispExtractor, LuaExtractor, OCamlExtractor, OdinExtractor, PascalExtractor, PerlExtractor,
    PythonExtractor, RubyExtractor, RustExtractor, ScalaExtractor, ShellExtractor, SwiftExtractor,
    TypeScriptExtractor, ZigExtractor,
};
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
/// Tries the generic pattern-driven extractor first (requires both a
/// `Language` grammar and YAML-defined semantic patterns). Falls back to
/// per-language extractors for languages without patterns or without a
/// dynamic grammar.
pub(super) fn create_extractor(
    language: &str,
    dynamic_lang: Option<Language>,
) -> Option<Box<dyn ChunkExtractor>> {
    // Try generic extractor if we have both a grammar and patterns
    if let Some(lang) = dynamic_lang {
        if let Some(patterns) = bundled_patterns().get(language) {
            return Some(Box::new(GenericExtractor::new(language, lang, patterns.clone())));
        }
        // Have grammar but no patterns — fall through to per-language extractors
        return create_legacy_extractor(language, Some(lang));
    }

    // No dynamic grammar — use per-language extractors with static grammars
    create_legacy_extractor(language, None)
}

/// Legacy per-language extractor creation.
///
/// This is the original match-based dispatch. It will be removed once all
/// languages are validated against the generic extractor.
fn create_legacy_extractor(
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
        "ruby" => Some(Box::new(
            dynamic_lang
                .map(RubyExtractor::with_language)
                .unwrap_or_else(RubyExtractor::new),
        )),
        "swift" => Some(Box::new(
            dynamic_lang
                .map(SwiftExtractor::with_language)
                .unwrap_or_else(SwiftExtractor::new),
        )),
        "shell" | "bash" => Some(Box::new(
            dynamic_lang
                .map(ShellExtractor::with_language)
                .unwrap_or_else(ShellExtractor::new),
        )),
        "lua" => Some(Box::new(
            dynamic_lang
                .map(LuaExtractor::with_language)
                .unwrap_or_else(LuaExtractor::new),
        )),
        "elixir" => Some(Box::new(
            dynamic_lang
                .map(ElixirExtractor::with_language)
                .unwrap_or_else(ElixirExtractor::new),
        )),
        "erlang" => Some(Box::new(
            dynamic_lang
                .map(ErlangExtractor::with_language)
                .unwrap_or_else(ErlangExtractor::new),
        )),
        "scala" => Some(Box::new(
            dynamic_lang
                .map(ScalaExtractor::with_language)
                .unwrap_or_else(ScalaExtractor::new),
        )),
        "haskell" => Some(Box::new(
            dynamic_lang
                .map(HaskellExtractor::with_language)
                .unwrap_or_else(HaskellExtractor::new),
        )),
        "zig" => Some(Box::new(
            dynamic_lang
                .map(ZigExtractor::with_language)
                .unwrap_or_else(ZigExtractor::new),
        )),
        "odin" => Some(Box::new(
            dynamic_lang
                .map(OdinExtractor::with_language)
                .unwrap_or_else(OdinExtractor::new),
        )),
        "clojure" => Some(Box::new(
            dynamic_lang
                .map(ClojureExtractor::with_language)
                .unwrap_or_else(ClojureExtractor::new),
        )),
        "ocaml" => Some(Box::new(
            dynamic_lang
                .map(OCamlExtractor::with_language)
                .unwrap_or_else(OCamlExtractor::new),
        )),
        "fortran" => Some(Box::new(
            dynamic_lang
                .map(FortranExtractor::with_language)
                .unwrap_or_else(FortranExtractor::new),
        )),
        "ada" => Some(Box::new(
            dynamic_lang
                .map(AdaExtractor::with_language)
                .unwrap_or_else(AdaExtractor::new),
        )),
        "perl" => Some(Box::new(
            dynamic_lang
                .map(PerlExtractor::with_language)
                .unwrap_or_else(PerlExtractor::new),
        )),
        "pascal" => Some(Box::new(
            dynamic_lang
                .map(PascalExtractor::with_language)
                .unwrap_or_else(PascalExtractor::new),
        )),
        "lisp" | "commonlisp" => Some(Box::new(
            dynamic_lang
                .map(LispExtractor::with_language)
                .unwrap_or_else(LispExtractor::new),
        )),
        _ => None,
    }
}

/// Try to get a Language from the provider.
pub(super) fn get_language_from_provider(
    provider: Option<&dyn LanguageProvider>,
    language_name: &str,
) -> Option<Language> {
    provider.and_then(|p| p.get_language(language_name))
}
