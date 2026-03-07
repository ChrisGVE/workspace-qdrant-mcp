//! Chunking strategy selection and configuration.
//!
//! Maps language identifiers to the appropriate tree-sitter extractors,
//! supporting both statically compiled and dynamically loaded grammars.

use tree_sitter::Language;

use crate::tree_sitter::languages::{
    AdaExtractor, CExtractor, ClojureExtractor, CppExtractor, ElixirExtractor, ErlangExtractor,
    FortranExtractor, GoExtractor, HaskellExtractor, JavaExtractor, JavaScriptExtractor,
    LispExtractor, LuaExtractor, OCamlExtractor, OdinExtractor, PascalExtractor, PerlExtractor,
    PythonExtractor, RubyExtractor, RustExtractor, ScalaExtractor, ShellExtractor, SwiftExtractor,
    TypeScriptExtractor, ZigExtractor,
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
