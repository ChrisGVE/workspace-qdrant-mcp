//! Tree-sitter parser wrapper for multiple languages.
//!
//! Grammars are loaded dynamically via `GrammarManager` at runtime — no
//! static grammars are bundled. The `TreeSitterParser` uses a
//! `LanguageProvider` (typically `LoadedGrammarsProvider`) to resolve
//! languages by name.

mod language_provider;
mod tree_sitter_parser;

#[cfg(test)]
mod tests;

pub use language_provider::{
    get_language, get_static_language, LanguageProvider, StaticLanguageProvider,
};
pub use tree_sitter_parser::TreeSitterParser;
