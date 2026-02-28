//! Tree-sitter parser wrapper for multiple languages.
//!
//! This module provides a wrapper around tree-sitter parsers that supports both
//! static (compiled-in) and dynamic (loaded at runtime) grammars.
//!
//! # Language Loading Strategy
//!
//! Languages can be loaded in two ways:
//!
//! 1. **Static loading** (with `static-grammars` feature): Uses grammars compiled
//!    into the binary via the `tree_sitter_*` crates. This is fast and reliable
//!    but increases binary size.
//!
//! 2. **Dynamic loading**: Uses the `GrammarManager` to load grammars from
//!    shared library files at runtime. This allows adding language support
//!    without recompilation and reduces binary size.
//!
//! The `TreeSitterParser` tries static loading first (if available), then falls
//! back to dynamic loading if a `LanguageProvider` is available.
//!
//! # Features
//!
//! - `static-grammars` (default): Include bundled tree-sitter grammars for
//!   common languages. Disable to reduce binary size by ~5-10MB.

mod language_provider;
mod tree_sitter_parser;

#[cfg(test)]
mod tests;

pub use language_provider::{
    get_language, get_static_language, LanguageProvider, StaticLanguageProvider,
};
pub use tree_sitter_parser::TreeSitterParser;
