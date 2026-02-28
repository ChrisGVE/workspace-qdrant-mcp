//! Tree-sitter parser wrapper.
//!
//! Provides the [`TreeSitterParser`] struct that wraps a tree-sitter `Parser`
//! with language configuration, supporting both static and dynamic grammars.

use tree_sitter::{Language, Parser, Tree};

use crate::error::DaemonError;

use super::language_provider::{get_static_language, LanguageProvider};

/// Tree-sitter parser wrapper.
///
/// Wraps a tree-sitter Parser with language configuration and provides
/// a convenient interface for parsing source code.
pub struct TreeSitterParser {
    parser: Parser,
    language_name: String,
    language: Language,
}

impl TreeSitterParser {
    /// Create a new parser for the specified language using static loading.
    ///
    /// This constructor uses the compiled-in grammars from the `tree_sitter_*` crates.
    /// For dynamic grammar loading, use `with_language()` or `with_provider()`.
    pub fn new(language_name: &str) -> Result<Self, DaemonError> {
        let language = get_static_language(language_name).ok_or_else(|| {
            DaemonError::ParseError(format!("Unsupported language: {}", language_name))
        })?;

        Self::with_language(language_name, language)
    }

    /// Create a new parser with a pre-loaded Language.
    ///
    /// This constructor accepts a Language that was loaded externally,
    /// either statically or dynamically via `GrammarManager`.
    pub fn with_language(language_name: &str, language: Language) -> Result<Self, DaemonError> {
        let mut parser = Parser::new();
        parser.set_language(&language).map_err(|e| {
            DaemonError::ParseError(format!("Failed to set language {}: {}", language_name, e))
        })?;

        Ok(Self {
            parser,
            language_name: language_name.to_string(),
            language,
        })
    }

    /// Create a new parser using a language provider.
    ///
    /// The provider is queried for the language. This allows using either
    /// static or dynamic languages interchangeably.
    pub fn with_provider(
        language_name: &str,
        provider: &dyn LanguageProvider,
    ) -> Result<Self, DaemonError> {
        let language = provider.get_language(language_name).ok_or_else(|| {
            DaemonError::ParseError(format!(
                "Language '{}' not available from provider",
                language_name
            ))
        })?;

        Self::with_language(language_name, language)
    }

    /// Create a new parser, trying static first then a fallback provider.
    ///
    /// This is useful for graceful degradation: use compiled-in grammars when
    /// available, but fall back to dynamically loaded grammars for additional
    /// language support.
    pub fn with_fallback(
        language_name: &str,
        fallback_provider: Option<&dyn LanguageProvider>,
    ) -> Result<Self, DaemonError> {
        // Try static loading first
        if let Some(language) = get_static_language(language_name) {
            return Self::with_language(language_name, language);
        }

        // Try fallback provider
        if let Some(provider) = fallback_provider {
            if let Some(language) = provider.get_language(language_name) {
                return Self::with_language(language_name, language);
            }
        }

        Err(DaemonError::ParseError(format!(
            "Unsupported language '{}': not available statically or from provider",
            language_name
        )))
    }

    /// Parse source code and return the AST.
    pub fn parse(&mut self, source: &str) -> Result<Tree, DaemonError> {
        self.parser.parse(source, None).ok_or_else(|| {
            DaemonError::ParseError(format!(
                "Failed to parse {} source code",
                self.language_name
            ))
        })
    }

    /// Parse source code with an existing tree for incremental parsing.
    ///
    /// Incremental parsing can be significantly faster when only small
    /// portions of the source have changed.
    pub fn parse_incremental(
        &mut self,
        source: &str,
        old_tree: Option<&Tree>,
    ) -> Result<Tree, DaemonError> {
        self.parser.parse(source, old_tree).ok_or_else(|| {
            DaemonError::ParseError(format!(
                "Failed to parse {} source code (incremental)",
                self.language_name
            ))
        })
    }

    /// Get the language name.
    pub fn language_name(&self) -> &str {
        &self.language_name
    }

    /// Get the underlying tree-sitter Language.
    pub fn language(&self) -> &Language {
        &self.language
    }

    /// Reset the parser state.
    ///
    /// This should be called if parsing fails due to an error
    /// and you want to parse different content.
    pub fn reset(&mut self) {
        self.parser.reset();
    }
}
