//! Language provider infrastructure for tree-sitter grammars.
//!
//! This module defines the [`LanguageProvider`] trait and the
//! [`StaticLanguageProvider`] (always empty — grammars are loaded
//! dynamically via `GrammarManager`).

use tree_sitter::Language;

/// Trait for providing tree-sitter languages.
///
/// This trait abstracts over different ways to obtain a Language instance,
/// allowing both static (compiled-in) and dynamic (loaded at runtime) languages.
pub trait LanguageProvider: Send + Sync {
    /// Get a language by name.
    ///
    /// Returns `None` if the language is not available through this provider.
    fn get_language(&self, name: &str) -> Option<Language>;

    /// Check if this provider supports a language.
    fn supports_language(&self, name: &str) -> bool {
        self.get_language(name).is_some()
    }

    /// List available languages from this provider.
    fn available_languages(&self) -> Vec<&str>;
}

/// Static language provider — always empty.
///
/// Grammars are loaded dynamically via `GrammarManager` / `LoadedGrammarsProvider`.
/// This struct exists for API compatibility; use dynamic providers in production.
#[derive(Debug, Default, Clone)]
pub struct StaticLanguageProvider;

impl StaticLanguageProvider {
    /// Create a new static language provider.
    pub fn new() -> Self {
        Self
    }

    /// No languages are available statically (grammars are downloaded dynamically).
    pub const SUPPORTED_LANGUAGES: &'static [&'static str] = &[];
}

impl LanguageProvider for StaticLanguageProvider {
    fn get_language(&self, name: &str) -> Option<Language> {
        get_static_language(name)
    }

    fn supports_language(&self, name: &str) -> bool {
        Self::SUPPORTED_LANGUAGES.contains(&name)
    }

    fn available_languages(&self) -> Vec<&str> {
        Self::SUPPORTED_LANGUAGES.to_vec()
    }
}

/// Always returns `None` — static grammars are not bundled.
///
/// Use a dynamic `LanguageProvider` (e.g. `LoadedGrammarsProvider`) instead.
pub fn get_static_language(_name: &str) -> Option<Language> {
    None
}

/// Convenience alias for [`get_static_language`].
///
/// For dynamic loading, use `TreeSitterParser::with_provider()` or `with_fallback()`.
pub fn get_language(name: &str) -> Option<Language> {
    get_static_language(name)
}
