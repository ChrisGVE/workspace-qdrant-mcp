//! Language provider infrastructure for tree-sitter grammars.
//!
//! This module defines the [`LanguageProvider`] trait and the
//! [`StaticLanguageProvider`] implementation that loads grammars compiled
//! into the binary via the `tree_sitter_*` crates.

use tree_sitter::Language;

// Static language instances (only when static-grammars feature is enabled)
#[cfg(feature = "static-grammars")]
use std::sync::OnceLock;

#[cfg(feature = "static-grammars")]
static RUST_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static PYTHON_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static JAVASCRIPT_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static TYPESCRIPT_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static TSX_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static GO_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static JAVA_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static C_LANGUAGE: OnceLock<Language> = OnceLock::new();
#[cfg(feature = "static-grammars")]
static CPP_LANGUAGE: OnceLock<Language> = OnceLock::new();

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

/// Static language provider using compiled-in grammars.
///
/// This provider uses the `tree_sitter_*` crates that are compiled into the binary.
/// It's only available when the `static-grammars` feature is enabled.
///
/// When `static-grammars` is disabled, use dynamic loading via `GrammarManager`.
#[derive(Debug, Default, Clone)]
pub struct StaticLanguageProvider;

impl StaticLanguageProvider {
    /// Create a new static language provider.
    pub fn new() -> Self {
        Self
    }

    /// List of languages supported by the static provider.
    ///
    /// Note: This list is populated only when `static-grammars` feature is enabled.
    #[cfg(feature = "static-grammars")]
    pub const SUPPORTED_LANGUAGES: &'static [&'static str] = &[
        "rust",
        "python",
        "javascript",
        "jsx",
        "typescript",
        "tsx",
        "go",
        "java",
        "c",
        "cpp",
    ];

    /// When static-grammars is disabled, no languages are supported statically.
    #[cfg(not(feature = "static-grammars"))]
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

// Static language loading functions (only when static-grammars feature is enabled)

#[cfg(feature = "static-grammars")]
fn rust_language() -> Language {
    RUST_LANGUAGE
        .get_or_init(|| tree_sitter_rust::LANGUAGE.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn python_language() -> Language {
    PYTHON_LANGUAGE
        .get_or_init(|| tree_sitter_python::LANGUAGE.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn javascript_language() -> Language {
    JAVASCRIPT_LANGUAGE
        .get_or_init(|| tree_sitter_javascript::LANGUAGE.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn typescript_language() -> Language {
    TYPESCRIPT_LANGUAGE
        .get_or_init(|| tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn tsx_language() -> Language {
    TSX_LANGUAGE
        .get_or_init(|| tree_sitter_typescript::LANGUAGE_TSX.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn go_language() -> Language {
    GO_LANGUAGE
        .get_or_init(|| tree_sitter_go::LANGUAGE.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn java_language() -> Language {
    JAVA_LANGUAGE
        .get_or_init(|| tree_sitter_java::LANGUAGE.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn c_language() -> Language {
    C_LANGUAGE
        .get_or_init(|| tree_sitter_c::LANGUAGE.into())
        .clone()
}

#[cfg(feature = "static-grammars")]
fn cpp_language() -> Language {
    CPP_LANGUAGE
        .get_or_init(|| tree_sitter_cpp::LANGUAGE.into())
        .clone()
}

/// Get a statically-compiled tree-sitter Language for a language name.
///
/// This function only returns languages that are compiled into the binary
/// when the `static-grammars` feature is enabled.
/// For dynamically loaded languages, use a `LanguageProvider` implementation.
#[cfg(feature = "static-grammars")]
pub fn get_static_language(name: &str) -> Option<Language> {
    match name {
        "rust" => Some(rust_language()),
        "python" => Some(python_language()),
        "javascript" | "jsx" => Some(javascript_language()),
        "typescript" => Some(typescript_language()),
        "tsx" => Some(tsx_language()),
        "go" => Some(go_language()),
        "java" => Some(java_language()),
        "c" => Some(c_language()),
        "cpp" => Some(cpp_language()),
        _ => None,
    }
}

/// Without static-grammars feature, no languages are available statically.
#[cfg(not(feature = "static-grammars"))]
pub fn get_static_language(_name: &str) -> Option<Language> {
    None
}

/// Get the tree-sitter Language for a language name.
///
/// This is a convenience function that uses static loading only.
/// For dynamic loading support, use `TreeSitterParser::with_provider()`.
pub fn get_language(name: &str) -> Option<Language> {
    get_static_language(name)
}
