//! Tree-sitter parser wrapper for multiple languages.
//!
//! This module provides a wrapper around tree-sitter parsers that supports both
//! static (compiled-in) and dynamic (loaded at runtime) grammars.
//!
//! # Language Loading Strategy
//!
//! Languages can be loaded in two ways:
//!
//! 1. **Static loading** (default): Uses grammars compiled into the binary via
//!    the `tree_sitter_*` crates. This is fast and reliable but requires
//!    recompilation to add new languages.
//!
//! 2. **Dynamic loading**: Uses the `GrammarManager` to load grammars from
//!    shared library files at runtime. This allows adding language support
//!    without recompilation.
//!
//! The `TreeSitterParser` tries static loading first, then falls back to
//! dynamic loading if a `LanguageProvider` is available.

use std::sync::OnceLock;

use tree_sitter::{Language, Parser, Tree};

use crate::error::DaemonError;

/// Thread-safe language instances for static loading.
static RUST_LANGUAGE: OnceLock<Language> = OnceLock::new();
static PYTHON_LANGUAGE: OnceLock<Language> = OnceLock::new();
static JAVASCRIPT_LANGUAGE: OnceLock<Language> = OnceLock::new();
static TYPESCRIPT_LANGUAGE: OnceLock<Language> = OnceLock::new();
static TSX_LANGUAGE: OnceLock<Language> = OnceLock::new();
static GO_LANGUAGE: OnceLock<Language> = OnceLock::new();
static JAVA_LANGUAGE: OnceLock<Language> = OnceLock::new();
static C_LANGUAGE: OnceLock<Language> = OnceLock::new();
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
/// This is the default provider that uses the `tree_sitter_*` crates.
#[derive(Debug, Default, Clone)]
pub struct StaticLanguageProvider;

impl StaticLanguageProvider {
    /// Create a new static language provider.
    pub fn new() -> Self {
        Self
    }

    /// List of languages supported by the static provider.
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

/// Get the tree-sitter language for Rust (static).
fn rust_language() -> Language {
    RUST_LANGUAGE
        .get_or_init(|| tree_sitter_rust::LANGUAGE.into())
        .clone()
}

/// Get the tree-sitter language for Python (static).
fn python_language() -> Language {
    PYTHON_LANGUAGE
        .get_or_init(|| tree_sitter_python::LANGUAGE.into())
        .clone()
}

/// Get the tree-sitter language for JavaScript (static).
fn javascript_language() -> Language {
    JAVASCRIPT_LANGUAGE
        .get_or_init(|| tree_sitter_javascript::LANGUAGE.into())
        .clone()
}

/// Get the tree-sitter language for TypeScript (static).
fn typescript_language() -> Language {
    TYPESCRIPT_LANGUAGE
        .get_or_init(|| tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .clone()
}

/// Get the tree-sitter language for TSX (static).
fn tsx_language() -> Language {
    TSX_LANGUAGE
        .get_or_init(|| tree_sitter_typescript::LANGUAGE_TSX.into())
        .clone()
}

/// Get the tree-sitter language for Go (static).
fn go_language() -> Language {
    GO_LANGUAGE
        .get_or_init(|| tree_sitter_go::LANGUAGE.into())
        .clone()
}

/// Get the tree-sitter language for Java (static).
fn java_language() -> Language {
    JAVA_LANGUAGE
        .get_or_init(|| tree_sitter_java::LANGUAGE.into())
        .clone()
}

/// Get the tree-sitter language for C (static).
fn c_language() -> Language {
    C_LANGUAGE
        .get_or_init(|| tree_sitter_c::LANGUAGE.into())
        .clone()
}

/// Get the tree-sitter language for C++ (static).
fn cpp_language() -> Language {
    CPP_LANGUAGE
        .get_or_init(|| tree_sitter_cpp::LANGUAGE.into())
        .clone()
}

/// Get a statically-compiled tree-sitter Language for a language name.
///
/// This function only returns languages that are compiled into the binary.
/// For dynamically loaded languages, use a `LanguageProvider` implementation.
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

/// Get the tree-sitter Language for a language name.
///
/// This is a convenience function that uses static loading only.
/// For dynamic loading support, use `TreeSitterParser::with_provider()`.
pub fn get_language(name: &str) -> Option<Language> {
    get_static_language(name)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    // Test static language provider
    #[test]
    fn test_static_provider_supported_languages() {
        let provider = StaticLanguageProvider::new();
        assert!(provider.supports_language("rust"));
        assert!(provider.supports_language("python"));
        assert!(provider.supports_language("javascript"));
        assert!(provider.supports_language("jsx"));
        assert!(provider.supports_language("typescript"));
        assert!(provider.supports_language("tsx"));
        assert!(provider.supports_language("go"));
        assert!(provider.supports_language("java"));
        assert!(provider.supports_language("c"));
        assert!(provider.supports_language("cpp"));
        assert!(!provider.supports_language("unknown"));
    }

    #[test]
    fn test_static_provider_get_language() {
        let provider = StaticLanguageProvider::new();
        assert!(provider.get_language("rust").is_some());
        assert!(provider.get_language("python").is_some());
        assert!(provider.get_language("unknown").is_none());
    }

    #[test]
    fn test_static_provider_available_languages() {
        let provider = StaticLanguageProvider::new();
        let languages = provider.available_languages();
        assert!(languages.contains(&"rust"));
        assert!(languages.contains(&"python"));
        assert_eq!(languages.len(), StaticLanguageProvider::SUPPORTED_LANGUAGES.len());
    }

    // Test get_language and get_static_language
    #[test]
    fn test_get_language() {
        assert!(get_language("rust").is_some());
        assert!(get_language("python").is_some());
        assert!(get_language("javascript").is_some());
        assert!(get_language("typescript").is_some());
        assert!(get_language("tsx").is_some());
        assert!(get_language("go").is_some());
        assert!(get_language("java").is_some());
        assert!(get_language("c").is_some());
        assert!(get_language("cpp").is_some());
        assert!(get_language("unknown").is_none());
    }

    #[test]
    fn test_get_static_language() {
        assert!(get_static_language("rust").is_some());
        assert!(get_static_language("jsx").is_some()); // jsx maps to javascript
        assert!(get_static_language("unknown").is_none());
    }

    // Test parser creation methods
    #[test]
    fn test_parser_creation_new() {
        let parser = TreeSitterParser::new("rust");
        assert!(parser.is_ok());
        let parser = parser.unwrap();
        assert_eq!(parser.language_name(), "rust");

        let parser = TreeSitterParser::new("unknown");
        assert!(parser.is_err());
    }

    #[test]
    fn test_parser_creation_with_language() {
        let language = get_static_language("rust").unwrap();
        let parser = TreeSitterParser::with_language("rust", language);
        assert!(parser.is_ok());
        let parser = parser.unwrap();
        assert_eq!(parser.language_name(), "rust");
    }

    #[test]
    fn test_parser_creation_with_provider() {
        let provider = StaticLanguageProvider::new();
        let parser = TreeSitterParser::with_provider("rust", &provider);
        assert!(parser.is_ok());

        let parser = TreeSitterParser::with_provider("unknown", &provider);
        assert!(parser.is_err());
    }

    #[test]
    fn test_parser_creation_with_fallback() {
        // Static language should work without fallback
        let parser = TreeSitterParser::with_fallback("rust", None);
        assert!(parser.is_ok());

        // Unknown language should fail without fallback
        let parser = TreeSitterParser::with_fallback("unknown", None);
        assert!(parser.is_err());

        // With fallback provider for static languages
        let provider = StaticLanguageProvider::new();
        let parser = TreeSitterParser::with_fallback("python", Some(&provider));
        assert!(parser.is_ok());
    }

    #[test]
    fn test_parser_language_accessor() {
        let parser = TreeSitterParser::new("rust").unwrap();
        let language = parser.language();
        // Language should be valid (can create a parser from it)
        let mut test_parser = Parser::new();
        assert!(test_parser.set_language(language).is_ok());
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = TreeSitterParser::new("rust").unwrap();
        let source = "fn test() {}";
        let _ = parser.parse(source);
        parser.reset();
        // Should still work after reset
        let tree = parser.parse(source);
        assert!(tree.is_ok());
    }

    // Test parsing various languages
    #[test]
    fn test_parse_rust() {
        let mut parser = TreeSitterParser::new("rust").unwrap();
        let source = r#"
fn hello() {
    println!("Hello, world!");
}
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());

        let tree = tree.unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "source_file");
    }

    #[test]
    fn test_parse_python() {
        let mut parser = TreeSitterParser::new("python").unwrap();
        let source = r#"
def hello():
    print("Hello, world!")
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());

        let tree = tree.unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "module");
    }

    #[test]
    fn test_parse_javascript() {
        let mut parser = TreeSitterParser::new("javascript").unwrap();
        let source = r#"
function hello() {
    console.log("Hello, world!");
}
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());
    }

    #[test]
    fn test_parse_typescript() {
        let mut parser = TreeSitterParser::new("typescript").unwrap();
        let source = r#"
function hello(): void {
    console.log("Hello, world!");
}
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());
    }

    #[test]
    fn test_parse_go() {
        let mut parser = TreeSitterParser::new("go").unwrap();
        let source = r#"
package main

func hello() {
    fmt.Println("Hello, world!")
}
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());
    }

    #[test]
    fn test_parse_java() {
        let mut parser = TreeSitterParser::new("java").unwrap();
        let source = r#"
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
}
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());
    }

    #[test]
    fn test_parse_c() {
        let mut parser = TreeSitterParser::new("c").unwrap();
        let source = r#"
#include <stdio.h>

int main() {
    printf("Hello, world!\n");
    return 0;
}
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());
    }

    #[test]
    fn test_parse_cpp() {
        let mut parser = TreeSitterParser::new("cpp").unwrap();
        let source = r#"
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
"#;
        let tree = parser.parse(source);
        assert!(tree.is_ok());
    }

    #[test]
    fn test_parse_incremental() {
        let mut parser = TreeSitterParser::new("rust").unwrap();
        let source_v1 = "fn foo() { 1 }";
        let tree_v1 = parser.parse(source_v1).unwrap();

        // Parse again with old tree for incremental parsing
        let source_v2 = "fn foo() { 2 }";
        let tree_v2 = parser.parse_incremental(source_v2, Some(&tree_v1));
        assert!(tree_v2.is_ok());

        let tree_v2 = tree_v2.unwrap();
        let root = tree_v2.root_node();
        assert_eq!(root.kind(), "source_file");
    }

    // Test custom language provider
    struct MockProvider {
        languages: std::collections::HashMap<String, Language>,
    }

    impl MockProvider {
        fn new() -> Self {
            let mut languages = std::collections::HashMap::new();
            // Add rust from static provider for testing
            if let Some(lang) = get_static_language("rust") {
                languages.insert("mock_rust".to_string(), lang);
            }
            Self { languages }
        }
    }

    impl LanguageProvider for MockProvider {
        fn get_language(&self, name: &str) -> Option<Language> {
            self.languages.get(name).cloned()
        }

        fn available_languages(&self) -> Vec<&str> {
            self.languages.keys().map(|s| s.as_str()).collect()
        }
    }

    #[test]
    fn test_custom_language_provider() {
        let provider = MockProvider::new();
        assert!(provider.supports_language("mock_rust"));
        assert!(!provider.supports_language("rust")); // Not in mock

        let parser = TreeSitterParser::with_provider("mock_rust", &provider);
        assert!(parser.is_ok());

        let mut parser = parser.unwrap();
        let tree = parser.parse("fn test() {}");
        assert!(tree.is_ok());
    }

    #[test]
    fn test_fallback_to_custom_provider() {
        let provider = MockProvider::new();

        // "rust" is static, should succeed without using provider
        let parser = TreeSitterParser::with_fallback("rust", Some(&provider));
        assert!(parser.is_ok());

        // "mock_rust" is only in provider, should use fallback
        let parser = TreeSitterParser::with_fallback("mock_rust", Some(&provider));
        assert!(parser.is_ok());

        // "totally_unknown" should fail even with provider
        let parser = TreeSitterParser::with_fallback("totally_unknown", Some(&provider));
        assert!(parser.is_err());
    }
}
