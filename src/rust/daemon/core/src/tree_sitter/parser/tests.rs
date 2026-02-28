//! Unit tests for the tree-sitter parser module.

use tree_sitter::Parser;

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
    assert_eq!(
        languages.len(),
        StaticLanguageProvider::SUPPORTED_LANGUAGES.len()
    );
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
    languages: std::collections::HashMap<String, tree_sitter::Language>,
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
    fn get_language(&self, name: &str) -> Option<tree_sitter::Language> {
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
