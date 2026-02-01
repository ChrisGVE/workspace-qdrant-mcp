//! Tree-sitter parser wrapper for multiple languages.

use std::sync::OnceLock;

use tree_sitter::{Language, Parser, Tree};

use crate::error::DaemonError;

/// Thread-safe language instances.
static RUST_LANGUAGE: OnceLock<Language> = OnceLock::new();
static PYTHON_LANGUAGE: OnceLock<Language> = OnceLock::new();
static JAVASCRIPT_LANGUAGE: OnceLock<Language> = OnceLock::new();
static TYPESCRIPT_LANGUAGE: OnceLock<Language> = OnceLock::new();
static TSX_LANGUAGE: OnceLock<Language> = OnceLock::new();
static GO_LANGUAGE: OnceLock<Language> = OnceLock::new();
static JAVA_LANGUAGE: OnceLock<Language> = OnceLock::new();
static C_LANGUAGE: OnceLock<Language> = OnceLock::new();
static CPP_LANGUAGE: OnceLock<Language> = OnceLock::new();

/// Get the tree-sitter language for Rust.
fn rust_language() -> Language {
    RUST_LANGUAGE.get_or_init(|| tree_sitter_rust::LANGUAGE.into()).clone()
}

/// Get the tree-sitter language for Python.
fn python_language() -> Language {
    PYTHON_LANGUAGE.get_or_init(|| tree_sitter_python::LANGUAGE.into()).clone()
}

/// Get the tree-sitter language for JavaScript.
fn javascript_language() -> Language {
    JAVASCRIPT_LANGUAGE.get_or_init(|| tree_sitter_javascript::LANGUAGE.into()).clone()
}

/// Get the tree-sitter language for TypeScript.
fn typescript_language() -> Language {
    TYPESCRIPT_LANGUAGE.get_or_init(|| tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()).clone()
}

/// Get the tree-sitter language for TSX.
fn tsx_language() -> Language {
    TSX_LANGUAGE.get_or_init(|| tree_sitter_typescript::LANGUAGE_TSX.into()).clone()
}

/// Get the tree-sitter language for Go.
fn go_language() -> Language {
    GO_LANGUAGE.get_or_init(|| tree_sitter_go::LANGUAGE.into()).clone()
}

/// Get the tree-sitter language for Java.
fn java_language() -> Language {
    JAVA_LANGUAGE.get_or_init(|| tree_sitter_java::LANGUAGE.into()).clone()
}

/// Get the tree-sitter language for C.
fn c_language() -> Language {
    C_LANGUAGE.get_or_init(|| tree_sitter_c::LANGUAGE.into()).clone()
}

/// Get the tree-sitter language for C++.
fn cpp_language() -> Language {
    CPP_LANGUAGE.get_or_init(|| tree_sitter_cpp::LANGUAGE.into()).clone()
}

/// Get the tree-sitter Language for a language name.
pub fn get_language(name: &str) -> Option<Language> {
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

/// Tree-sitter parser wrapper.
pub struct TreeSitterParser {
    parser: Parser,
    language_name: String,
}

impl TreeSitterParser {
    /// Create a new parser for the specified language.
    pub fn new(language_name: &str) -> Result<Self, DaemonError> {
        let language = get_language(language_name).ok_or_else(|| {
            DaemonError::ParseError(format!("Unsupported language: {}", language_name))
        })?;

        let mut parser = Parser::new();
        parser.set_language(&language).map_err(|e| {
            DaemonError::ParseError(format!("Failed to set language {}: {}", language_name, e))
        })?;

        Ok(Self {
            parser,
            language_name: language_name.to_string(),
        })
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

    /// Get the language name.
    pub fn language_name(&self) -> &str {
        &self.language_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_parser_creation() {
        let parser = TreeSitterParser::new("rust");
        assert!(parser.is_ok());

        let parser = TreeSitterParser::new("unknown");
        assert!(parser.is_err());
    }

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
}
