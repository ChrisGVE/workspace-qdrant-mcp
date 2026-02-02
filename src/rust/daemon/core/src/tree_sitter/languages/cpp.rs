//! C++ language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for C++ source code.
pub struct CppExtractor {
    language: Option<Language>,
}

impl CppExtractor {
    pub fn new() -> Self {
        Self { language: None }
    }

    /// Create an extractor with a pre-loaded Language.
    pub fn with_language(language: Language) -> Self {
        Self {
            language: Some(language),
        }
    }

    fn create_parser(&self) -> Result<TreeSitterParser, DaemonError> {
        match &self.language {
            Some(lang) => TreeSitterParser::with_language("cpp", lang.clone()),
            None => TreeSitterParser::new("cpp"),
        }
    }

    /// Extract preamble (includes, using, namespace declarations at top).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "preproc_include" | "preproc_def" | "preproc_ifdef" | "preproc_ifndef"
                | "using_declaration" | "namespace_alias_definition" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "comment" => {
                    if preamble_items.is_empty() || child.start_position().row <= last_preamble_line + 1 {
                        preamble_items.push(node_text(&child, source).to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                _ => {
                    if !preamble_items.is_empty() {
                        break;
                    }
                }
            }
        }

        if preamble_items.is_empty() {
            return None;
        }

        Some(SemanticChunk::preamble(
            preamble_items.join("\n"),
            last_preamble_line,
            "cpp",
            file_path,
        ))
    }

    /// Extract a function.
    fn extract_function(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent: Option<&str>,
    ) -> SemanticChunk {
        // Try to find the function name in order of preference
        let name = if let Some(id) = find_child_by_kind(node, "identifier") {
            node_text(&id, source)
        } else if let Some(id) = find_child_by_kind(node, "field_identifier") {
            node_text(&id, source)
        } else if let Some(decl) = find_child_by_kind(node, "function_declarator") {
            find_child_by_kind(&decl, "identifier")
                .map(|n| node_text(&n, source))
                .unwrap_or("anonymous")
        } else {
            "anonymous"
        };

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let chunk_type = if parent.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        // Extract signature
        let signature = if let Some(body) = find_child_by_kind(node, "compound_statement") {
            let sig_end = body.start_byte();
            let sig = &source[node.start_byte()..sig_end].trim();
            Some(sig.to_string())
        } else {
            content.lines().next().map(|l| l.to_string())
        };

        // Extract doc comment
        let docstring = self.extract_doc_comment(node, source);

        // Extract calls
        let calls = if let Some(body) = find_child_by_kind(node, "compound_statement") {
            extract_function_calls(&body, source)
        } else {
            Vec::new()
        };

        let mut chunk = SemanticChunk::new(
            chunk_type,
            name,
            content,
            start_line,
            end_line,
            "cpp",
            file_path,
        )
        .with_calls(calls);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }
        if let Some(p) = parent {
            chunk = chunk.with_parent(p);
        }

        chunk
    }

    /// Extract a class.
    fn extract_class(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut class_chunk = SemanticChunk::new(
            ChunkType::Class,
            &name,
            content,
            start_line,
            end_line,
            "cpp",
            file_path,
        );

        if let Some(doc) = docstring {
            class_chunk = class_chunk.with_docstring(doc);
        }

        chunks.push(class_chunk);

        // Extract methods
        if let Some(body) = find_child_by_kind(node, "field_declaration_list") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "function_definition" {
                    chunks.push(self.extract_function(&child, source, file_path, Some(&name)));
                }
            }
        }

        chunks
    }

    /// Extract a struct.
    fn extract_struct(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut struct_chunk = SemanticChunk::new(
            ChunkType::Struct,
            &name,
            content,
            start_line,
            end_line,
            "cpp",
            file_path,
        );

        if let Some(doc) = docstring {
            struct_chunk = struct_chunk.with_docstring(doc);
        }

        chunks.push(struct_chunk);

        // Extract methods if any
        if let Some(body) = find_child_by_kind(node, "field_declaration_list") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "function_definition" {
                    chunks.push(self.extract_function(&child, source, file_path, Some(&name)));
                }
            }
        }

        chunks
    }

    /// Extract a namespace.
    fn extract_namespace(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "identifier")
            .or_else(|| find_child_by_kind(node, "namespace_identifier"))
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        chunks.push(SemanticChunk::new(
            ChunkType::Module,
            &name,
            content,
            start_line,
            end_line,
            "cpp",
            file_path,
        ));

        // Extract items from namespace body
        if let Some(body) = find_child_by_kind(node, "declaration_list") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                match child.kind() {
                    "function_definition" => {
                        chunks.push(self.extract_function(&child, source, file_path, Some(&name)));
                    }
                    "class_specifier" => {
                        chunks.extend(self.extract_class(&child, source, file_path));
                    }
                    "struct_specifier" => {
                        chunks.extend(self.extract_struct(&child, source, file_path));
                    }
                    _ => {}
                }
            }
        }

        chunks
    }

    /// Extract doc comment.
    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut doc_lines = Vec::new();

        while let Some(sibling) = prev {
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                if text.starts_with("/**") || text.starts_with("///") {
                    if text.starts_with("/**") {
                        let cleaned = text
                            .trim_start_matches("/**")
                            .trim_end_matches("*/")
                            .lines()
                            .map(|l| l.trim().trim_start_matches('*').trim())
                            .collect::<Vec<_>>()
                            .join("\n")
                            .trim()
                            .to_string();
                        doc_lines.push(cleaned);
                    } else {
                        doc_lines.push(text.trim_start_matches("///").trim().to_string());
                    }
                    prev = sibling.prev_sibling();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if doc_lines.is_empty() {
            None
        } else {
            doc_lines.reverse();
            Some(doc_lines.join("\n"))
        }
    }
}

impl ChunkExtractor for CppExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let mut parser = self.create_parser()?;
        let tree = parser.parse(source)?;
        let root = tree.root_node();
        let file_path_str = file_path.to_string_lossy().to_string();

        let mut chunks = Vec::new();

        if let Some(preamble) = self.extract_preamble(&root, source, &file_path_str) {
            chunks.push(preamble);
        }

        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            match child.kind() {
                "function_definition" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str, None));
                }
                "class_specifier" => {
                    chunks.extend(self.extract_class(&child, source, &file_path_str));
                }
                "struct_specifier" => {
                    chunks.extend(self.extract_struct(&child, source, &file_path_str));
                }
                "namespace_definition" => {
                    chunks.extend(self.extract_namespace(&child, source, &file_path_str));
                }
                "declaration" => {
                    // Check for class/struct definitions
                    if let Some(class_spec) = find_child_by_kind(&child, "class_specifier") {
                        chunks.extend(self.extract_class(&class_spec, source, &file_path_str));
                    } else if let Some(struct_spec) = find_child_by_kind(&child, "struct_specifier") {
                        chunks.extend(self.extract_struct(&struct_spec, source, &file_path_str));
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "cpp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let source = r#"
#include <iostream>

/// Print hello
void hello() {
    std::cout << "Hello!" << std::endl;
}
"#;
        let path = PathBuf::from("test.cpp");
        let extractor = CppExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "hello");
    }

    #[test]
    fn test_extract_class() {
        let source = r#"
class Person {
public:
    Person(std::string name) : name_(name) {}
    void greet() { std::cout << "Hello!" << std::endl; }
private:
    std::string name_;
};
"#;
        let path = PathBuf::from("test.cpp");
        let extractor = CppExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let class_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Class);
        assert!(class_chunk.is_some());
        assert_eq!(class_chunk.unwrap().symbol_name, "Person");
    }

    #[test]
    fn test_extract_namespace() {
        let source = r#"
namespace myapp {
    void helper() {}
}
"#;
        let path = PathBuf::from("test.cpp");
        let extractor = CppExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let ns_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Module);
        assert!(ns_chunk.is_some());
        assert_eq!(ns_chunk.unwrap().symbol_name, "myapp");
    }
}
