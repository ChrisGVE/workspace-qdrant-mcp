//! JavaScript language chunk extractor.

use std::path::Path;

use tree_sitter::Node;

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for JavaScript source code.
pub struct JavaScriptExtractor;

impl JavaScriptExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract preamble (imports, requires).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "import_statement" | "export_statement" => {
                    // Only include export { } from or import statements
                    let text = node_text(&child, source);
                    if text.contains("from") || child.kind() == "import_statement" {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "expression_statement" => {
                    // Check for require() calls
                    let text = node_text(&child, source);
                    if text.contains("require(") {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    } else {
                        break;
                    }
                }
                "lexical_declaration" => {
                    // const/let with require
                    let text = node_text(&child, source);
                    if text.contains("require(") {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    } else {
                        break;
                    }
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
            "javascript",
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
        let name = find_child_by_kind(node, "identifier")
            .or_else(|| find_child_by_kind(node, "property_identifier"))
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let is_async = content.trim_start().starts_with("async ");
        let chunk_type = if is_async {
            ChunkType::AsyncFunction
        } else if parent.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        // Extract signature
        let signature = content.lines().next().map(|l| l.trim_end_matches('{').trim().to_string());

        // Extract JSDoc
        let docstring = self.extract_jsdoc(node, source);

        // Extract calls
        let calls = if let Some(body) = find_child_by_kind(node, "statement_block") {
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
            "javascript",
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

        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_jsdoc(node, source);

        let mut class_chunk = SemanticChunk::new(
            ChunkType::Class,
            &name,
            content,
            start_line,
            end_line,
            "javascript",
            file_path,
        );

        if let Some(doc) = docstring {
            class_chunk = class_chunk.with_docstring(doc);
        }

        chunks.push(class_chunk);

        // Extract methods
        if let Some(body) = find_child_by_kind(node, "class_body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "method_definition" {
                    chunks.push(self.extract_function(&child, source, file_path, Some(&name)));
                }
            }
        }

        chunks
    }

    /// Extract JSDoc comment from the immediate previous sibling.
    fn extract_jsdoc(&self, node: &Node, source: &str) -> Option<String> {
        let prev = node.prev_sibling()?;
        if prev.kind() == "comment" {
            let text = node_text(&prev, source);
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
                return Some(cleaned);
            }
        }
        None
    }
}

impl ChunkExtractor for JavaScriptExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let mut parser = TreeSitterParser::new("javascript")?;
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
                "function_declaration" | "generator_function_declaration" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str, None));
                }
                "class_declaration" => {
                    chunks.extend(self.extract_class(&child, source, &file_path_str));
                }
                "lexical_declaration" | "variable_declaration" => {
                    // Check for arrow functions or function expressions
                    let mut decl_cursor = child.walk();
                    for decl_child in child.children(&mut decl_cursor) {
                        if decl_child.kind() == "variable_declarator" {
                            if let Some(value) = find_child_by_kind(&decl_child, "arrow_function")
                                .or_else(|| find_child_by_kind(&decl_child, "function"))
                            {
                                chunks.push(self.extract_function(&value, source, &file_path_str, None));
                            }
                        }
                    }
                }
                "export_statement" => {
                    // Extract exported declarations
                    if let Some(decl) = find_child_by_kind(&child, "function_declaration") {
                        chunks.push(self.extract_function(&decl, source, &file_path_str, None));
                    } else if let Some(decl) = find_child_by_kind(&child, "class_declaration") {
                        chunks.extend(self.extract_class(&decl, source, &file_path_str));
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "javascript"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let source = r#"
/**
 * Say hello.
 */
function hello() {
    console.log("Hello!");
}
"#;
        let path = PathBuf::from("test.js");
        let extractor = JavaScriptExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "hello");
    }

    #[test]
    fn test_extract_class() {
        let source = r#"
class Person {
    constructor(name) {
        this.name = name;
    }

    greet() {
        console.log(`Hello, ${this.name}!`);
    }
}
"#;
        let path = PathBuf::from("test.js");
        let extractor = JavaScriptExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let class_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Class);
        assert!(class_chunk.is_some());
        assert_eq!(class_chunk.unwrap().symbol_name, "Person");
    }

    #[test]
    fn test_extract_arrow_function() {
        let source = r#"
const greet = async (name) => {
    return `Hello, ${name}!`;
};
"#;
        let path = PathBuf::from("test.js");
        let extractor = JavaScriptExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
    }
}
