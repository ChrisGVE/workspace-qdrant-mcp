//! JavaScript language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

use super::helpers::{build_function_chunk, clean_block_doc_comment, first_line_signature};

/// Extractor for JavaScript source code.
pub struct JavaScriptExtractor {
    language: Option<Language>,
}

impl JavaScriptExtractor {
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
            Some(lang) => TreeSitterParser::with_language("javascript", lang.clone()),
            None => TreeSitterParser::new("javascript"),
        }
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

    /// Extract JSDoc comment from the immediate previous sibling.
    fn extract_jsdoc(&self, node: &Node, source: &str) -> Option<String> {
        let prev = node.prev_sibling()?;
        if prev.kind() == "comment" {
            clean_block_doc_comment(&prev, source)
        } else {
            None
        }
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
        let is_async = content.trim_start().starts_with("async ");
        let chunk_type = if is_async {
            ChunkType::AsyncFunction
        } else if parent.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        build_function_chunk(
            chunk_type,
            name,
            node,
            source,
            file_path,
            "javascript",
            "statement_block",
            first_line_signature(node, source),
            self.extract_jsdoc(node, source),
            parent,
        )
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
}

impl ChunkExtractor for JavaScriptExtractor {
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
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let Some(lang) = get_language("javascript") else {
            return;
        };
        let source = r#"
/**
 * Say hello.
 */
function hello() {
    console.log("Hello!");
}
"#;
        let path = PathBuf::from("test.js");
        let extractor = JavaScriptExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "hello");
    }

    #[test]
    fn test_extract_class() {
        let Some(lang) = get_language("javascript") else {
            return;
        };
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
        let extractor = JavaScriptExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let class_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Class);
        assert!(class_chunk.is_some());
        assert_eq!(class_chunk.unwrap().symbol_name, "Person");
    }

    #[test]
    fn test_extract_arrow_function() {
        let Some(lang) = get_language("javascript") else {
            return;
        };
        let source = r#"
const greet = async (name) => {
    return `Hello, ${name}!`;
};
"#;
        let path = PathBuf::from("test.js");
        let extractor = JavaScriptExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
    }
}
