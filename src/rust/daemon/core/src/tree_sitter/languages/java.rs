//! Java language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Java source code.
pub struct JavaExtractor {
    language: Option<Language>,
}

impl JavaExtractor {
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
            Some(lang) => TreeSitterParser::with_language("java", lang.clone()),
            None => TreeSitterParser::new("java"),
        }
    }

    /// Extract preamble (package, imports).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "package_declaration" | "import_declaration" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "line_comment" | "block_comment" => {
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
            "java",
            file_path,
        ))
    }

    /// Extract a method.
    fn extract_method(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent: &str,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        // Extract signature
        let signature = content.lines().next().map(|l| l.trim_end_matches('{').trim().to_string());

        // Extract Javadoc
        let docstring = self.extract_javadoc(node, source);

        // Extract calls
        let calls = if let Some(body) = find_child_by_kind(node, "block") {
            extract_function_calls(&body, source)
        } else {
            Vec::new()
        };

        let mut chunk = SemanticChunk::new(
            ChunkType::Method,
            name,
            content,
            start_line,
            end_line,
            "java",
            file_path,
        )
        .with_parent(parent)
        .with_calls(calls);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
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

        let docstring = self.extract_javadoc(node, source);

        let mut class_chunk = SemanticChunk::new(
            ChunkType::Class,
            &name,
            content,
            start_line,
            end_line,
            "java",
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
                match child.kind() {
                    "method_declaration" | "constructor_declaration" => {
                        chunks.push(self.extract_method(&child, source, file_path, &name));
                    }
                    _ => {}
                }
            }
        }

        chunks
    }

    /// Extract an interface.
    fn extract_interface(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_javadoc(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Interface,
            name,
            content,
            start_line,
            end_line,
            "java",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract an enum.
    fn extract_enum(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_javadoc(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Enum,
            name,
            content,
            start_line,
            end_line,
            "java",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract Javadoc comment from the immediate previous sibling.
    fn extract_javadoc(&self, node: &Node, source: &str) -> Option<String> {
        let prev = node.prev_sibling()?;
        if prev.kind() == "block_comment" {
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

impl ChunkExtractor for JavaExtractor {
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
                "class_declaration" => {
                    chunks.extend(self.extract_class(&child, source, &file_path_str));
                }
                "interface_declaration" => {
                    chunks.push(self.extract_interface(&child, source, &file_path_str));
                }
                "enum_declaration" => {
                    chunks.push(self.extract_enum(&child, source, &file_path_str));
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "java"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_class() {
        let source = r#"
package com.example;

/**
 * A person class.
 */
public class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    public void greet() {
        System.out.println("Hello!");
    }
}
"#;
        let path = PathBuf::from("Person.java");
        let extractor = JavaExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let class_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Class);
        assert!(class_chunk.is_some());
        assert_eq!(class_chunk.unwrap().symbol_name, "Person");

        let methods: Vec<_> = chunks.iter().filter(|c| c.chunk_type == ChunkType::Method).collect();
        assert_eq!(methods.len(), 2); // constructor + greet
    }

    #[test]
    fn test_extract_interface() {
        let source = r#"
package com.example;

public interface Greeter {
    void greet();
}
"#;
        let path = PathBuf::from("Greeter.java");
        let extractor = JavaExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let iface_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Interface);
        assert!(iface_chunk.is_some());
        assert_eq!(iface_chunk.unwrap().symbol_name, "Greeter");
    }
}
