//! C language chunk extractor.

use std::path::Path;

use tree_sitter::Node;

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for C source code.
pub struct CExtractor;

impl CExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract preamble (includes, defines).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "preproc_include" | "preproc_def" | "preproc_ifdef" | "preproc_ifndef" => {
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
            "c",
            file_path,
        ))
    }

    /// Extract a function.
    fn extract_function(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        // First try direct identifier
        let name = if let Some(id) = find_child_by_kind(node, "identifier") {
            node_text(&id, source)
        } else if let Some(decl) = find_child_by_kind(node, "function_declarator") {
            // Look in declarator
            find_child_by_kind(&decl, "identifier")
                .map(|n| node_text(&n, source))
                .unwrap_or("anonymous")
        } else {
            "anonymous"
        };

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        // Extract signature (everything before body)
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
            ChunkType::Function,
            name,
            content,
            start_line,
            end_line,
            "c",
            file_path,
        )
        .with_calls(calls);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract a struct.
    fn extract_struct(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Struct,
            name,
            content,
            start_line,
            end_line,
            "c",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract an enum.
    fn extract_enum(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Enum,
            name,
            content,
            start_line,
            end_line,
            "c",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract doc comment (/* */ or // preceding).
    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut doc_lines = Vec::new();

        while let Some(sibling) = prev {
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                if text.starts_with("/*") {
                    let cleaned = text
                        .trim_start_matches("/*")
                        .trim_end_matches("*/")
                        .lines()
                        .map(|l| l.trim().trim_start_matches('*').trim())
                        .collect::<Vec<_>>()
                        .join("\n")
                        .trim()
                        .to_string();
                    doc_lines.push(cleaned);
                } else if text.starts_with("//") {
                    doc_lines.push(text.trim_start_matches("//").trim().to_string());
                }
                prev = sibling.prev_sibling();
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

impl ChunkExtractor for CExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let mut parser = TreeSitterParser::new("c")?;
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
                    chunks.push(self.extract_function(&child, source, &file_path_str));
                }
                "struct_specifier" | "type_definition" => {
                    // Check if it's a struct
                    if find_child_by_kind(&child, "struct_specifier").is_some()
                        || child.kind() == "struct_specifier"
                    {
                        chunks.push(self.extract_struct(&child, source, &file_path_str));
                    }
                }
                "enum_specifier" => {
                    chunks.push(self.extract_enum(&child, source, &file_path_str));
                }
                "declaration" => {
                    // Check for typedef struct
                    if let Some(struct_spec) = find_child_by_kind(&child, "struct_specifier") {
                        chunks.push(self.extract_struct(&struct_spec, source, &file_path_str));
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "c"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let source = r#"
#include <stdio.h>

/* Print hello */
void hello() {
    printf("Hello!\n");
}
"#;
        let path = PathBuf::from("test.c");
        let extractor = CExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "hello");
    }

    #[test]
    fn test_extract_struct() {
        let source = r#"
struct Person {
    char* name;
    int age;
};
"#;
        let path = PathBuf::from("test.c");
        let extractor = CExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let struct_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Struct);
        assert!(struct_chunk.is_some());
        assert_eq!(struct_chunk.unwrap().symbol_name, "Person");
    }

    #[test]
    fn test_extract_preamble() {
        let source = r#"
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 100

int main() { return 0; }
"#;
        let path = PathBuf::from("test.c");
        let extractor = CExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let preamble = chunks.iter().find(|c| c.chunk_type == ChunkType::Preamble);
        assert!(preamble.is_some());
        assert!(preamble.unwrap().content.contains("#include <stdio.h>"));
    }
}
