//! Zig language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Zig source code.
pub struct ZigExtractor {
    language: Option<Language>,
}

impl ZigExtractor {
    pub fn new() -> Self {
        Self { language: None }
    }

    pub fn with_language(language: Language) -> Self {
        Self {
            language: Some(language),
        }
    }

    fn create_parser(&self) -> Result<TreeSitterParser, DaemonError> {
        match &self.language {
            Some(lang) => TreeSitterParser::with_language("zig", lang.clone()),
            None => TreeSitterParser::new("zig"),
        }
    }

    fn extract_preamble(
        &self,
        root: &Node,
        source: &str,
        file_path: &str,
    ) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            let text = node_text(&child, source);
            match child.kind() {
                "line_comment" | "doc_comment" => {
                    if preamble_items.is_empty()
                        || child.start_position().row <= last_preamble_line + 1
                    {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "VarDecl" | "TopLevelDecl" => {
                    // @import statements
                    if text.contains("@import") || text.contains("@cImport") {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    } else if !preamble_items.is_empty() {
                        break;
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
            "zig",
            file_path,
        ))
    }

    fn extract_function(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "IDENTIFIER")
            .or_else(|| find_child_by_kind(node, "identifier"))
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let is_pub = content.trim_start().starts_with("pub ");
        let symbol_kind = if is_pub { "public_function" } else { "function" };

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches('{').trim().to_string());

        let calls = if let Some(body) = find_child_by_kind(node, "Block") {
            extract_function_calls(&body, source)
        } else {
            Vec::new()
        };

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Function,
            name,
            content,
            start_line,
            end_line,
            "zig",
            file_path,
        )
        .with_symbol_kind(symbol_kind)
        .with_calls(calls);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    fn extract_container(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> SemanticChunk {
        // Zig structs/enums/unions are container declarations
        let name = find_child_by_kind(node, "IDENTIFIER")
            .or_else(|| find_child_by_kind(node, "identifier"))
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let chunk_type = if content.contains("enum") {
            ChunkType::Enum
        } else if content.contains("union") {
            ChunkType::Struct
        } else {
            ChunkType::Struct
        };

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            chunk_type, name, content, start_line, end_line, "zig", file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut lines = Vec::new();

        while let Some(sibling) = prev {
            let kind = sibling.kind();
            if kind == "doc_comment" || kind == "line_comment" {
                let text = node_text(&sibling, source);
                let cleaned = if text.starts_with("///") {
                    text.trim_start_matches("///").trim().to_string()
                } else if text.starts_with("//") {
                    text.trim_start_matches("//").trim().to_string()
                } else {
                    text.to_string()
                };
                lines.push(cleaned);
                prev = sibling.prev_sibling();
            } else {
                break;
            }
        }

        if lines.is_empty() {
            None
        } else {
            lines.reverse();
            Some(lines.join("\n"))
        }
    }
}

impl ChunkExtractor for ZigExtractor {
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

        // Walk all top-level declarations
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            let text = node_text(&child, source);
            match child.kind() {
                "FnProto" | "TopLevelDecl" => {
                    if text.contains("fn ") {
                        chunks.push(self.extract_function(&child, source, &file_path_str));
                    } else if text.contains("struct")
                        || text.contains("enum")
                        || text.contains("union")
                    {
                        chunks.push(self.extract_container(&child, source, &file_path_str));
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "zig"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let Some(lang) = get_language("zig") else {
            return;
        };
        let source = r#"const std = @import("std");

/// Greet a person
pub fn greet(name: []const u8) void {
    std.debug.print("Hello, {s}!\n", .{name});
}
"#;
        let path = PathBuf::from("test.zig");
        let extractor = ZigExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        // Should produce at least preamble + function chunks
        assert!(!chunks.is_empty());
    }
}
