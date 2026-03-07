//! Pascal language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Pascal source code.
pub struct PascalExtractor {
    language: Option<Language>,
}

impl PascalExtractor {
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
            Some(lang) => TreeSitterParser::with_language("pascal", lang.clone()),
            None => TreeSitterParser::new("pascal"),
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
                "uses_clause" | "unit_header" | "program_header" => {
                    preamble_items.push(text.to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "comment" | "line_comment" => {
                    if preamble_items.is_empty()
                        || child.start_position().row <= last_preamble_line + 1
                    {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                _ => {
                    // Also look for `program`, `unit`, `uses` keywords
                    let lower = text.to_lowercase();
                    if lower.starts_with("program ")
                        || lower.starts_with("unit ")
                        || lower.starts_with("uses ")
                    {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    } else if !preamble_items.is_empty() {
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
            "pascal",
            file_path,
        ))
    }

    fn extract_routine(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .or_else(|| find_child_by_kind(node, "name"))
            .map(|n| node_text(&n, source))
            .unwrap_or_else(|| {
                let text = node_text(node, source);
                text.split_whitespace()
                    .nth(1)
                    .and_then(|s| s.split('(').next())
                    .unwrap_or("anonymous")
            });

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let kind = node.kind();
        let symbol_kind = if kind.contains("function") {
            "function"
        } else {
            "procedure"
        };

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches(';').trim().to_string());

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Function,
            name,
            content,
            start_line,
            end_line,
            "pascal",
            file_path,
        )
        .with_symbol_kind(symbol_kind);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    fn extract_type_section(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "class_type" | "record_type" | "object_type" => {
                    let name = find_child_by_kind(&child, "identifier")
                        .map(|n| node_text(&n, source))
                        .unwrap_or("anonymous");

                    let content = node_text(&child, source);
                    let start_line = child.start_position().row + 1;
                    let end_line = child.end_position().row + 1;

                    let chunk_type = if child.kind() == "class_type" {
                        ChunkType::Class
                    } else {
                        ChunkType::Struct
                    };

                    chunks.push(SemanticChunk::new(
                        chunk_type,
                        name,
                        content,
                        start_line,
                        end_line,
                        "pascal",
                        file_path,
                    ));
                }
                _ => {}
            }
        }

        chunks
    }

    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut lines = Vec::new();

        while let Some(sibling) = prev {
            let kind = sibling.kind();
            if kind == "comment" || kind == "line_comment" {
                let text = node_text(&sibling, source);
                let cleaned = if text.starts_with("//") {
                    text.trim_start_matches("//").trim().to_string()
                } else if text.starts_with("{") {
                    text.trim_start_matches('{')
                        .trim_end_matches('}')
                        .trim()
                        .to_string()
                } else if text.starts_with("(*") {
                    text.trim_start_matches("(*")
                        .trim_end_matches("*)")
                        .trim()
                        .to_string()
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

impl ChunkExtractor for PascalExtractor {
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

        // Walk all nodes recursively since Pascal structure is nested
        self.walk_for_routines(&root, source, &file_path_str, &mut chunks);

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "pascal"
    }
}

impl PascalExtractor {
    fn walk_for_routines(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        chunks: &mut Vec<SemanticChunk>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "function_declaration" | "procedure_declaration" | "function_definition"
                | "procedure_definition" => {
                    chunks.push(self.extract_routine(&child, source, file_path));
                }
                "type_section" | "type_declaration" => {
                    chunks.extend(self.extract_type_section(&child, source, file_path));
                }
                _ => {
                    if child.child_count() > 0 {
                        self.walk_for_routines(&child, source, file_path, chunks);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_procedure() {
        let Some(lang) = get_language("pascal") else {
            return;
        };
        let source = r#"program Bookshelf;

procedure Greet(Name: string);
begin
  WriteLn('Hello, ', Name, '!');
end;

begin
  Greet('World');
end.
"#;
        let path = PathBuf::from("test.pas");
        let extractor = PascalExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
    }
}
