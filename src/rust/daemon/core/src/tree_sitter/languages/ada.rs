//! Ada language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Ada source code.
pub struct AdaExtractor {
    language: Option<Language>,
}

impl AdaExtractor {
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
            Some(lang) => TreeSitterParser::with_language("ada", lang.clone()),
            None => TreeSitterParser::new("ada"),
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
            match child.kind() {
                "with_clause" | "use_clause" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "comment" => {
                    if preamble_items.is_empty()
                        || child.start_position().row <= last_preamble_line + 1
                    {
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
            "ada",
            file_path,
        ))
    }

    fn extract_subprogram(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .or_else(|| find_child_by_kind(node, "name"))
            .map(|n| node_text(&n, source))
            .unwrap_or_else(|| {
                let text = node_text(node, source);
                text.split_whitespace()
                    .nth(1)
                    .unwrap_or("anonymous")
            });

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let kind = node.kind();
        let (chunk_type, symbol_kind) = if kind.contains("function") {
            (ChunkType::Function, "function")
        } else {
            (ChunkType::Function, "procedure")
        };

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches(" is").trim().to_string());

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            chunk_type, name, content, start_line, end_line, "ada", file_path,
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

    fn extract_package(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "identifier")
            .or_else(|| find_child_by_kind(node, "name"))
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut pkg_chunk = SemanticChunk::new(
            ChunkType::Module,
            &name,
            content,
            start_line,
            end_line,
            "ada",
            file_path,
        );
        if let Some(doc) = docstring {
            pkg_chunk = pkg_chunk.with_docstring(doc);
        }
        chunks.push(pkg_chunk);

        // Walk children for subprograms
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "subprogram_body" | "subprogram_declaration" | "procedure_specification"
                | "function_specification" => {
                    chunks.push(self.extract_subprogram(&child, source, file_path));
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
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                // Ada comments: --
                lines.push(text.trim_start_matches("--").trim().to_string());
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

impl ChunkExtractor for AdaExtractor {
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
                "subprogram_body" | "subprogram_declaration" => {
                    chunks.push(self.extract_subprogram(&child, source, &file_path_str));
                }
                "package_body" | "package_declaration" => {
                    chunks.extend(self.extract_package(&child, source, &file_path_str));
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "ada"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_procedure() {
        let Some(lang) = get_language("ada") else {
            return;
        };
        let source = r#"with Ada.Text_IO;

-- Greet a person
procedure Greet is
begin
   Ada.Text_IO.Put_Line("Hello!");
end Greet;
"#;
        let path = PathBuf::from("greet.adb");
        let extractor = AdaExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
    }
}
