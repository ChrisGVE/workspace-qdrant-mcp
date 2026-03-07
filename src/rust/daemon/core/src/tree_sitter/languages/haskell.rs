//! Haskell language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Haskell source code.
pub struct HaskellExtractor {
    language: Option<Language>,
}

impl HaskellExtractor {
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
            Some(lang) => TreeSitterParser::with_language("haskell", lang.clone()),
            None => TreeSitterParser::new("haskell"),
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
                "import" | "import_declaration" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "header" | "module" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "pragma" => {
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
            "haskell",
            file_path,
        ))
    }

    fn extract_function(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        // Function name is typically the first child (variable/identifier)
        let name = find_child_by_kind(node, "variable")
            .or_else(|| find_child_by_kind(node, "name"))
            .map(|n| node_text(&n, source))
            .unwrap_or_else(|| {
                // Try first word of content
                node_text(node, source)
                    .split_whitespace()
                    .next()
                    .unwrap_or("anonymous")
            });

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim().to_string());

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Function,
            name,
            content,
            start_line,
            end_line,
            "haskell",
            file_path,
        );

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    fn extract_type_decl(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "type")
            .or_else(|| find_child_by_kind(node, "name"))
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let chunk_type = match node.kind() {
            "type_alias" | "type_synomym" => ChunkType::TypeAlias,
            "newtype" | "adt" | "data_type" => ChunkType::Struct,
            "class" | "class_declaration" => ChunkType::Trait,
            _ => ChunkType::Struct,
        };

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            chunk_type, name, content, start_line, end_line, "haskell", file_path,
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
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                // Haddock comments: -- | or -- ^
                let cleaned = if text.starts_with("-- |") {
                    text.trim_start_matches("-- |").trim().to_string()
                } else if text.starts_with("-- ^") {
                    text.trim_start_matches("-- ^").trim().to_string()
                } else if text.starts_with("--") {
                    text.trim_start_matches("--").trim().to_string()
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

impl ChunkExtractor for HaskellExtractor {
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
                "function" | "bind" | "top_splice" | "signature" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str));
                }
                "data_type" | "newtype" | "adt" | "type_alias" | "type_synonym" => {
                    chunks.push(self.extract_type_decl(&child, source, &file_path_str));
                }
                "class" | "class_declaration" | "instance" | "instance_declaration" => {
                    chunks.push(self.extract_type_decl(&child, source, &file_path_str));
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "haskell"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let Some(lang) = get_language("haskell") else {
            return;
        };
        let source = r#"module Main where

-- | Greet a person by name
greet :: String -> String
greet name = "Hello, " ++ name ++ "!"
"#;
        let path = PathBuf::from("Main.hs");
        let extractor = HaskellExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        // Should have preamble and at least one function
        assert!(!chunks.is_empty());
    }
}
