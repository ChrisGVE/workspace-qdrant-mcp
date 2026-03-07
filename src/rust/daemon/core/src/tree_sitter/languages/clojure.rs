//! Clojure language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::node_text;
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Clojure source code.
pub struct ClojureExtractor {
    language: Option<Language>,
}

impl ClojureExtractor {
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
            Some(lang) => TreeSitterParser::with_language("clojure", lang.clone()),
            None => TreeSitterParser::new("clojure"),
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
                "comment" => {
                    if preamble_items.is_empty()
                        || child.start_position().row <= last_preamble_line + 1
                    {
                        preamble_items.push(node_text(&child, source).to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "list_lit" => {
                    let text = node_text(&child, source);
                    if text.starts_with("(ns ") || text.starts_with("(require ")
                        || text.starts_with("(import ") || text.starts_with("(use ")
                    {
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
            "clojure",
            file_path,
        ))
    }

    fn extract_def(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> Option<SemanticChunk> {
        let text = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        // Determine kind and name from first symbol
        let trimmed = text.trim_start_matches('(');
        let first_word = trimmed.split_whitespace().next()?;
        let name = trimmed.split_whitespace().nth(1)?;
        // Clean name of any trailing parens
        let name = name.trim_end_matches(')');

        let (chunk_type, symbol_kind) = match first_word {
            "defn" => (ChunkType::Function, "function"),
            "defn-" => (ChunkType::Function, "private_function"),
            "defmacro" => (ChunkType::Macro, "macro"),
            "defprotocol" => (ChunkType::Interface, "protocol"),
            "defrecord" => (ChunkType::Struct, "record"),
            "deftype" => (ChunkType::Struct, "type"),
            "defmulti" => (ChunkType::Function, "multimethod"),
            "defmethod" => (ChunkType::Method, "method"),
            "def" => (ChunkType::Constant, "constant"),
            _ => return None,
        };

        // Extract docstring (second element if it's a string)
        let docstring = self.extract_clojure_docstring(&text);

        let signature = text.lines().next().map(|l| l.trim().to_string());

        let mut chunk = SemanticChunk::new(
            chunk_type, name, text, start_line, end_line, "clojure", file_path,
        )
        .with_symbol_kind(symbol_kind);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        Some(chunk)
    }

    fn extract_clojure_docstring(&self, text: &str) -> Option<String> {
        // Clojure docstrings appear as the 3rd token: (defn name "docstring" ...)
        let trimmed = text.trim_start_matches('(');
        let parts: Vec<&str> = trimmed.splitn(4, ' ').collect();
        if parts.len() >= 3 {
            let maybe_doc = parts[2].trim();
            if maybe_doc.starts_with('"') {
                let doc = maybe_doc.trim_matches('"');
                if !doc.is_empty() {
                    return Some(doc.to_string());
                }
            }
        }
        None
    }

    fn is_def_form(text: &str) -> bool {
        let trimmed = text.trim_start_matches('(');
        let first = trimmed.split_whitespace().next().unwrap_or("");
        matches!(
            first,
            "defn" | "defn-" | "defmacro" | "defprotocol" | "defrecord"
                | "deftype" | "defmulti" | "defmethod" | "def"
        )
    }
}

impl ChunkExtractor for ClojureExtractor {
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
            if child.kind() == "list_lit" {
                let text = node_text(&child, source);
                if Self::is_def_form(&text) {
                    if let Some(chunk) = self.extract_def(&child, source, &file_path_str) {
                        chunks.push(chunk);
                    }
                }
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "clojure"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_defn() {
        let Some(lang) = get_language("clojure") else {
            return;
        };
        let source = r#"(ns bookshelf.core)

(defn find-by-author [books author]
  (filter #(= (:author %) author) books))
"#;
        let path = PathBuf::from("test.clj");
        let extractor = ClojureExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
    }
}
