//! Common Lisp language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::node_text;
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Common Lisp source code.
pub struct LispExtractor {
    language: Option<Language>,
}

impl LispExtractor {
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
            Some(lang) => TreeSitterParser::with_language("commonlisp", lang.clone()),
            None => TreeSitterParser::new("commonlisp"),
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
                "comment" | "block_comment" => {
                    if preamble_items.is_empty()
                        || child.start_position().row <= last_preamble_line + 1
                    {
                        preamble_items.push(node_text(&child, source).to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "list_lit" | "defun_form" | "defpackage_form" | "in_package_form" => {
                    let text = node_text(&child, source);
                    if text.starts_with("(defpackage")
                        || text.starts_with("(in-package")
                        || text.starts_with("(require")
                        || text.starts_with("(use-package")
                        || text.starts_with("(ql:quickload")
                        || text.starts_with("(asdf:")
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
            "lisp",
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

        let trimmed = text.trim_start_matches('(');
        let first_word = trimmed.split_whitespace().next()?;
        let name = trimmed
            .split_whitespace()
            .nth(1)
            .map(|s| s.trim_end_matches(')'))
            .unwrap_or("anonymous");

        let (chunk_type, symbol_kind) = match first_word {
            "defun" => (ChunkType::Function, "function"),
            "defmacro" => (ChunkType::Macro, "macro"),
            "defgeneric" => (ChunkType::Function, "generic_function"),
            "defmethod" => (ChunkType::Method, "method"),
            "defclass" => (ChunkType::Class, "class"),
            "defstruct" => (ChunkType::Struct, "struct"),
            "defvar" | "defparameter" | "defconstant" => (ChunkType::Constant, "constant"),
            "deftype" => (ChunkType::TypeAlias, "type"),
            _ => return None,
        };

        // Extract docstring (CL puts docstring after parameter list)
        let docstring = self.extract_lisp_docstring(&text);

        let signature = text.lines().next().map(|l| l.trim().to_string());

        let mut chunk = SemanticChunk::new(
            chunk_type, name, text, start_line, end_line, "lisp", file_path,
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

    fn extract_lisp_docstring(&self, text: &str) -> Option<String> {
        // Common Lisp docstrings are the first string after the parameter list
        // (defun name (params) "docstring" body...)
        // Simple heuristic: find first standalone string literal after params
        let lines: Vec<&str> = text.lines().collect();
        for line in &lines[1..] {
            let trimmed = line.trim();
            if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() > 2 {
                return Some(trimmed.trim_matches('"').to_string());
            }
            if !trimmed.is_empty() && !trimmed.starts_with('"') {
                break;
            }
        }
        None
    }

    fn is_def_form(text: &str) -> bool {
        let trimmed = text.trim_start_matches('(');
        let first = trimmed.split_whitespace().next().unwrap_or("");
        matches!(
            first,
            "defun" | "defmacro" | "defgeneric" | "defmethod" | "defclass" | "defstruct"
                | "defvar" | "defparameter" | "defconstant" | "deftype"
        )
    }
}

impl ChunkExtractor for LispExtractor {
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
            let text = node_text(&child, source);
            match child.kind() {
                "list_lit" | "defun_form" | "defmacro_form" | "defgeneric_form"
                | "defmethod_form" | "defclass_form" | "defvar_form" | "defparameter_form" => {
                    if Self::is_def_form(&text) {
                        if let Some(chunk) = self.extract_def(&child, source, &file_path_str) {
                            chunks.push(chunk);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "lisp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_defun() {
        let Some(lang) = get_language("commonlisp") else {
            return;
        };
        let source = r#"(defpackage :bookshelf
  (:use :cl))

(in-package :bookshelf)

(defun greet (name)
  "Greet a person by name."
  (format t "Hello, ~a!~%" name))
"#;
        let path = PathBuf::from("test.lisp");
        let extractor = LispExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
    }
}
