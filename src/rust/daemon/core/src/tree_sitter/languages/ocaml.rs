//! OCaml language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for OCaml source code.
pub struct OCamlExtractor {
    language: Option<Language>,
}

impl OCamlExtractor {
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
            Some(lang) => TreeSitterParser::with_language("ocaml", lang.clone()),
            None => TreeSitterParser::new("ocaml"),
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
                "open_statement" | "open_module" => {
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
            "ocaml",
            file_path,
        ))
    }

    fn extract_let_binding(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        // Extract name from let binding
        let name = find_child_by_kind(node, "value_name")
            .or_else(|| find_child_by_kind(node, "value_pattern"))
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

        // Detect if it's a function (has parameters) or a value
        let is_function = content.contains("fun ") || {
            // Check if there are parameter patterns after the name
            let after_name = content
                .split(name)
                .nth(1)
                .unwrap_or("")
                .trim_start();
            after_name.starts_with('(') || after_name.starts_with("~")
                || (after_name.starts_with(|c: char| c.is_alphabetic()) && !after_name.starts_with('='))
        };

        let chunk_type = if is_function {
            ChunkType::Function
        } else {
            ChunkType::Constant
        };

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim().to_string());

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            chunk_type, name, content, start_line, end_line, "ocaml", file_path,
        );

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    fn extract_type_def(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_constructor")
            .or_else(|| find_child_by_kind(node, "type_variable"))
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

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::TypeAlias,
            name,
            content,
            start_line,
            end_line,
            "ocaml",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    fn extract_module_def(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "module_name")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Module,
            name,
            content,
            start_line,
            end_line,
            "ocaml",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let prev = node.prev_sibling();

        if let Some(sibling) = prev {
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                // OCaml doc comments: (** ... *)
                if text.starts_with("(**") {
                    let cleaned = text
                        .trim_start_matches("(**")
                        .trim_end_matches("*)")
                        .trim()
                        .to_string();
                    return Some(cleaned);
                }
            }
        }

        None
    }
}

impl ChunkExtractor for OCamlExtractor {
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
                "value_definition" | "let_binding" | "expression_item" => {
                    let text = node_text(&child, source);
                    if text.starts_with("let ") {
                        chunks.push(self.extract_let_binding(&child, source, &file_path_str));
                    }
                }
                "type_definition" => {
                    chunks.push(self.extract_type_def(&child, source, &file_path_str));
                }
                "module_definition" => {
                    chunks.push(self.extract_module_def(&child, source, &file_path_str));
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "ocaml"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_let_function() {
        let Some(lang) = get_language("ocaml") else {
            return;
        };
        let source = r#"
let greet name =
  Printf.printf "Hello, %s!\n" name
"#;
        let path = PathBuf::from("test.ml");
        let extractor = OCamlExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
    }
}
