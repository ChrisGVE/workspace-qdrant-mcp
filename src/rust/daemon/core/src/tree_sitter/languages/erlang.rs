//! Erlang language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Erlang source code.
pub struct ErlangExtractor {
    language: Option<Language>,
}

impl ErlangExtractor {
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
            Some(lang) => TreeSitterParser::with_language("erlang", lang.clone()),
            None => TreeSitterParser::new("erlang"),
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
                "module_attribute" | "attribute" => {
                    let text = node_text(&child, source);
                    if text.starts_with("-module")
                        || text.starts_with("-export")
                        || text.starts_with("-import")
                        || text.starts_with("-include")
                        || text.starts_with("-define")
                        || text.starts_with("-record")
                        || text.starts_with("-behaviour")
                        || text.starts_with("-behavior")
                    {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    } else if !preamble_items.is_empty() {
                        break;
                    }
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
            "erlang",
            file_path,
        ))
    }

    fn extract_function(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        // Function name is the atom at the beginning
        let name = find_child_by_kind(node, "atom")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches(" ->").trim().to_string());

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Function,
            name,
            content,
            start_line,
            end_line,
            "erlang",
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

    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut lines = Vec::new();

        while let Some(sibling) = prev {
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                // Erlang doc comments: %% @doc or just %%
                let cleaned = text
                    .trim_start_matches('%')
                    .trim_start_matches('%')
                    .trim()
                    .to_string();
                lines.push(cleaned);
                prev = sibling.prev_sibling();
            } else if sibling.kind() == "attribute" || sibling.kind() == "module_attribute" {
                let text = node_text(&sibling, source);
                if text.starts_with("-spec") {
                    lines.push(text.to_string());
                    prev = sibling.prev_sibling();
                } else {
                    break;
                }
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

impl ChunkExtractor for ErlangExtractor {
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
            // Erlang functions are `function_clause` nodes, or grouped as `function`
            match child.kind() {
                "function" | "function_clause" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str));
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "erlang"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let Some(lang) = get_language("erlang") else {
            return;
        };
        let source = r#"-module(greeter).
-export([hello/1]).

%% Greet a person
hello(Name) ->
    io:format("Hello, ~s!~n", [Name]).
"#;
        let path = PathBuf::from("greeter.erl");
        let extractor = ErlangExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
    }
}
