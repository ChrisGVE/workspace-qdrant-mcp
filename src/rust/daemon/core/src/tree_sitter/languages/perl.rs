//! Perl language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Perl source code.
pub struct PerlExtractor {
    language: Option<Language>,
}

impl PerlExtractor {
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
            Some(lang) => TreeSitterParser::with_language("perl", lang.clone()),
            None => TreeSitterParser::new("perl"),
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
                "use_no_statement" | "use_constant_statement" | "use_version" => {
                    preamble_items.push(text.to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "comment" => {
                    if preamble_items.is_empty()
                        || child.start_position().row <= last_preamble_line + 1
                    {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "expression_statement" => {
                    // `use strict;`, `use warnings;`, `use Module;`
                    if text.starts_with("use ") || text.starts_with("require ") {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    } else if !preamble_items.is_empty() {
                        break;
                    }
                }
                "package_statement" => {
                    preamble_items.push(text.to_string());
                    last_preamble_line = child.end_position().row + 1;
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
            "perl",
            file_path,
        ))
    }

    fn extract_subroutine(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
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

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches('{').trim().to_string());

        let calls = if let Some(body) = find_child_by_kind(node, "block") {
            extract_function_calls(&body, source)
        } else {
            Vec::new()
        };

        let docstring = self.extract_pod_doc(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Function,
            name,
            content,
            start_line,
            end_line,
            "perl",
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

    fn extract_pod_doc(&self, node: &Node, source: &str) -> Option<String> {
        // Look for preceding comments
        let mut prev = node.prev_sibling();
        let mut lines = Vec::new();

        while let Some(sibling) = prev {
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                lines.push(text.trim_start_matches('#').trim().to_string());
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

impl ChunkExtractor for PerlExtractor {
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
                "subroutine_declaration_statement" | "function_definition" => {
                    chunks.push(self.extract_subroutine(&child, source, &file_path_str));
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "perl"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_subroutine() {
        let Some(lang) = get_language("perl") else {
            return;
        };
        let source = r#"use strict;
use warnings;

# Greet a person
sub greet {
    my ($name) = @_;
    print "Hello, $name!\n";
}
"#;
        let path = PathBuf::from("test.pl");
        let extractor = PerlExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
    }
}
