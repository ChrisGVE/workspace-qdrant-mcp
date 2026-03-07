//! Fortran language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Fortran source code.
pub struct FortranExtractor {
    language: Option<Language>,
}

impl FortranExtractor {
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
            Some(lang) => TreeSitterParser::with_language("fortran", lang.clone()),
            None => TreeSitterParser::new("fortran"),
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
                "use_statement" | "include_statement" | "implicit_statement" => {
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
                "program" => {
                    // Include `program` header line in preamble
                    let text = node_text(&child, source);
                    let first_line = text.lines().next().unwrap_or("");
                    if first_line.to_lowercase().starts_with("program") {
                        preamble_items.push(first_line.to_string());
                        last_preamble_line = child.start_position().row + 1;
                    }
                    break;
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
            "fortran",
            file_path,
        ))
    }

    fn extract_subprogram(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "name")
            .or_else(|| find_child_by_kind(node, "identifier"))
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
        let chunk_type = if kind.contains("function") {
            ChunkType::Function
        } else {
            ChunkType::Function // subroutines are functions in our model
        };

        let symbol_kind = if kind.contains("subroutine") {
            "subroutine"
        } else {
            "function"
        };

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim().to_string());

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            chunk_type, name, content, start_line, end_line, "fortran", file_path,
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

    fn extract_module(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "name")
            .or_else(|| find_child_by_kind(node, "identifier"))
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut module_chunk = SemanticChunk::new(
            ChunkType::Module,
            &name,
            content,
            start_line,
            end_line,
            "fortran",
            file_path,
        );
        if let Some(doc) = docstring {
            module_chunk = module_chunk.with_docstring(doc);
        }
        chunks.push(module_chunk);

        // Walk children for contained subprograms
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "function" | "subroutine" | "function_statement" | "subroutine_statement" => {
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
                // Fortran comments start with ! or C (column 1)
                let cleaned = text.trim_start_matches('!').trim().to_string();
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

impl ChunkExtractor for FortranExtractor {
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
                "function" | "subroutine" | "function_statement" | "subroutine_statement" => {
                    chunks.push(self.extract_subprogram(&child, source, &file_path_str));
                }
                "module" => {
                    chunks.extend(self.extract_module(&child, source, &file_path_str));
                }
                "program" => {
                    // Extract functions within program
                    let mut inner = child.walk();
                    for grandchild in child.children(&mut inner) {
                        match grandchild.kind() {
                            "function" | "subroutine" => {
                                chunks.push(self.extract_subprogram(
                                    &grandchild,
                                    source,
                                    &file_path_str,
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "fortran"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_subroutine() {
        let Some(lang) = get_language("fortran") else {
            return;
        };
        let source = r#"program main
  implicit none
  call greet("World")
contains
  subroutine greet(name)
    character(len=*), intent(in) :: name
    print *, "Hello, ", name, "!"
  end subroutine
end program
"#;
        let path = PathBuf::from("test.f90");
        let extractor = FortranExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
    }
}
