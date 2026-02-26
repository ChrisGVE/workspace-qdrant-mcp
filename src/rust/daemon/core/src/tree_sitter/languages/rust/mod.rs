//! Rust language chunk extractor.
//!
//! Supports both static and dynamically loaded tree-sitter grammars
//! for extracting semantic chunks from Rust source code.

mod extraction;
#[cfg(test)]
mod tests;

use std::path::Path;

use tree_sitter::Language;

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Rust source code.
///
/// Supports both static and dynamically loaded tree-sitter grammars.
pub struct RustExtractor {
    /// Optional pre-loaded language for dynamic grammar support.
    language: Option<Language>,
}

impl RustExtractor {
    /// Create a new extractor using the static Rust grammar.
    pub fn new() -> Self {
        Self { language: None }
    }

    /// Create an extractor with a pre-loaded Language.
    ///
    /// Use this when you have a dynamically loaded grammar from
    /// the GrammarManager.
    pub fn with_language(language: Language) -> Self {
        Self {
            language: Some(language),
        }
    }

    /// Create a parser, using the pre-loaded language if available.
    fn create_parser(&self) -> Result<TreeSitterParser, DaemonError> {
        match &self.language {
            Some(lang) => TreeSitterParser::with_language("rust", lang.clone()),
            None => TreeSitterParser::new("rust"),
        }
    }
}

impl ChunkExtractor for RustExtractor {
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

        // Extract preamble
        if let Some(preamble) = self.extract_preamble(&root, source, &file_path_str) {
            chunks.push(preamble);
        }

        // Walk the AST and extract top-level items
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            match child.kind() {
                "function_item" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str, None));
                }
                "struct_item" => {
                    chunks.push(self.extract_struct(&child, source, &file_path_str));
                }
                "trait_item" => {
                    chunks.push(self.extract_trait(&child, source, &file_path_str));
                }
                "impl_item" => {
                    chunks.extend(self.extract_impl(&child, source, &file_path_str));
                }
                "enum_item" => {
                    chunks.push(self.extract_enum(&child, source, &file_path_str));
                }
                "macro_definition" => {
                    chunks.push(self.extract_macro(&child, source, &file_path_str));
                }
                "const_item" | "static_item" => {
                    chunks.push(self.extract_const(&child, source, &file_path_str));
                }
                "type_item" => {
                    chunks.push(self.extract_type_alias(&child, source, &file_path_str));
                }
                "mod_item" => {
                    // For mod with body, extract as module
                    if find_child_by_kind(&child, "declaration_list").is_some() {
                        let name = find_child_by_kind(&child, "identifier")
                            .map(|n| node_text(&n, source))
                            .unwrap_or("anonymous");
                        let content = node_text(&child, source);
                        let start_line = child.start_position().row + 1;
                        let end_line = child.end_position().row + 1;

                        chunks.push(SemanticChunk::new(
                            ChunkType::Module,
                            name,
                            content,
                            start_line,
                            end_line,
                            "rust",
                            &file_path_str,
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "rust"
    }
}
