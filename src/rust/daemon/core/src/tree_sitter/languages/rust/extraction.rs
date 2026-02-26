//! Node extraction and symbol resolution for Rust source code.

use tree_sitter::Node;

use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::types::{ChunkType, SemanticChunk};

use super::RustExtractor;

impl RustExtractor {
    /// Extract preamble (use statements, extern crates, mod declarations at top).
    pub(crate) fn extract_preamble(
        &self,
        root: &Node,
        source: &str,
        file_path: &str,
    ) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        // Collect all preamble items (use, extern crate, mod without body, attributes)
        for child in root.children(&mut cursor) {
            match child.kind() {
                "use_declaration" | "extern_crate_declaration" | "attribute_item"
                | "inner_attribute_item" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "mod_item" => {
                    // Only include if it's a declaration without body (e.g., `mod foo;`)
                    if find_child_by_kind(&child, "declaration_list").is_none() {
                        preamble_items.push(node_text(&child, source).to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "line_comment" | "block_comment" => {
                    // Include top-level comments in preamble
                    if last_preamble_line == 0 || child.start_position().row <= last_preamble_line {
                        preamble_items.push(node_text(&child, source).to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                _ => {
                    // Stop at first non-preamble item
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
            "rust",
            file_path,
        ))
    }

    /// Extract a function definition.
    pub(crate) fn extract_function(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent: Option<&str>,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        // Determine if async - tree-sitter-rust represents async in function_modifiers
        // or directly as a modifier child. We check both the text and any modifier nodes.
        let is_async = node_text(node, source).trim_start().starts_with("async ")
            || node
                .children(&mut node.walk())
                .any(|c| {
                    c.kind() == "function_modifiers"
                        && node_text(&c, source).contains("async")
                });

        let chunk_type = if is_async {
            ChunkType::AsyncFunction
        } else if parent.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        // Extract signature (everything before the body)
        let signature = if let Some(body) = find_child_by_kind(node, "block") {
            let sig_end = body.start_byte();
            let sig = &source[node.start_byte()..sig_end].trim();
            Some(sig.to_string())
        } else {
            None
        };

        // Extract docstring (preceding comments)
        let docstring = self.extract_preceding_docstring(node, source);

        // Extract function calls
        let calls = if let Some(body) = find_child_by_kind(node, "block") {
            extract_function_calls(&body, source)
        } else {
            Vec::new()
        };

        let mut chunk = SemanticChunk::new(
            chunk_type,
            name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        )
        .with_calls(calls);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }
        if let Some(p) = parent {
            chunk = chunk.with_parent(p);
        }

        chunk
    }

    /// Extract a struct definition.
    pub(crate) fn extract_struct(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_preceding_docstring(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Struct,
            name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract a trait definition.
    pub(crate) fn extract_trait(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_preceding_docstring(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Trait,
            name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract an impl block with its individual methods.
    pub(crate) fn extract_impl(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        // Get the type being implemented
        let impl_type = find_child_by_kind(node, "type_identifier")
            .or_else(|| find_child_by_kind(node, "generic_type"))
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        // Check if it's a trait impl
        let trait_name = node
            .children(&mut node.walk())
            .find(|c| c.kind() == "type_identifier")
            .and_then(|_| {
                // Look for "for" keyword to distinguish trait impl
                if node_text(node, source).contains(" for ") {
                    find_child_by_kind(node, "type_identifier")
                        .map(|n| node_text(&n, source).to_string())
                } else {
                    None
                }
            });

        let parent_name = trait_name
            .as_ref()
            .map(|t| format!("{}::{}", t, impl_type))
            .unwrap_or_else(|| impl_type.clone());

        // Extract the impl block itself as a chunk
        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;
        let docstring = self.extract_preceding_docstring(node, source);

        let mut impl_chunk = SemanticChunk::new(
            ChunkType::Impl,
            &parent_name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        );

        if let Some(doc) = docstring {
            impl_chunk = impl_chunk.with_docstring(doc);
        }

        chunks.push(impl_chunk);

        // Also extract individual methods
        if let Some(body) = find_child_by_kind(node, "declaration_list") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "function_item" {
                    chunks.push(self.extract_function(
                        &child,
                        source,
                        file_path,
                        Some(&parent_name),
                    ));
                }
            }
        }

        chunks
    }

    /// Extract an enum definition.
    pub(crate) fn extract_enum(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_preceding_docstring(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Enum,
            name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract a macro definition.
    pub(crate) fn extract_macro(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_preceding_docstring(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Macro,
            name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract const or static item.
    pub(crate) fn extract_const(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_preceding_docstring(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Constant,
            name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract type alias.
    pub(crate) fn extract_type_alias(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_preceding_docstring(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::TypeAlias,
            name,
            content,
            start_line,
            end_line,
            "rust",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract preceding doc comments (`///` or `/** */`).
    pub(crate) fn extract_preceding_docstring(
        &self,
        node: &Node,
        source: &str,
    ) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut doc_lines = Vec::new();

        while let Some(sibling) = prev {
            match sibling.kind() {
                "line_comment" => {
                    let text = node_text(&sibling, source);
                    if text.starts_with("///") || text.starts_with("//!") {
                        doc_lines.push(
                            text.trim_start_matches("///")
                                .trim_start_matches("//!")
                                .trim()
                                .to_string(),
                        );
                    } else {
                        break;
                    }
                }
                "block_comment" => {
                    let text = node_text(&sibling, source);
                    if text.starts_with("/**") || text.starts_with("/*!") {
                        let cleaned = text
                            .trim_start_matches("/**")
                            .trim_start_matches("/*!")
                            .trim_end_matches("*/")
                            .trim();
                        doc_lines.push(cleaned.to_string());
                    } else {
                        break;
                    }
                }
                "attribute_item" | "inner_attribute_item" => {
                    // Skip attributes, continue looking for docs
                }
                _ => break,
            }
            prev = sibling.prev_sibling();
        }

        if doc_lines.is_empty() {
            None
        } else {
            doc_lines.reverse();
            Some(doc_lines.join("\n"))
        }
    }
}
