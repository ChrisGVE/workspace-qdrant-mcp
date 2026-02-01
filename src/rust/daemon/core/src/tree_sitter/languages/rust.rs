//! Rust language chunk extractor.

use std::path::Path;

use tree_sitter::Node;

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Rust source code.
pub struct RustExtractor;

impl RustExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract preamble (use statements, extern crates, mod declarations at top).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
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
    fn extract_function(
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
                .any(|c| c.kind() == "function_modifiers" && node_text(&c, source).contains("async"));

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
    fn extract_struct(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
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
    fn extract_trait(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
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

    /// Extract an impl block.
    fn extract_impl(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
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
                    chunks.push(self.extract_function(&child, source, file_path, Some(&parent_name)));
                }
            }
        }

        chunks
    }

    /// Extract an enum definition.
    fn extract_enum(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
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
    fn extract_macro(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
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
    fn extract_const(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
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
    fn extract_type_alias(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
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

    /// Extract preceding doc comments (/// or /** */).
    fn extract_preceding_docstring(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut doc_lines = Vec::new();

        while let Some(sibling) = prev {
            match sibling.kind() {
                "line_comment" => {
                    let text = node_text(&sibling, source);
                    if text.starts_with("///") || text.starts_with("//!") {
                        doc_lines.push(text.trim_start_matches("///").trim_start_matches("//!").trim().to_string());
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

impl ChunkExtractor for RustExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let mut parser = TreeSitterParser::new("rust")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let source = r#"
/// A simple function.
fn hello() {
    println!("Hello!");
}
"#;
        let path = PathBuf::from("test.rs");
        let extractor = RustExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        assert!(!chunks.is_empty());
        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        let fn_chunk = fn_chunk.unwrap();
        assert_eq!(fn_chunk.symbol_name, "hello");
        assert!(fn_chunk.docstring.is_some());
        assert!(fn_chunk.docstring.as_ref().unwrap().contains("simple function"));
    }

    #[test]
    fn test_extract_struct() {
        let source = r#"
/// A person struct.
pub struct Person {
    name: String,
    age: u32,
}
"#;
        let path = PathBuf::from("test.rs");
        let extractor = RustExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let struct_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Struct);
        assert!(struct_chunk.is_some());
        let struct_chunk = struct_chunk.unwrap();
        assert_eq!(struct_chunk.symbol_name, "Person");
    }

    #[test]
    fn test_extract_impl() {
        let source = r#"
impl Person {
    fn new(name: String) -> Self {
        Self { name, age: 0 }
    }

    fn greet(&self) {
        println!("Hello, {}!", self.name);
    }
}
"#;
        let path = PathBuf::from("test.rs");
        let extractor = RustExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        // Should have impl block + methods
        assert!(chunks.len() >= 3);
        let impl_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Impl);
        assert!(impl_chunk.is_some());

        let methods: Vec<_> = chunks.iter().filter(|c| c.chunk_type == ChunkType::Method).collect();
        assert_eq!(methods.len(), 2);
    }

    #[test]
    fn test_extract_trait() {
        let source = r#"
/// A greeter trait.
pub trait Greeter {
    fn greet(&self);
}
"#;
        let path = PathBuf::from("test.rs");
        let extractor = RustExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let trait_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Trait);
        assert!(trait_chunk.is_some());
        assert_eq!(trait_chunk.unwrap().symbol_name, "Greeter");
    }

    #[test]
    fn test_extract_preamble() {
        let source = r#"
use std::collections::HashMap;
use std::io::Result;

mod utils;

fn main() {}
"#;
        let path = PathBuf::from("test.rs");
        let extractor = RustExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let preamble = chunks.iter().find(|c| c.chunk_type == ChunkType::Preamble);
        assert!(preamble.is_some());
        let preamble = preamble.unwrap();
        assert!(preamble.content.contains("use std::collections::HashMap"));
        assert!(preamble.content.contains("mod utils"));
    }

    #[test]
    fn test_extract_async_function() {
        let source = r#"
async fn fetch_data() -> Result<String> {
    Ok("data".to_string())
}
"#;
        let path = PathBuf::from("test.rs");
        let extractor = RustExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let async_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::AsyncFunction);
        assert!(async_chunk.is_some());
        assert_eq!(async_chunk.unwrap().symbol_name, "fetch_data");
    }

    #[test]
    fn test_extract_function_calls() {
        let source = r#"
fn process() {
    helper();
    validate(data);
    transform();
}
"#;
        let path = PathBuf::from("test.rs");
        let extractor = RustExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        let calls = &fn_chunk.unwrap().calls;
        assert!(calls.contains(&"helper".to_string()));
        assert!(calls.contains(&"validate".to_string()));
        assert!(calls.contains(&"transform".to_string()));
    }
}
