//! TypeScript language chunk extractor.

use std::path::Path;

use tree_sitter::Node;

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for TypeScript source code.
pub struct TypeScriptExtractor {
    is_tsx: bool,
}

impl TypeScriptExtractor {
    pub fn new(is_tsx: bool) -> Self {
        Self { is_tsx }
    }

    fn language_name(&self) -> &'static str {
        if self.is_tsx { "tsx" } else { "typescript" }
    }

    /// Extract preamble (imports, type imports).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "import_statement" | "export_statement" => {
                    let text = node_text(&child, source);
                    if text.contains("from") || child.kind() == "import_statement" {
                        preamble_items.push(text.to_string());
                        last_preamble_line = child.end_position().row + 1;
                    }
                }
                "comment" => {
                    if preamble_items.is_empty() || child.start_position().row <= last_preamble_line + 1 {
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
            self.language_name(),
            file_path,
        ))
    }

    /// Extract a function.
    fn extract_function(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent: Option<&str>,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "identifier")
            .or_else(|| find_child_by_kind(node, "property_identifier"))
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let is_async = content.trim_start().starts_with("async ");
        let chunk_type = if is_async {
            ChunkType::AsyncFunction
        } else if parent.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        let signature = content.lines().next().map(|l| l.trim_end_matches('{').trim().to_string());
        let docstring = self.extract_tsdoc(node, source);

        let calls = if let Some(body) = find_child_by_kind(node, "statement_block") {
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
            self.language_name(),
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

    /// Extract a class.
    fn extract_class(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "type_identifier")
            .or_else(|| find_child_by_kind(node, "identifier"))
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_tsdoc(node, source);

        let mut class_chunk = SemanticChunk::new(
            ChunkType::Class,
            &name,
            content,
            start_line,
            end_line,
            self.language_name(),
            file_path,
        );

        if let Some(doc) = docstring {
            class_chunk = class_chunk.with_docstring(doc);
        }

        chunks.push(class_chunk);

        // Extract methods
        if let Some(body) = find_child_by_kind(node, "class_body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                match child.kind() {
                    "method_definition" | "public_field_definition" => {
                        if find_child_by_kind(&child, "statement_block").is_some()
                            || find_child_by_kind(&child, "arrow_function").is_some()
                        {
                            chunks.push(self.extract_function(&child, source, file_path, Some(&name)));
                        }
                    }
                    _ => {}
                }
            }
        }

        chunks
    }

    /// Extract an interface.
    fn extract_interface(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_tsdoc(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Interface,
            name,
            content,
            start_line,
            end_line,
            self.language_name(),
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

        let docstring = self.extract_tsdoc(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::TypeAlias,
            name,
            content,
            start_line,
            end_line,
            self.language_name(),
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract TSDoc comment from the immediate previous sibling.
    fn extract_tsdoc(&self, node: &Node, source: &str) -> Option<String> {
        let prev = node.prev_sibling()?;
        if prev.kind() == "comment" {
            let text = node_text(&prev, source);
            if text.starts_with("/**") {
                let cleaned = text
                    .trim_start_matches("/**")
                    .trim_end_matches("*/")
                    .lines()
                    .map(|l| l.trim().trim_start_matches('*').trim())
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim()
                    .to_string();
                return Some(cleaned);
            }
        }
        None
    }
}

impl ChunkExtractor for TypeScriptExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let lang = if self.is_tsx { "tsx" } else { "typescript" };
        let mut parser = TreeSitterParser::new(lang)?;
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
                "function_declaration" | "generator_function_declaration" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str, None));
                }
                "class_declaration" => {
                    chunks.extend(self.extract_class(&child, source, &file_path_str));
                }
                "interface_declaration" => {
                    chunks.push(self.extract_interface(&child, source, &file_path_str));
                }
                "type_alias_declaration" => {
                    chunks.push(self.extract_type_alias(&child, source, &file_path_str));
                }
                "lexical_declaration" | "variable_declaration" => {
                    let mut decl_cursor = child.walk();
                    for decl_child in child.children(&mut decl_cursor) {
                        if decl_child.kind() == "variable_declarator" {
                            if let Some(value) = find_child_by_kind(&decl_child, "arrow_function")
                                .or_else(|| find_child_by_kind(&decl_child, "function"))
                            {
                                chunks.push(self.extract_function(&value, source, &file_path_str, None));
                            }
                        }
                    }
                }
                "export_statement" => {
                    if let Some(decl) = find_child_by_kind(&child, "function_declaration") {
                        chunks.push(self.extract_function(&decl, source, &file_path_str, None));
                    } else if let Some(decl) = find_child_by_kind(&child, "class_declaration") {
                        chunks.extend(self.extract_class(&decl, source, &file_path_str));
                    } else if let Some(decl) = find_child_by_kind(&child, "interface_declaration") {
                        chunks.push(self.extract_interface(&decl, source, &file_path_str));
                    } else if let Some(decl) = find_child_by_kind(&child, "type_alias_declaration") {
                        chunks.push(self.extract_type_alias(&decl, source, &file_path_str));
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        if self.is_tsx { "tsx" } else { "typescript" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let source = r#"
/**
 * Say hello.
 */
function hello(): void {
    console.log("Hello!");
}
"#;
        let path = PathBuf::from("test.ts");
        let extractor = TypeScriptExtractor::new(false);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "hello");
    }

    #[test]
    fn test_extract_interface() {
        let source = r#"
interface Person {
    name: string;
    age: number;
}
"#;
        let path = PathBuf::from("test.ts");
        let extractor = TypeScriptExtractor::new(false);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let iface_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Interface);
        assert!(iface_chunk.is_some());
        assert_eq!(iface_chunk.unwrap().symbol_name, "Person");
    }

    #[test]
    fn test_extract_type_alias() {
        let source = r#"
type ID = string | number;
"#;
        let path = PathBuf::from("test.ts");
        let extractor = TypeScriptExtractor::new(false);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let type_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::TypeAlias);
        assert!(type_chunk.is_some());
        assert_eq!(type_chunk.unwrap().symbol_name, "ID");
    }
}
