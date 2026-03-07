//! Swift language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Swift source code.
pub struct SwiftExtractor {
    language: Option<Language>,
}

impl SwiftExtractor {
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
            Some(lang) => TreeSitterParser::with_language("swift", lang.clone()),
            None => TreeSitterParser::new("swift"),
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
                "import_declaration" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "comment" | "multiline_comment" => {
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
            "swift",
            file_path,
        ))
    }

    fn extract_function(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent: Option<&str>,
    ) -> SemanticChunk {
        let name = find_child_by_kind(node, "simple_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let chunk_type = if parent.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches('{').trim().to_string());

        let calls = if let Some(body) = find_child_by_kind(node, "function_body") {
            extract_function_calls(&body, source)
        } else {
            Vec::new()
        };

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            chunk_type, name, content, start_line, end_line, "swift", file_path,
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

    fn extract_type_decl(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
    ) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "type_identifier")
            .or_else(|| find_child_by_kind(node, "simple_identifier"))
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let chunk_type = match node.kind() {
            "class_declaration" => ChunkType::Class,
            "struct_declaration" => ChunkType::Struct,
            "protocol_declaration" => ChunkType::Interface,
            "enum_declaration" => ChunkType::Enum,
            _ => ChunkType::Class,
        };

        let docstring = self.extract_doc_comment(node, source);

        let mut type_chunk = SemanticChunk::new(
            chunk_type, &name, content, start_line, end_line, "swift", file_path,
        );
        if let Some(doc) = docstring {
            type_chunk = type_chunk.with_docstring(doc);
        }
        chunks.push(type_chunk);

        // Extract methods from class body
        if let Some(body) = find_child_by_kind(node, "class_body") {
            self.extract_methods_from_body(&body, source, file_path, &name, &mut chunks);
        }

        chunks
    }

    fn extract_methods_from_body(
        &self,
        body: &Node,
        source: &str,
        file_path: &str,
        parent_name: &str,
        chunks: &mut Vec<SemanticChunk>,
    ) {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            match child.kind() {
                "function_declaration" => {
                    chunks.push(self.extract_function(
                        &child,
                        source,
                        file_path,
                        Some(parent_name),
                    ));
                }
                "class_declaration" | "struct_declaration" | "enum_declaration"
                | "protocol_declaration" => {
                    chunks.extend(self.extract_type_decl(&child, source, file_path));
                }
                _ => {}
            }
        }
    }

    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut lines = Vec::new();

        while let Some(sibling) = prev {
            if sibling.kind() == "comment" || sibling.kind() == "multiline_comment" {
                let text = node_text(&sibling, source);
                if text.starts_with("///") {
                    lines.push(text.trim_start_matches("///").trim().to_string());
                } else if text.starts_with("/**") {
                    let cleaned = text
                        .trim_start_matches("/**")
                        .trim_end_matches("*/")
                        .lines()
                        .map(|l| l.trim().trim_start_matches('*').trim())
                        .collect::<Vec<_>>()
                        .join("\n")
                        .trim()
                        .to_string();
                    lines.push(cleaned);
                } else if text.starts_with("//") {
                    lines.push(text.trim_start_matches("//").trim().to_string());
                }
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

impl ChunkExtractor for SwiftExtractor {
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
                "function_declaration" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str, None));
                }
                "class_declaration" | "struct_declaration" | "enum_declaration"
                | "protocol_declaration" => {
                    chunks.extend(self.extract_type_decl(&child, source, &file_path_str));
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "swift"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let Some(lang) = get_language("swift") else {
            return;
        };
        let source = r#"
func greet(name: String) -> String {
    return "Hello, \(name)!"
}
"#;
        let path = PathBuf::from("test.swift");
        let extractor = SwiftExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "greet");
    }

    #[test]
    fn test_extract_class() {
        let Some(lang) = get_language("swift") else {
            return;
        };
        let source = r#"
class Person {
    var name: String

    init(name: String) {
        self.name = name
    }

    func greet() -> String {
        return "Hello, \(name)!"
    }
}
"#;
        let path = PathBuf::from("test.swift");
        let extractor = SwiftExtractor::with_language(lang);
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let class_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Class);
        assert!(class_chunk.is_some());
    }
}
