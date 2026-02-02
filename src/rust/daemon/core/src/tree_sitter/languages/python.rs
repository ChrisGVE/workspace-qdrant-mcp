//! Python language chunk extractor.

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Python source code.
pub struct PythonExtractor {
    language: Option<Language>,
}

impl PythonExtractor {
    pub fn new() -> Self {
        Self { language: None }
    }

    /// Create an extractor with a pre-loaded Language.
    pub fn with_language(language: Language) -> Self {
        Self {
            language: Some(language),
        }
    }

    fn create_parser(&self) -> Result<TreeSitterParser, DaemonError> {
        match &self.language {
            Some(lang) => TreeSitterParser::with_language("python", lang.clone()),
            None => TreeSitterParser::new("python"),
        }
    }

    /// Extract preamble (imports, from imports, module docstring).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();
        let mut found_module_docstring = false;

        for child in root.children(&mut cursor) {
            match child.kind() {
                "import_statement" | "import_from_statement" | "future_import_statement" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
                "expression_statement" => {
                    // Check for module docstring (first string in file)
                    if !found_module_docstring {
                        if let Some(_string_node) = find_child_by_kind(&child, "string") {
                            preamble_items.push(node_text(&child, source).to_string());
                            last_preamble_line = child.end_position().row + 1;
                            found_module_docstring = true;
                            continue;
                        }
                    }
                    break;
                }
                "comment" => {
                    // Include top-level comments in preamble
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
            "python",
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

        // Determine if async
        let is_async = node.kind() == "async_function_definition"
            || content.trim_start().starts_with("async ");

        let chunk_type = if is_async {
            ChunkType::AsyncFunction
        } else if parent.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        // Determine symbol kind
        let symbol_kind = if is_async && parent.is_some() {
            "async_method"
        } else if is_async {
            "async_function"
        } else if parent.is_some() {
            if name.starts_with("__") && name.ends_with("__") {
                "dunder_method"
            } else if name.starts_with("_") {
                "private_method"
            } else {
                "method"
            }
        } else {
            "function"
        };

        // Extract signature (def line)
        let signature = content
            .lines()
            .next()
            .map(|l| l.trim_end_matches(':').to_string());

        // Extract docstring
        let docstring = self.extract_docstring(node, source);

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
            "python",
            file_path,
        )
        .with_symbol_kind(symbol_kind)
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

    /// Extract a class definition.
    fn extract_class(&self, node: &Node, source: &str, file_path: &str) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = find_child_by_kind(node, "identifier")
            .map(|n| node_text(&n, source).to_string())
            .unwrap_or_else(|| "anonymous".to_string());

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_docstring(node, source);

        let mut class_chunk = SemanticChunk::new(
            ChunkType::Class,
            &name,
            content,
            start_line,
            end_line,
            "python",
            file_path,
        );

        if let Some(doc) = docstring {
            class_chunk = class_chunk.with_docstring(doc);
        }

        chunks.push(class_chunk);

        // Extract methods
        if let Some(body) = find_child_by_kind(node, "block") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                match child.kind() {
                    "function_definition" | "async_function_definition" => {
                        chunks.push(self.extract_function(&child, source, file_path, Some(&name)));
                    }
                    "decorated_definition" => {
                        // Handle decorated methods
                        if let Some(func) = find_child_by_kind(&child, "function_definition")
                            .or_else(|| find_child_by_kind(&child, "async_function_definition"))
                        {
                            chunks.push(self.extract_function(&func, source, file_path, Some(&name)));
                        }
                    }
                    _ => {}
                }
            }
        }

        chunks
    }

    /// Extract docstring from a function or class.
    fn extract_docstring(&self, node: &Node, source: &str) -> Option<String> {
        // Look for docstring as first statement in body
        let body = find_child_by_kind(node, "block")?;
        let mut cursor = body.walk();

        for child in body.children(&mut cursor) {
            if child.kind() == "expression_statement" {
                if let Some(string_node) = find_child_by_kind(&child, "string") {
                    let text = node_text(&string_node, source);
                    // Clean up docstring
                    let cleaned = text
                        .trim_start_matches("\"\"\"")
                        .trim_start_matches("'''")
                        .trim_end_matches("\"\"\"")
                        .trim_end_matches("'''")
                        .trim();
                    return Some(cleaned.to_string());
                }
            }
            // Stop if we hit a non-docstring statement
            break;
        }

        None
    }
}

impl ChunkExtractor for PythonExtractor {
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

        // Walk the AST
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            match child.kind() {
                "function_definition" | "async_function_definition" => {
                    chunks.push(self.extract_function(&child, source, &file_path_str, None));
                }
                "class_definition" => {
                    chunks.extend(self.extract_class(&child, source, &file_path_str));
                }
                "decorated_definition" => {
                    // Handle decorated functions/classes
                    if let Some(func) = find_child_by_kind(&child, "function_definition")
                        .or_else(|| find_child_by_kind(&child, "async_function_definition"))
                    {
                        chunks.push(self.extract_function(&func, source, &file_path_str, None));
                    } else if let Some(class) = find_child_by_kind(&child, "class_definition") {
                        chunks.extend(self.extract_class(&class, source, &file_path_str));
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "python"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let source = r#"
def hello():
    """Say hello."""
    print("Hello!")
"#;
        let path = PathBuf::from("test.py");
        let extractor = PythonExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        let fn_chunk = fn_chunk.unwrap();
        assert_eq!(fn_chunk.symbol_name, "hello");
        assert!(fn_chunk.docstring.is_some());
        assert!(fn_chunk.docstring.as_ref().unwrap().contains("Say hello"));
    }

    #[test]
    fn test_extract_class() {
        let source = r#"
class Person:
    """A person class."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        """Greet someone."""
        print(f"Hello, {self.name}!")
"#;
        let path = PathBuf::from("test.py");
        let extractor = PythonExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let class_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Class);
        assert!(class_chunk.is_some());
        assert_eq!(class_chunk.unwrap().symbol_name, "Person");

        let methods: Vec<_> = chunks.iter().filter(|c| c.chunk_type == ChunkType::Method).collect();
        assert_eq!(methods.len(), 2);
    }

    #[test]
    fn test_extract_async_function() {
        let source = r#"
async def fetch_data():
    """Fetch data asynchronously."""
    return await get_data()
"#;
        let path = PathBuf::from("test.py");
        let extractor = PythonExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let async_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::AsyncFunction);
        assert!(async_chunk.is_some());
        assert_eq!(async_chunk.unwrap().symbol_name, "fetch_data");
    }

    #[test]
    fn test_extract_preamble() {
        let source = r#"
"""Module docstring."""

import os
from typing import List

def main():
    pass
"#;
        let path = PathBuf::from("test.py");
        let extractor = PythonExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let preamble = chunks.iter().find(|c| c.chunk_type == ChunkType::Preamble);
        assert!(preamble.is_some());
        let preamble = preamble.unwrap();
        assert!(preamble.content.contains("import os"));
        assert!(preamble.content.contains("from typing"));
    }

    #[test]
    fn test_decorated_function() {
        let source = r#"
@decorator
def decorated_func():
    pass
"#;
        let path = PathBuf::from("test.py");
        let extractor = PythonExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "decorated_func");
    }
}
