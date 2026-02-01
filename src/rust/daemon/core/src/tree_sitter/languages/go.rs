//! Go language chunk extractor.

use std::path::Path;

use tree_sitter::Node;

use crate::error::DaemonError;
use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// Extractor for Go source code.
pub struct GoExtractor;

impl GoExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract preamble (package, imports).
    fn extract_preamble(&self, root: &Node, source: &str, file_path: &str) -> Option<SemanticChunk> {
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                "package_clause" | "import_declaration" => {
                    preamble_items.push(node_text(&child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
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
            "go",
            file_path,
        ))
    }

    /// Extract a function.
    fn extract_function(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        receiver: Option<&str>,
    ) -> SemanticChunk {
        // For method_declaration, the name comes after the receiver parameter_list
        // So we need to find the identifier that's a direct child, not inside parameter_list
        let name = self.find_function_name(node, source).unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let chunk_type = if receiver.is_some() {
            ChunkType::Method
        } else {
            ChunkType::Function
        };

        // Extract signature (first line)
        let signature = content.lines().next().map(|l| l.trim_end_matches('{').trim().to_string());

        // Extract doc comment
        let docstring = self.extract_doc_comment(node, source);

        // Extract calls
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
            "go",
            file_path,
        )
        .with_calls(calls);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }
        if let Some(r) = receiver {
            chunk = chunk.with_parent(r);
        }

        chunk
    }

    /// Extract a struct.
    fn extract_struct(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Struct,
            name,
            content,
            start_line,
            end_line,
            "go",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract an interface.
    fn extract_interface(&self, node: &Node, source: &str, file_path: &str) -> SemanticChunk {
        let name = find_child_by_kind(node, "type_identifier")
            .map(|n| node_text(&n, source))
            .unwrap_or("anonymous");

        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = self.extract_doc_comment(node, source);

        let mut chunk = SemanticChunk::new(
            ChunkType::Interface,
            name,
            content,
            start_line,
            end_line,
            "go",
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunk
    }

    /// Extract Go doc comment.
    fn extract_doc_comment(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        let mut doc_lines = Vec::new();

        while let Some(sibling) = prev {
            if sibling.kind() == "comment" {
                let text = node_text(&sibling, source);
                doc_lines.push(text.trim_start_matches("//").trim().to_string());
                prev = sibling.prev_sibling();
            } else {
                break;
            }
        }

        if doc_lines.is_empty() {
            None
        } else {
            doc_lines.reverse();
            Some(doc_lines.join("\n"))
        }
    }

    /// Find the function/method name.
    /// For methods, the name comes after the receiver parameter_list.
    fn find_function_name<'a>(&self, node: &'a Node<'a>, source: &'a str) -> Option<&'a str> {
        let mut cursor = node.walk();
        let mut found_receiver = false;

        for child in node.children(&mut cursor) {
            // For method_declaration, the receiver comes in a parameter_list
            if child.kind() == "parameter_list" {
                if !found_receiver && node.kind() == "method_declaration" {
                    // This is the receiver parameter_list
                    found_receiver = true;
                    continue;
                }
            }
            // Look for identifier or field_identifier
            if child.kind() == "identifier" || child.kind() == "field_identifier" {
                // For function_declaration, identifier comes first
                // For method_declaration, identifier comes after receiver parameter_list
                if node.kind() == "function_declaration" || found_receiver {
                    return Some(node_text(&child, source));
                }
            }
        }
        None
    }

    /// Get receiver type from method declaration.
    fn get_receiver_type(&self, node: &Node, source: &str) -> Option<String> {
        let params = find_child_by_kind(node, "parameter_list")?;
        let mut cursor = params.walk();
        for child in params.children(&mut cursor) {
            if child.kind() == "parameter_declaration" {
                if let Some(type_node) = find_child_by_kind(&child, "type_identifier")
                    .or_else(|| find_child_by_kind(&child, "pointer_type"))
                {
                    let type_text = node_text(&type_node, source);
                    return Some(type_text.trim_start_matches('*').to_string());
                }
            }
        }
        None
    }
}

impl ChunkExtractor for GoExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let mut parser = TreeSitterParser::new("go")?;
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
                "method_declaration" => {
                    let receiver = self.get_receiver_type(&child, source);
                    chunks.push(self.extract_function(
                        &child,
                        source,
                        &file_path_str,
                        receiver.as_deref(),
                    ));
                }
                "type_declaration" => {
                    // Check for struct or interface
                    let mut type_cursor = child.walk();
                    for type_child in child.children(&mut type_cursor) {
                        if type_child.kind() == "type_spec" {
                            if find_child_by_kind(&type_child, "struct_type").is_some() {
                                chunks.push(self.extract_struct(&type_child, source, &file_path_str));
                            } else if find_child_by_kind(&type_child, "interface_type").is_some() {
                                chunks.push(self.extract_interface(&type_child, source, &file_path_str));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        "go"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_function() {
        let source = r#"
package main

// Hello prints a greeting.
func Hello() {
    fmt.Println("Hello!")
}
"#;
        let path = PathBuf::from("test.go");
        let extractor = GoExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(fn_chunk.is_some());
        assert_eq!(fn_chunk.unwrap().symbol_name, "Hello");
    }

    #[test]
    fn test_extract_method() {
        let source = r#"
package main

func (p *Person) Greet() {
    fmt.Println("Hello!")
}
"#;
        let path = PathBuf::from("test.go");
        let extractor = GoExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let method_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Method);
        assert!(method_chunk.is_some());
        let method = method_chunk.unwrap();
        assert_eq!(method.symbol_name, "Greet");
        assert_eq!(method.parent_symbol, Some("Person".to_string()));
    }

    #[test]
    fn test_extract_struct() {
        let source = r#"
package main

// Person represents a person.
type Person struct {
    Name string
    Age  int
}
"#;
        let path = PathBuf::from("test.go");
        let extractor = GoExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let struct_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Struct);
        assert!(struct_chunk.is_some());
        assert_eq!(struct_chunk.unwrap().symbol_name, "Person");
    }

    #[test]
    fn test_extract_interface() {
        let source = r#"
package main

type Greeter interface {
    Greet()
}
"#;
        let path = PathBuf::from("test.go");
        let extractor = GoExtractor::new();
        let chunks = extractor.extract_chunks(source, &path).unwrap();

        let iface_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Interface);
        assert!(iface_chunk.is_some());
        assert_eq!(iface_chunk.unwrap().symbol_name, "Greeter");
    }
}
