//! Tree-sitter node helper functions.
//!
//! Utility functions for extracting text and navigating tree-sitter AST nodes.
//! Used by all language-specific extractors.

use tree_sitter::Node;

/// Helper to extract text from a node.
pub fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    let start = node.start_byte();
    let end = node.end_byte();
    &source[start..end]
}

/// Helper to find a child node by kind.
pub fn find_child_by_kind<'a>(node: &'a Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == kind {
            return Some(child);
        }
    }
    None
}

/// Helper to find all children of a specific kind.
pub fn find_children_by_kind<'a>(node: &'a Node<'a>, kind: &str) -> Vec<Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .filter(|child| child.kind() == kind)
        .collect()
}

/// Helper to extract function calls from a node.
pub fn extract_function_calls(node: &Node, source: &str) -> Vec<String> {
    let mut calls = Vec::new();
    let mut cursor = node.walk();

    fn visit(
        node: &Node,
        source: &str,
        calls: &mut Vec<String>,
        cursor: &mut tree_sitter::TreeCursor,
    ) {
        match node.kind() {
            "call_expression" | "function_call" | "invocation_expression" | "call" => {
                // Try to get the function name
                if let Some(callee) = node
                    .child_by_field_name("function")
                    .or_else(|| node.child_by_field_name("callee"))
                    .or_else(|| node.child(0))
                {
                    let name = node_text(&callee, source);
                    // Extract just the function name (not the full path)
                    let clean_name = name
                        .rsplit("::")
                        .next()
                        .unwrap_or(name)
                        .rsplit('.')
                        .next()
                        .unwrap_or(name);
                    if !clean_name.is_empty() && !calls.contains(&clean_name.to_string()) {
                        calls.push(clean_name.to_string());
                    }
                }
            }
            _ => {}
        }

        // Visit children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i as u32) {
                visit(&child, source, calls, cursor);
            }
        }
    }

    visit(node, source, &mut calls, &mut cursor);
    calls
}
