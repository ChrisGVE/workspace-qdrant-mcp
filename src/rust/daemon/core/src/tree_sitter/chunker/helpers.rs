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
                    // Reduce the callee expression to its bare function name.
                    // Generic/turbofish arguments are stripped first, so a call
                    // like `foo::<String, _>()` yields `foo` rather than the
                    // type-argument fragments `<String` / `_>`.
                    if let Some(clean_name) = clean_callee_name(name) {
                        if !calls.contains(&clean_name) {
                            calls.push(clean_name);
                        }
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

/// Reduce a callee expression to its bare function name.
///
/// Strips balanced generic/turbofish argument lists (`foo::<T>` → `foo`,
/// `Vec::<u8>::new` → `new`) and qualifier paths (`a::b::c` → `c`,
/// `obj.method` → `method`). Returns `None` when nothing identifier-like
/// remains. Stripping generics here keeps type-argument fragments such as
/// `<String` or `_>` (and the comma between them) out of the call list at the
/// source, rather than relying on a downstream filter to discard them.
fn clean_callee_name(name: &str) -> Option<String> {
    let stripped = strip_generic_args(name);
    let after_colons = stripped.rsplit("::").find(|s| !s.is_empty()).unwrap_or("");
    let base = after_colons
        .rsplit('.')
        .find(|s| !s.is_empty())
        .unwrap_or("")
        .trim();
    if base.is_empty() {
        None
    } else {
        Some(base.to_string())
    }
}

/// Remove balanced `<...>` generic/turbofish sections from a callee expression.
///
/// Characters inside angle brackets (including the commas that separate type
/// arguments) are dropped; everything at bracket depth zero is kept. Nesting is
/// handled so `Foo<Bar<Baz>>::method` collapses cleanly.
fn strip_generic_args(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth: u32 = 0;
    for ch in s.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{clean_callee_name, strip_generic_args};

    #[test]
    fn strip_generics_removes_balanced_sections() {
        assert_eq!(strip_generic_args("foo::<String, _>"), "foo::");
        assert_eq!(strip_generic_args("Vec::<u8>::new"), "Vec::::new");
        assert_eq!(strip_generic_args("Foo<Bar<Baz>>::method"), "Foo::method");
        assert_eq!(strip_generic_args("plain"), "plain");
    }

    #[test]
    fn clean_callee_strips_turbofish() {
        // The turbofish must not leak `<String` / `_>` into the call list.
        assert_eq!(clean_callee_name("foo::<String, _>").as_deref(), Some("foo"));
        assert_eq!(clean_callee_name("query::<String, _>").as_deref(), Some("query"));
    }

    #[test]
    fn clean_callee_keeps_last_segment() {
        assert_eq!(clean_callee_name("println").as_deref(), Some("println"));
        assert_eq!(
            clean_callee_name("std::collections::HashMap::new").as_deref(),
            Some("new")
        );
        assert_eq!(clean_callee_name("Vec::<u8>::new").as_deref(), Some("new"));
        assert_eq!(clean_callee_name("self.process").as_deref(), Some("process"));
        assert_eq!(clean_callee_name("obj.method::<T>").as_deref(), Some("method"));
    }

    #[test]
    fn clean_callee_rejects_pure_generic() {
        // A callee that is nothing but a (mangled) generic list has no name.
        assert_eq!(clean_callee_name("<String, _>"), None);
        assert_eq!(clean_callee_name(""), None);
    }
}
