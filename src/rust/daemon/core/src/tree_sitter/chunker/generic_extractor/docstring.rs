//! Docstring extraction strategies for the generic AST chunker.
//!
//! Each method corresponds to a `DocstringStyle` variant and extracts
//! documentation from a tree-sitter `Node` relative to `source`.

use tree_sitter::Node;

use crate::language_registry::types::{DocstringStyle, SemanticPatterns};
use crate::tree_sitter::chunker::helpers::{find_child_by_kind, node_text};

/// Dispatch docstring extraction based on the configured `DocstringStyle`.
pub(super) fn extract_docstring(
    patterns: &SemanticPatterns,
    node: &Node,
    source: &str,
) -> Option<String> {
    match patterns.docstring_style {
        DocstringStyle::FirstStringInBody => docstring_first_string(patterns, node, source),
        DocstringStyle::PrecedingComments => docstring_preceding_comments(patterns, node, source),
        DocstringStyle::Javadoc => docstring_javadoc(node, source),
        DocstringStyle::Haddock => docstring_haddock(patterns, node, source),
        DocstringStyle::ElixirAttr => docstring_elixir_attr(node, source),
        DocstringStyle::OcamlDoc => docstring_ocaml(node, source),
        DocstringStyle::Pod => docstring_pod(patterns, node, source),
        DocstringStyle::None => None,
    }
}

/// Python-style: first string expression in function/class body.
fn docstring_first_string(
    patterns: &SemanticPatterns,
    node: &Node,
    source: &str,
) -> Option<String> {
    let body_type = patterns.body_node.as_deref().unwrap_or("block");
    let body = find_child_by_kind(node, body_type)?;
    let mut cursor = body.walk();

    for child in body.children(&mut cursor) {
        if child.kind() == "expression_statement" {
            if let Some(string_node) = find_child_by_kind(&child, "string") {
                let text = node_text(&string_node, source);
                return Some(
                    text.trim_start_matches("\"\"\"")
                        .trim_start_matches("'''")
                        .trim_end_matches("\"\"\"")
                        .trim_end_matches("'''")
                        .trim()
                        .to_string(),
                );
            }
        }
        break;
    }
    None
}

/// C/C++/Rust/Go/Ruby style: comment nodes preceding the definition.
fn docstring_preceding_comments(
    patterns: &SemanticPatterns,
    node: &Node,
    source: &str,
) -> Option<String> {
    let comment_types = &patterns.comment_nodes;
    if comment_types.is_empty() {
        return None;
    }

    let mut comments = Vec::new();
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        if comment_types.iter().any(|t| t == sibling.kind()) {
            let text = node_text(&sibling, source);
            let is_doc = text.starts_with("///")
                || text.starts_with("//!")
                || text.starts_with("/**")
                || text.starts_with("##")
                || text.starts_with("-- |")
                || text.starts_with("---");
            if is_doc || comments.is_empty() {
                comments.push(text.to_string());
                prev = sibling.prev_sibling();
                continue;
            }
        }
        break;
    }

    if comments.is_empty() {
        return None;
    }

    comments.reverse();
    let joined = comments.join("\n");

    let cleaned: String = joined
        .lines()
        .map(|l| {
            l.trim()
                .trim_start_matches("///")
                .trim_start_matches("//!")
                .trim_start_matches("## ")
                .trim_start_matches("-- |")
                .trim_start_matches("-- ")
                .trim_start_matches("---")
                .trim_start()
        })
        .collect::<Vec<_>>()
        .join("\n");

    Some(cleaned.trim().to_string())
}

/// Java/JS/TS/Scala style: `/** ... */` block comment.
fn docstring_javadoc(node: &Node, source: &str) -> Option<String> {
    let prev = node.prev_sibling()?;
    let text = node_text(&prev, source);
    if text.starts_with("/**") {
        let cleaned = text
            .trim_start_matches("/**")
            .trim_end_matches("*/")
            .lines()
            .map(|l| l.trim().trim_start_matches("* ").trim_start_matches('*'))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();
        if !cleaned.is_empty() {
            return Some(cleaned);
        }
    }
    None
}

/// Haskell style: `-- |` Haddock comments.
fn docstring_haddock(patterns: &SemanticPatterns, node: &Node, source: &str) -> Option<String> {
    docstring_preceding_comments(patterns, node, source)
}

/// Elixir style: `@doc` attribute.
fn docstring_elixir_attr(node: &Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        let text = node_text(&sibling, source);
        if text.starts_with("@doc") {
            let doc = text
                .trim_start_matches("@doc")
                .trim()
                .trim_start_matches("\"\"\"")
                .trim_end_matches("\"\"\"")
                .trim()
                .to_string();
            return Some(doc);
        }
        if sibling.kind() == "comment" {
            prev = sibling.prev_sibling();
            continue;
        }
        break;
    }
    None
}

/// OCaml style: `(** ... *)` doc comments.
fn docstring_ocaml(node: &Node, source: &str) -> Option<String> {
    let prev = node.prev_sibling()?;
    if prev.kind() == "comment" {
        let text = node_text(&prev, source);
        if text.starts_with("(**") {
            let cleaned = text
                .trim_start_matches("(**")
                .trim_end_matches("*)")
                .trim()
                .to_string();
            return Some(cleaned);
        }
    }
    None
}

/// Perl POD style (simplified — extracts preceding comment block).
fn docstring_pod(patterns: &SemanticPatterns, node: &Node, source: &str) -> Option<String> {
    docstring_preceding_comments(patterns, node, source)
}
