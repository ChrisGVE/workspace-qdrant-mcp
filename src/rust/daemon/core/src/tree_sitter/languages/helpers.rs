//! Shared helpers for language-specific chunk extractors.
//!
//! These free functions capture patterns that appear identically across multiple
//! language extractors (TypeScript, JavaScript, Java, Go) to avoid duplication.

use tree_sitter::Node;

use crate::tree_sitter::chunker::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::types::{ChunkType, SemanticChunk};

/// Clean a `/** ... */` block-comment into plain text.
///
/// Strips the `/**` prefix, `*/` suffix, and leading `*` decorators from each
/// line.  Returns `None` when the node is not a block comment that starts with
/// `/**`.
pub(super) fn clean_block_doc_comment(node: &Node, source: &str) -> Option<String> {
    let text = node_text(node, source);
    if !text.starts_with("/**") {
        return None;
    }
    let cleaned = text
        .trim_start_matches("/**")
        .trim_end_matches("*/")
        .lines()
        .map(|l| l.trim().trim_start_matches('*').trim())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    Some(cleaned)
}

/// Collect consecutive preceding `//` line-comments as a doc string.
///
/// Walks backwards through previous siblings collecting every `comment` node
/// whose text starts with `//` (but not `///` — those belong to Rust docs).
/// The final text is joined in forward order.  Returns `None` when there are
/// no such comments immediately before `node`.
pub(super) fn collect_preceding_line_doc_comments(node: &Node, source: &str) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut lines: Vec<String> = Vec::new();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = node_text(&sibling, source);
            lines.push(text.trim_start_matches("//").trim().to_string());
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

/// Extract the first-line signature from node content.
///
/// Takes the first line of the node's source text and trims any trailing `{`
/// plus surrounding whitespace.  Returns `None` when the content is empty.
pub(super) fn first_line_signature(node: &Node, source: &str) -> Option<String> {
    let content = node_text(node, source);
    content
        .lines()
        .next()
        .map(|l| l.trim_end_matches('{').trim().to_string())
}

/// Build a `SemanticChunk` for a function/method node.
///
/// Extracts calls from `body_kind` child (e.g. `"statement_block"` for
/// JS/TS, `"block"` for Go/Java), assembles the chunk, and optionally
/// attaches `signature`, `docstring`, and `parent`.
pub(super) fn build_function_chunk(
    chunk_type: ChunkType,
    name: &str,
    node: &Node,
    source: &str,
    file_path: &str,
    language: &'static str,
    body_kind: &str,
    signature: Option<String>,
    docstring: Option<String>,
    parent: Option<&str>,
) -> SemanticChunk {
    let content = node_text(node, source);
    let start_line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;

    let calls = if let Some(body) = find_child_by_kind(node, body_kind) {
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
        language,
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
