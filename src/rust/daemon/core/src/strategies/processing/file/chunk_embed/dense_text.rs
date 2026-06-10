//! Context-enriched text for dense embeddings.
//!
//! The dense vector for a chunk is computed over the raw chunk content plus a
//! compact context header (relative path, symbol breadcrumb, docstring excerpt)
//! so that file- and symbol-level vocabulary participates in semantic matching
//! — "where is RRF fusion implemented" should pull in `search-qdrant.ts` even
//! when the matching chunk body never spells out the file name.
//!
//! Only the DENSE embedding sees this text. The stored payload `content`, the
//! sparse/BM25 vector, and the chunk content hash all keep using the raw chunk,
//! so exact search, same-file dedup, and change detection are unaffected.

use std::collections::HashMap;

/// Hard cap on the docstring excerpt included in the header. Keeps the header
/// from crowding out code in the embedding model's sequence window.
const DOCSTRING_MAX_CHARS: usize = 240;

/// Build the text submitted to the dense embedding provider for one chunk.
///
/// Header shape (each part omitted when unavailable):
/// ```text
/// {relative_path} | {parent_symbol}::{symbol_name} ({chunk_type})
/// {docstring excerpt}
///
/// {raw chunk content}
/// ```
pub(super) fn build_dense_embedding_text(
    relative_path: &str,
    chunk_content: &str,
    chunk_metadata: &HashMap<String, String>,
) -> String {
    let mut header = String::new();

    if !relative_path.is_empty() {
        header.push_str(relative_path);
    }

    if let Some(symbol) = non_empty(chunk_metadata, "symbol_name") {
        if !header.is_empty() {
            header.push_str(" | ");
        }
        if let Some(parent) = non_empty(chunk_metadata, "parent_symbol") {
            header.push_str(parent);
            header.push_str("::");
        }
        header.push_str(symbol);
        if let Some(kind) = non_empty(chunk_metadata, "chunk_type") {
            header.push_str(" (");
            header.push_str(kind);
            header.push(')');
        }
    }

    if let Some(doc) = non_empty(chunk_metadata, "docstring") {
        let excerpt = truncate_chars(doc.trim(), DOCSTRING_MAX_CHARS);
        if !excerpt.is_empty() {
            if !header.is_empty() {
                header.push('\n');
            }
            header.push_str(excerpt);
        }
    }

    if header.is_empty() {
        return chunk_content.to_string();
    }
    format!("{header}\n\n{chunk_content}")
}

fn non_empty<'m>(metadata: &'m HashMap<String, String>, key: &str) -> Option<&'m str> {
    metadata.get(key).map(String::as_str).filter(|s| !s.is_empty())
}

/// Truncate to at most `max_chars` characters on a char boundary.
fn truncate_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn full_header_with_path_symbol_and_docstring() {
        let metadata = meta(&[
            ("symbol_name", "applyRRFFusion"),
            ("parent_symbol", "SearchQdrant"),
            ("chunk_type", "method"),
            ("docstring", "Fuse dense and sparse rankings."),
        ]);
        let text = build_dense_embedding_text(
            "src/tools/search-qdrant.ts",
            "function body",
            &metadata,
        );
        assert_eq!(
            text,
            "src/tools/search-qdrant.ts | SearchQdrant::applyRRFFusion (method)\n\
             Fuse dense and sparse rankings.\n\nfunction body"
        );
    }

    #[test]
    fn non_code_chunk_gets_path_only_header() {
        let text = build_dense_embedding_text("docs/specs/04-write-path.md", "Some prose.", &meta(&[]));
        assert_eq!(text, "docs/specs/04-write-path.md\n\nSome prose.");
    }

    #[test]
    fn empty_context_returns_raw_content() {
        let text = build_dense_embedding_text("", "raw chunk", &meta(&[]));
        assert_eq!(text, "raw chunk");
    }

    #[test]
    fn long_docstring_is_truncated() {
        let long_doc = "x".repeat(1000);
        let metadata = meta(&[("docstring", long_doc.as_str())]);
        let text = build_dense_embedding_text("a.rs", "body", &metadata);
        let header_line = text.lines().nth(1).unwrap();
        assert_eq!(header_line.chars().count(), 240);
    }

    #[test]
    fn raw_content_is_preserved_verbatim_after_header() {
        let metadata = meta(&[("symbol_name", "foo")]);
        let body = "fn foo() {\n    1 + 1\n}";
        let text = build_dense_embedding_text("a.rs", body, &metadata);
        assert!(text.ends_with(body));
    }
}
