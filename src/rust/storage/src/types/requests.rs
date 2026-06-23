//! Facade request DTOs — the inputs to `search`/`ingest_file` (§6).
//!
//! Location: `wqm-storage/src/types/requests.rs`. Logical context: the
//! caller-supplied side of the facade. [`SearchQuery`] carries a hybrid-search
//! request; [`IngestFileRequest`] carries one file's chunk batch for the dedup
//! ladder (§4.1).

use std::collections::HashMap;

/// A hybrid-search request (dense + sparse + RRF, §6.2).
///
/// `text` is the human query. When the caller has already embedded it, the
/// optional precomputed vectors are reused instead of re-embedding (the rebuild
/// and warm-cache paths supply them); otherwise the facade embeds `text`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchQuery {
    /// The raw query text.
    pub text: String,
    /// Optional precomputed dense embedding for `text`.
    pub dense_vector: Option<Vec<f32>>,
    /// Optional precomputed sparse embedding (term id -> weight) for `text`.
    pub sparse_vector: Option<HashMap<u32, f32>>,
    /// Optional minimum score; results below it are dropped.
    pub score_threshold: Option<f32>,
}

impl SearchQuery {
    /// A text-only query (the facade will embed it).
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }
}

/// One chunk of a file to ingest: its position, raw text, and content hash.
///
/// `content_hash` is the `chunk_content_hash` (hex `SHA256` of `text`) that feeds
/// the four-slot `content_key` and decides the dedup ladder (HIT vs MISS, §4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInput {
    /// Zero-based position of this chunk within the file.
    pub chunk_index: u32,
    /// The chunk's raw text (embedded on a `content_key` miss).
    pub text: String,
    /// Hex `SHA256` of `text` — the `chunk_content_hash`.
    pub content_hash: String,
}

/// A request to ingest one file's chunk batch for a `(branch_id, file)` (§4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestFileRequest {
    /// Branch-relative path of the file.
    pub path: String,
    /// Detected language, when known (drives chunking/embedding choices upstream).
    pub language: Option<String>,
    /// The file's chunks, in order.
    pub chunks: Vec<ChunkInput>,
}

impl IngestFileRequest {
    /// Construct a request for `path` with its ordered chunks.
    pub fn new(path: impl Into<String>, chunks: Vec<ChunkInput>) -> Self {
        Self {
            path: path.into(),
            language: None,
            chunks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_query_text_only_has_no_vectors() {
        let q = SearchQuery::new("hello");
        assert_eq!(q.text, "hello");
        assert!(q.dense_vector.is_none());
        assert!(q.sparse_vector.is_none());
        assert!(q.score_threshold.is_none());
    }

    #[test]
    fn ingest_request_keeps_chunk_order() {
        let req = IngestFileRequest::new(
            "src/main.rs",
            vec![
                ChunkInput {
                    chunk_index: 0,
                    text: "a".into(),
                    content_hash: "h0".into(),
                },
                ChunkInput {
                    chunk_index: 1,
                    text: "b".into(),
                    content_hash: "h1".into(),
                },
            ],
        );
        assert_eq!(req.path, "src/main.rs");
        assert_eq!(req.language, None);
        assert_eq!(req.chunks.len(), 2);
        assert_eq!(req.chunks[1].chunk_index, 1);
    }
}
