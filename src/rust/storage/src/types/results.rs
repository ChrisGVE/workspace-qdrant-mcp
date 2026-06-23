//! Facade result DTOs — FTS hits and file listings (§6.2).
//!
//! Location: `wqm-storage/src/types/results.rs`. Logical context: SQLite-derived
//! read results. (Hybrid vector-search results use the F0 `SearchResult` type,
//! re-exported from the parent module — not redefined here.) [`FtsResult`] is an
//! FTS5 hit; [`FileEntry`] is a row from a branch's file listing.

/// A full-text (FTS5) search hit, scoped to one branch (§6.2 `fts_search`).
#[derive(Debug, Clone, PartialEq)]
pub struct FtsResult {
    /// `files.file_id` of the matching file.
    pub file_id: i64,
    /// Branch-relative path of the matching file.
    pub path: String,
    /// `blobs.blob_id` of the matching chunk-blob.
    pub blob_id: i64,
    /// FTS5 relevance score (higher is better; sign follows the ranking function).
    pub score: f32,
    /// A text snippet around the match, when one was produced.
    pub snippet: Option<String>,
}

/// One file known to a branch (§6.2 `list_branch`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// `files.file_id` (minted per `(branch_id, path)`, §5.4).
    pub file_id: i64,
    /// Branch-relative path.
    pub path: String,
    /// Hex `SHA256` of the file's content at last ingest.
    pub content_hash: String,
    /// Number of chunk-blobs the file currently references.
    pub chunk_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_result_holds_its_fields() {
        let r = FtsResult {
            file_id: 7,
            path: "src/lib.rs".into(),
            blob_id: 42,
            score: 1.5,
            snippet: Some("fn main".into()),
        };
        assert_eq!(r.file_id, 7);
        assert_eq!(r.blob_id, 42);
        assert_eq!(r.snippet.as_deref(), Some("fn main"));
    }

    #[test]
    fn file_entry_holds_its_fields() {
        let e = FileEntry {
            file_id: 1,
            path: "README.md".into(),
            content_hash: "abc".into(),
            chunk_count: 3,
        };
        assert_eq!(e.path, "README.md");
        assert_eq!(e.chunk_count, 3);
    }
}
