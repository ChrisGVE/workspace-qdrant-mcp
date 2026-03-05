//! Data types for chunk embedding results and tracking records.

use crate::storage::DocumentPoint;
use crate::tracked_files_schema::{ChunkType as TrackedChunkType, ProcessingStatus};

/// Metadata for a single chunk tracked in SQLite `qdrant_chunks`.
pub struct ChunkRecord {
    pub point_id: String,
    pub chunk_index: i32,
    pub content_hash: String,
    pub chunk_type: Option<TrackedChunkType>,
    pub symbol_name: Option<String>,
    pub start_line: Option<i32>,
    pub end_line: Option<i32>,
}

/// Result of embedding all chunks for a file.
pub struct EmbedResult {
    pub points: Vec<DocumentPoint>,
    pub chunk_records: Vec<ChunkRecord>,
    pub lsp_status: ProcessingStatus,
    pub treesitter_status: ProcessingStatus,
}
