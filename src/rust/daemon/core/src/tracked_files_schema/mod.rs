//! Tracked Files and Qdrant Chunks Schema Definitions
//!
//! This module defines the types and schema for the `tracked_files` and `qdrant_chunks`
//! tables. Together they form the authoritative file inventory, replacing the need to
//! scroll Qdrant for file listings, recovery, and cleanup operations.
//!
//! Per docs/specs/04-write-path.md spec:
//! - `tracked_files` is written by the daemon, read by CLI
//! - `qdrant_chunks` is daemon-only (write and read)
//! - `qdrant_chunks` is a child of `tracked_files` with CASCADE delete

mod operations;
mod reconcile;
pub mod schema;
mod transactions;
pub mod types;

// Re-export types
pub use types::{ChunkType, ProcessingStatus, QdrantChunk, TrackedFile};

// Re-export schema constants
pub use schema::{
    CREATE_BASE_POINT_INDEX_SQL, CREATE_QDRANT_CHUNKS_INDEXES_SQL, CREATE_QDRANT_CHUNKS_SQL,
    CREATE_RECONCILE_INDEX_SQL, CREATE_REFCOUNT_INDEX_SQL, CREATE_TRACKED_FILES_INDEXES_SQL,
    CREATE_TRACKED_FILES_SQL, MIGRATE_V19_ADD_COLUMNS_SQL, MIGRATE_V28_ADD_COMPONENT_SQL,
    MIGRATE_V3_SQL, MIGRATE_V6_SQL, MIGRATE_V8_ADD_COLUMNS_SQL,
};

// Re-export pool-based operations
pub use operations::{
    compute_content_hash, compute_file_hash, compute_relative_path, delete_qdrant_chunks,
    delete_tracked_file, get_chunk_point_ids, get_file_mtime, get_tracked_file_paths,
    get_tracked_files_by_prefix, insert_qdrant_chunks, insert_tracked_file, is_incremental,
    lookup_tracked_file, lookup_watch_folder, set_incremental, update_tracked_file,
};

// Re-export transaction-aware operations
pub use transactions::{
    delete_qdrant_chunks_tx, delete_tracked_file_tx, insert_qdrant_chunks_tx,
    insert_tracked_file_tx, update_tracked_file_tx,
};

// Re-export reconcile operations
pub use reconcile::{clear_reconcile_flag_tx, get_files_needing_reconcile, mark_needs_reconcile};

#[cfg(test)]
mod tests;
