//! SQLite table schema definitions.
//!
//! One module per table, each exporting a `TABLE` constant, individual
//! column constants, and an `ALL_COLUMNS` slice.

pub mod qdrant_chunks;
pub mod tracked_files;
pub mod unified_queue;
pub mod watch_folders;
