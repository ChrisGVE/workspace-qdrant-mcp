//! Full-text search over `fts_content` (the sole FTS5 module, AC-F10.5).
//!
//! File: `wqm-storage/src/fts/mod.rs`
//! Location: `src/rust/storage/src/fts/` (read crate)
//! Context: arch §9 FP-2 / AC-F10.5 — exactly ONE FTS5 module in the read
//!   crate. `facade/read/fts.rs` must not exist; all FTS entry points route
//!   through `fts::search::fts_search`.
//!
//! Neighbors: `search.rs` (implementation + sanitizer), `crate::facade::read`
//!   (consumer).

pub mod search;

pub use search::{fts_search, sanitize_fts_query};
