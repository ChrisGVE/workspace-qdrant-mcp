//! Branch-lineage ingest chokepoint — the single tagging entry point (F6).
//!
//! File: src/rust/daemon/core/src/branch_index/mod.rs
//! Location: branch_index/ (sibling of strategies/, storage/, tracked_files_schema/).
//! Context: the branch-lineage indexing subsystem
//! (docs/architecture/branch-lineage-indexing.md §5.1). This module is the
//! facade for the BranchTagger: all ADD/ingest surfaces converge on
//! `tagger::tag_and_store`, which resolves `file_identity_id`, computes the
//! `content_key`, runs the three-case dedup ladder (Case-1 virtual point,
//! Case-2 copy-vector, Case-3 embed), and writes to state.db + Qdrant +
//! search.db atomically. No ingest surface writes `branches:[branch]` or calls
//! the embedding pipeline directly after this module ships.
//!
//! Sub-modules added progressively (task 22 ships mod.rs + tagger.rs):
//!   - `tagger`: `tag_and_store` + the three-case ladder (task 22).
//!   - `lineage`: `LineageStore` persist/read, CTE chain (task 25/26).
//!   - `resolve`: two-phase `resolve_view` (task 25/26).
//!   - `virtual_point`: virtual/tombstone payload builders (task 24).
//!   - `rekey`: Qdrant re-key batch pass (task 30).

pub mod tagger;

// ── Re-exported types consumed by ingest surfaces ──────────────────────────

pub use tagger::{IngestItem, TagOutcome};

// ── Public re-export of the entry point ────────────────────────────────────

pub use tagger::tag_and_store;
