//! Branch-tagging chokepoint (branch-lineage F6).
//!
//! File: `daemon/core/src/branch_index/mod.rs`
//! Context: the single chokepoint (arch §5.1, invariant P9) every file-ingest
//! path routes through before a Qdrant point is written. It owns the three-case
//! dedup ladder (virtual / copy-vector / embed), the `tracked_files` (state.db)
//! and `file_metadata` (search.db) writes, and the branch-lineage payload
//! fields. No caller writes branch information into a payload directly.
//!
//! This module is the FACADE: it defines the tagger's input/output types and
//! re-exports [`tag_and_store`]. The ladder itself lives in [`tagger`].
//!
//! ## Design (arch §5.1, Option C — locked 2026-06-21)
//! Case 3 (genuinely new content) reuses the production `embed_chunks` and then
//! RE-KEYS the resulting points to the content-key scheme — preserving LSP /
//! lexicon-sparse / oversize-splitting / the rich payload (no regression) and
//! keeping a single embed source (project FP-2, unify-to-prevent-drift). Cases
//! 1/2 build the same rich payload via `build_chunk_payload` WITHOUT embedding.

mod tagger;

pub(crate) use tagger::tag_and_store;

use std::collections::HashMap;
use std::path::Path;

use crate::core_types::{DocumentContent, TextChunk};
use crate::storage::DocumentPoint;
use crate::strategies::processing::file::chunk_embed::ChunkRecord;
use crate::unified_queue_schema::UnifiedQueueItem;

/// Input to [`tag_and_store`]: the branch-lineage identity + per-file metadata
/// for one file, constructed at the post-parse seam in `run_ingest_pipeline`
/// (after `parse_document`).
///
/// `file_hash` is the whole-file SHA-256 hex re-derived IN-PROCESS by
/// `parse_document` (SEC-6 — never the queue-supplied value). It is BOTH the
/// `content_key` third ingredient AND the Case-2 byte-identical locator input.
pub(crate) struct IngestItem<'a> {
    pub watch_folder_id: &'a str,
    /// TEXT in state.db (NOT i64).
    pub tenant_id: &'a str,
    pub branch: &'a str,
    pub collection: &'a str,
    /// Required by the §5.2 view layer.
    pub relative_path: &'a str,
    /// Absolute on-disk path; bound into search.db `file_metadata.file_path`
    /// (which holds an ABSOLUTE path — N5).
    pub abs_file_path: &'a str,
    /// Minted/inherited via `allocate_file_identity`.
    pub file_identity_id: uuid::Uuid,
    /// Whole-file SHA-256 hex (in-process; content_key ingredient + Case-2 locator).
    pub file_hash: &'a str,
    /// Content-bearing chunks (NOT `ChunkRecord`). Same slice as
    /// `EmbedInputs.document_content.chunks`.
    pub chunks: &'a [TextChunk],
    pub file_mtime: &'a str,
    pub file_type: Option<&'a str>,
    pub language: Option<&'a str>,
    pub is_test: bool,
    pub extension: Option<&'a str>,
    pub component: Option<&'a str>,
    pub base_point: Option<&'a str>,
    pub extra_payload: HashMap<String, serde_json::Value>,
}

/// The embedding-machinery references the tagger needs to drive `embed_chunks`
/// (Case 3) and `build_chunk_payload` (Cases 1/2/3) without re-deriving them.
///
/// Kept distinct from [`IngestItem`] (the branch-lineage identity bundle) so the
/// identity contract stays clean while Option C still reuses the one production
/// embed path. All in-scope at the post-parse seam in `run_ingest_pipeline`.
pub(crate) struct EmbedInputs<'a> {
    pub queue_item: &'a UnifiedQueueItem,
    pub document_content: &'a DocumentContent,
    pub file_path: &'a Path,
    pub file_document_id: &'a str,
    /// Project root path (for component detection in `inject_component`).
    pub base_path: &'a str,
}

/// The outcome of a [`tag_and_store`] call — which ladder arm ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TagOutcome {
    /// Case 1: a virtual view row + virtual (vectorless) Qdrant point.
    SharedExisting,
    /// Case 2: this file-identity's own real point, vector copied (not re-embedded).
    CopiedVector,
    /// Case 3: embedded + upserted as a new real point.
    EmbeddedNew,
    /// A delete tombstone was written. Constructed by the per-file delete path
    /// (task 23/24), which is not yet wired from this ADD chokepoint.
    #[allow(dead_code)]
    Tombstoned,
    /// A pure rename — metadata rewritten, no re-embed.
    MovedMetadataOnly,
}

/// What [`tag_and_store`] produced. `points`/`records` are non-empty only for
/// the real-point cases (2/3 + resurrection) that the downstream
/// concept/narrative/graph phases consume; virtual / move / idempotent outcomes
/// carry empty vecs (no re-embed → no graph work for this branch).
pub(crate) struct TagStored {
    pub outcome: TagOutcome,
    pub file_id: i64,
    pub points: Vec<DocumentPoint>,
    pub records: Vec<ChunkRecord>,
}

/// Errors the branch tagger can return.
#[derive(Debug, thiserror::Error)]
pub(crate) enum TaggerError {
    #[error("state.db error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("qdrant storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),
    #[error("embedding error: {0}")]
    Embed(String),
}
