//! Library-doc orphan migration on project delete (AC-F16.5).
//!
//! File: `wqm-storage-write/src/orphan.rs`
//! Location: `src/rust/storage-write/src/` (write-crate)
//! Context: When a project is deleted every library-collection doc attached to
//!   that project must survive in search. A "library doc" is a `files` row in
//!   the project's `store.db` whose `collection` column names a library bucket.
//!   This module is the public facade; implementation helpers live in
//!   [`ops`] (`orphan_ops.rs`) for codesize compliance (coding.md §X).
//!
//! ## Two-branch logic per doc
//!
//! 1. **Probe** — collect the doc's `chunk_content_hash` set from its blobs.
//!    `chunk_content_hash` is tenant-INDEPENDENT (SHA256 of raw chunk text);
//!    `content_key` is NOT used for cross-tenant matching (it embeds `tenant_id`
//!    and always differs — arch §5.4).
//!
//! 2. **Equal-cardinality match** — query the global library store for candidate
//!    docs sharing ANY of the same chunk hashes. An exact duplicate must have
//!    EQUAL cardinality AND every hash present (DOM-R8-N1 directional-subset
//!    hazard: a subset is NOT a match and takes the re-home branch).
//!
//! 3a. **DROP** (match found, products-then-truth FP-1): enqueue
//!     `QdrantOp::Delete` FIRST, then delete rows.
//!
//! 3b. **RE-HOME** (no match, truth-first FP-1 — Cluster B):
//!     (i)  INSERT destination blob rows in global store (recovery anchor);
//!     (ii) enqueue `QdrantOp::OverwritePayload` (`tenant_id` → global);
//!     (iii) delete source rows LAST.
//!     A crash between any step leaves the source row intact → case5 HEALs.
//!
//! ## Audit (SEC-F16-01)
//!
//! Every re-home emits `tracing::info!` with structured fields so the scope
//! promotion from project tenant to global tenant is never silent.
//!
//! ## Deferred wiring
//!
//! `migrate_project_library_docs` is ready for F18 wiring. The global library
//! store pool is opened by the caller via `crate::library::open_library_store`.
//! The live project-delete call site rides F18 / #175.
//!
//! Neighbors: [`crate::library`] (sentinel branch + global store open),
//!   [`crate::blob::ladder::{QdrantOp, QdrantSink}`] (op queue),
//!   [`crate::reconcile::case5`] (5th-case healer for crash-mid-re-tenant).

use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::blob::ladder::QdrantSink;

// Implementation helpers (probe, drop, rehome, GC).
#[path = "orphan_ops.rs"]
mod ops;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One library-collection doc in the project store: its file identity and
/// the set of chunk hashes that make up the doc's content fingerprint.
#[derive(Debug)]
pub struct LibraryDoc {
    /// `files.file_id` in the project store.
    pub file_id: i64,
    /// The library collection name (e.g. "libraries", "url").
    pub collection: String,
    /// Set of chunk records linked to this file via `blob_refs`.
    /// `chunk_content_hash` is tenant-independent SHA256 of raw chunk text.
    /// `point_id` is the STORED `blobs.point_id` (verbatim, DATA-05/SEC-4).
    pub chunks: Vec<ChunkRecord>,
}

/// One chunk belonging to a [`LibraryDoc`].
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub blob_id: i64,
    /// Tenant-independent hash (SHA256 of raw chunk text).
    pub chunk_content_hash: String,
    /// STORED `blobs.point_id` verbatim — never recomputed (DATA-05/SEC-4).
    pub point_id: String,
    /// Raw chunk text (durable — no re-embed needed on re-home).
    pub raw_text: String,
    /// Dense vector blob (durable — no re-embed).
    pub dense_vec: Vec<u8>,
    /// Sparse vector blob (durable — no re-embed).
    pub sparse_vec: Vec<u8>,
    /// `chunk_index` within the file (for the destination `blob_refs` row).
    pub chunk_index: i64,
}

/// Summary returned from [`migrate_project_library_docs`].
#[derive(Debug, Default)]
pub struct MigrationSummary {
    /// Docs dropped because an identical doc exists in the global library.
    pub dropped: u32,
    /// Docs re-homed to the global library (no identical copy existed).
    pub rehomed: u32,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Migrate all library-collection docs in `project_pool` on project delete.
///
/// For each library doc:
/// - Probes `global_pool` for an equal-cardinality chunk-hash set match.
/// - Match: DROP (products-then-truth FP-1).
/// - No match: RE-HOME to global (truth-first FP-1, Cluster B ordering).
///
/// `project_tenant_id` — the project's stable UUID string (from `store_meta`).
/// `global_tenant_id`  — the global library tenant UUID string.
/// `collection_id`     — the Qdrant collection name (for `QdrantOp` fields).
///
/// **Deferred wiring:** exposed as a clean public API for F18. The live
/// project-delete call site rides F18 / #175.
pub async fn migrate_project_library_docs<S>(
    project_pool: &SqlitePool,
    global_pool: &SqlitePool,
    sink: &mut S,
    project_tenant_id: &str,
    global_tenant_id: &str,
    collection_id: &str,
) -> Result<MigrationSummary, StorageError>
where
    S: QdrantSink,
{
    let docs = ops::collect_library_docs(project_pool).await?;

    let mut dropped = 0u32;
    let mut rehomed = 0u32;

    for doc in docs {
        let hashes: Vec<&str> = doc
            .chunks
            .iter()
            .map(|c| c.chunk_content_hash.as_str())
            .collect();
        let matched = ops::probe_global_for_equal_set(global_pool, &hashes).await?;

        if matched {
            ops::drop_project_doc(project_pool, sink, &doc, collection_id).await?;
            dropped += 1;
        } else {
            ops::rehome_doc_to_global(
                project_pool,
                global_pool,
                sink,
                &doc,
                project_tenant_id,
                global_tenant_id,
                collection_id,
            )
            .await?;
            rehomed += 1;
        }
    }

    Ok(MigrationSummary { dropped, rehomed })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "orphan_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "orphan_tests2.rs"]
mod tests2;
