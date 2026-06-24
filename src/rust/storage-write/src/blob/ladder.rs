//! The per-chunk dedup ladder: the two cases of arch §4.1.
//!
//! File: `wqm-storage-write/src/blob/ladder.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: Split out of [`crate::blob::dedup`] along the responsibility boundary
//!   (coding.md §X, AC-F6.6) — `dedup.rs` owns the FILE-level orchestration (the
//!   `files`/`concrete` upsert and the chunk loop); THIS module owns ONE chunk's
//!   write cycle under a held [`crate::blob::lock::ContentKeyLock`]:
//!     - content_key HIT  -> membership add, recompute branch_id[] from SQLite, PUT
//!       (read the STORED point_id; NEVER recompute it — DATA-05/SEC-4).
//!     - content_key MISS -> embed (lazy), INSERT blob, single-referrer upsert.
//!   The FP-1 ordering is enforced here: durable vectors land in SQLite BEFORE the
//!   Qdrant op is enqueued.
//!
//!   Qdrant writes are not executed inline — they are ENQUEUED into a [`QdrantSink`]
//!   so the batch flush can fire OUTSIDE the per-content_key lock (GP-6). Tests use an
//!   in-memory capturing sink to assert which op (upsert vs overwrite_payload PUT) each
//!   case produced and with exactly which payload.
//!
//! Neighbors: [`crate::blob::dedup`] (caller — file-level loop), [`crate::blob::embed`]
//!   (the lazy embed seam), [`crate::blob::membership`] (the canonical
//!   `compute_membership` producer — the HIT path delegates here, never re-implements
//!   the SELECT DISTINCT query; FP-2 / DR GP-1 / AC-F7.6).

use std::collections::HashMap;

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;
use wqm_common::hashing::{content_key_for_version, point_id};

use crate::blob::embed::{EmbeddedChunk, Embedder};
use crate::blob::vector_codec::{encode_dense, encode_sparse};

/// A Qdrant mutation the ladder enqueues for the batch flush (GP-6).
///
/// The three ladder cases map to the three variants:
///   - MISS -> [`QdrantOp::Upsert`] (new point: vectors + initial single-branch payload).
///   - HIT  -> [`QdrantOp::OverwritePayload`] (PUT — full payload replacement with the
///     recomputed full branch set; NO vectors, NO re-embed).
///   - DELETE (F9 orphan) -> [`QdrantOp::Delete`] (remove the point; no payload update).
///
/// `set_payload` (POST) is deliberately NOT representable: it has no array-append mode
/// and would silently drop prior memberships (arch §5.3 / AC-F6.2). Every payload
/// carries ALL three fields so a PUT can never delete one by omission.
#[derive(Debug, Clone, PartialEq)]
pub enum QdrantOp {
    /// New blob: upsert the point with its vectors and an initial single-branch payload.
    Upsert {
        point_id: String,
        dense: Vec<f32>,
        sparse: HashMap<u32, f32>,
        payload: BlobPayload,
    },
    /// Existing blob gained a branch: PUT (overwrite_payload) the full payload.
    /// `point_id` is the STORED `blobs.point_id` (read from SQLite, never recomputed)
    /// so a SEC-4 salted re-key is honored (AC-F6.2 / DATA-05).
    OverwritePayload {
        point_id: String,
        payload: BlobPayload,
    },
    /// Orphaned blob: delete the Qdrant point (F9 Step 3).
    ///
    /// Enqueued BEFORE the SQLite blobs row is deleted (FP-1: data product first,
    /// truth row last). The `collection` field names the Qdrant collection to target.
    /// The delete is applied by calling `QdrantWriteClient::delete_points` during
    /// the batch flush; tests capture it via `CaptureSink` without a live client.
    Delete {
        point_id: String,
        collection: String,
    },
}

/// The full Qdrant point payload (arch §5.3). ALL three fields are always present —
/// supplying only `branch_id` would silently delete `tenant_id`, breaking searches.
#[derive(Debug, Clone, PartialEq)]
pub struct BlobPayload {
    pub tenant_id: String,
    /// The FULL branch membership set. For a new blob this is `[current_branch_id]`;
    /// for a hit it is the set returned by `blob::membership::compute_membership`
    /// (the single canonical SELECT DISTINCT producer; AC-F7.6 / FP-2).
    pub branch_id: Vec<String>,
    pub collection_id: String,
}

/// The enqueue target for ladder-produced Qdrant ops. The real implementation buffers
/// ops and flushes them in ~1000-op batches outside the lock; tests capture them.
pub trait QdrantSink: Send {
    /// Enqueue one op for the batch flush. The ladder releases its lock before the
    /// flush actually runs (GP-6) — enqueueing is non-blocking and must not write.
    fn enqueue(&mut self, op: QdrantOp);
}

/// An in-memory capturing sink for unit tests: records ops in enqueue order.
#[derive(Debug, Default)]
pub struct CaptureSink {
    pub ops: Vec<QdrantOp>,
}

impl QdrantSink for CaptureSink {
    fn enqueue(&mut self, op: QdrantOp) {
        self.ops.push(op);
    }
}

/// Per-file context threaded through the chunk loop: the resolved IDs and the
/// once-per-session content_key version flag (PERF-R4-N1 — read once, cached here,
/// never per chunk).
pub struct ChunkContext<'a> {
    pub tenant_id: &'a str,
    pub branch_id: &'a str,
    pub collection_id: &'a str,
    pub file_id: i64,
    /// `projects.content_key_version` — read ONCE per session and cached (AC-F5.8 wiring).
    pub content_key_version: i64,
}

/// One chunk's content-addressing identity, computed once before the lock is taken so
/// the file-level loop can sort the locks by `content_key` (AC-F6.8).
#[derive(Debug, Clone)]
pub struct ChunkKey {
    /// The four-slot (or version-3) content_key for this chunk's content.
    pub content_key: String,
    /// The chunk's content hash (already in `ChunkInput`, carried for the blob INSERT).
    pub chunk_content_hash: String,
    /// Positional index within the file — written to `blob_refs.chunk_index`, NEVER to
    /// the point_id (blob points always use chunk_index 0, AC-F6.2).
    pub chunk_index: u32,
    /// The raw chunk text — embedded on a miss.
    pub text: String,
}

/// Compute a chunk's content_key from the cached version flag (AC-F6.2 / AC-F5.8).
///
/// `point_id` is intentionally NOT computed here: on a HIT the ladder reads the STORED
/// point_id, and on a MISS it derives `point_id(content_key, 0)` only after confirming
/// the blob is new. Deriving it eagerly would invite recomputing it on the hit path —
/// exactly the DATA-05 bug AC-F6.2 forbids.
pub fn chunk_key(ctx: &ChunkContext<'_>, chunk_content_hash: &str) -> String {
    // Chunk-blobs are the CODE bucket; the chunk hash is the identity slot, content-hash
    // slot empty (the F5 field contract).
    content_key_for_version(
        ctx.content_key_version,
        ctx.tenant_id,
        wqm_common::hashing::bucket::CODE,
        chunk_content_hash,
        "",
    )
}

/// The outcome of running ONE chunk through the ladder — which branch it took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkOutcome {
    /// content_key miss: a new blob was embedded and inserted.
    BlobCreated,
    /// content_key hit: an existing blob gained this branch's membership.
    BlobReused,
}

/// Run one chunk's write cycle under its already-held content_key lock (arch §4.1).
///
/// PRECONDITION: the caller holds the [`ContentKeyLock`](crate::blob::lock::ContentKeyLock)
/// for `key.content_key`. This function performs the SQLite writes inside one
/// transaction and enqueues exactly one [`QdrantOp`]; it never flushes Qdrant.
///
/// FP-1 ordering: the SQLite writes (including the durable vectors on a miss) commit
/// BEFORE the Qdrant op is enqueued.
pub async fn ingest_chunk(
    pool: &SqlitePool,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    ctx: &ChunkContext<'_>,
    key: &ChunkKey,
) -> Result<ChunkOutcome, StorageError> {
    // Look up the blob by its UNIQUE content_key. Present -> HIT; absent -> MISS.
    let existing = sqlx::query("SELECT blob_id, point_id FROM blobs WHERE content_key = ?")
        .bind(&key.content_key)
        .fetch_optional(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("blob lookup by content_key: {e}")))?;

    match existing {
        Some(row) => {
            let blob_id: i64 = row.get("blob_id");
            // Read the STORED point_id — do NOT recompute it (DATA-05/SEC-4): a salted
            // re-key persisted a point_id that no longer equals point_id(content_key, 0).
            let stored_point_id: String = row.get("point_id");
            ingest_hit(pool, sink, ctx, key, blob_id, stored_point_id).await?;
            Ok(ChunkOutcome::BlobReused)
        }
        None => {
            ingest_miss(pool, embedder, sink, ctx, key).await?;
            Ok(ChunkOutcome::BlobCreated)
        }
    }
}

/// content_key HIT (arch §4.1, AC-F6.2): the blob exists — add this branch's membership,
/// recompute the full branch set from SQLite, and PUT it against the STORED point_id.
async fn ingest_hit(
    pool: &SqlitePool,
    sink: &mut dyn QdrantSink,
    ctx: &ChunkContext<'_>,
    key: &ChunkKey,
    blob_id: i64,
    stored_point_id: String,
) -> Result<(), StorageError> {
    let now = wqm_common::timestamps::now_utc();

    // Add the referrer and the FTS membership row, both idempotent (the file may be
    // re-ingested; ON CONFLICT IGNORE makes a repeat a no-op).
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| StorageError::Sqlite(format!("begin hit tx: {e}")))?;

    sqlx::query(
        "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
         VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
    )
    .bind(ctx.branch_id)
    .bind(ctx.file_id)
    .bind(key.chunk_index as i64)
    .bind(blob_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| StorageError::Sqlite(format!("insert blob_refs (hit): {e}")))?;

    sqlx::query(
        "INSERT INTO fts_branch_membership(blob_id, branch_id) \
         VALUES (?, ?) ON CONFLICT DO NOTHING",
    )
    .bind(blob_id)
    .bind(ctx.branch_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| StorageError::Sqlite(format!("insert fts membership (hit): {e}")))?;

    tx.commit()
        .await
        .map_err(|e| StorageError::Sqlite(format!("commit hit tx: {e}")))?;
    let _ = now;

    // Recompute the FULL branch_id[] from SQLite truth, INSIDE the lock (arch §5.5).
    // Delegated to the single canonical producer (AC-F7.6 / FP-2 / DR GP-1):
    // `blob::membership::compute_membership` is the ONLY site of the
    // SELECT DISTINCT query in this crate.
    let branch_ids = crate::blob::membership::compute_membership(pool, blob_id).await?;

    // PUT (overwrite_payload) the full payload against the STORED point_id.
    sink.enqueue(QdrantOp::OverwritePayload {
        point_id: stored_point_id,
        payload: BlobPayload {
            tenant_id: ctx.tenant_id.to_string(),
            branch_id: branch_ids,
            collection_id: ctx.collection_id.to_string(),
        },
    });
    Ok(())
}

/// content_key MISS (arch §4.1, AC-F6.3): genuinely new content — embed, INSERT the
/// blob (FTS trigger fires), add the single referrer, and upsert with this one branch.
async fn ingest_miss(
    pool: &SqlitePool,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    ctx: &ChunkContext<'_>,
    key: &ChunkKey,
) -> Result<(), StorageError> {
    // Embed FIRST (the only embed in the whole ladder — a hit never reaches here).
    let EmbeddedChunk { dense, sparse } = embedder.embed(&key.text).await?;

    // The point_id for a new blob is the canonical derivation (chunk_index 0, AC-F6.2).
    let new_point_id = point_id(&key.content_key, 0).to_string();
    let now = wqm_common::timestamps::now_utc();

    let dense_blob = encode_dense(&dense);
    let sparse_blob = encode_sparse(&sparse);

    // FP-1: persist the durable vectors in SQLite BEFORE enqueuing any Qdrant op.
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| StorageError::Sqlite(format!("begin miss tx: {e}")))?;

    let insert = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&key.content_key)
    .bind(&key.chunk_content_hash)
    .bind(&new_point_id)
    .bind(ctx.tenant_id)
    .bind(&key.text)
    .bind(&dense_blob)
    .bind(&sparse_blob)
    .bind(&now)
    .execute(&mut *tx)
    .await
    .map_err(|e| StorageError::Sqlite(format!("insert blob (miss): {e}")))?;

    let blob_id = insert.last_insert_rowid();

    sqlx::query(
        "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
         VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
    )
    .bind(ctx.branch_id)
    .bind(ctx.file_id)
    .bind(key.chunk_index as i64)
    .bind(blob_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| StorageError::Sqlite(format!("insert blob_refs (miss): {e}")))?;

    sqlx::query(
        "INSERT INTO fts_branch_membership(blob_id, branch_id) \
         VALUES (?, ?) ON CONFLICT DO NOTHING",
    )
    .bind(blob_id)
    .bind(ctx.branch_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| StorageError::Sqlite(format!("insert fts membership (miss): {e}")))?;

    tx.commit()
        .await
        .map_err(|e| StorageError::Sqlite(format!("commit miss tx: {e}")))?;

    // Single referrer, lock held: the in-process branch_id IS the full membership at
    // point-of-enqueue, so no recompute query is needed (arch §4.1 new-blob note).
    sink.enqueue(QdrantOp::Upsert {
        point_id: new_point_id,
        dense,
        sparse,
        payload: BlobPayload {
            tenant_id: ctx.tenant_id.to_string(),
            branch_id: vec![ctx.branch_id.to_string()],
            collection_id: ctx.collection_id.to_string(),
        },
    });
    Ok(())
}

#[cfg(test)]
#[path = "ladder_tests.rs"]
mod tests;
