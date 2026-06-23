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

/// A Qdrant mutation the ladder enqueues for the batch flush (GP-6).
///
/// The two ladder cases map to the two variants:
///   - MISS -> [`QdrantOp::Upsert`] (new point: vectors + initial single-branch payload).
///   - HIT  -> [`QdrantOp::OverwritePayload`] (PUT — full payload replacement with the
///     recomputed full branch set; NO vectors, NO re-embed).
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

// ---------------------------------------------------------------------------
// Vector <-> BLOB encoding (stable little-endian, used for the SQLite columns)
// ---------------------------------------------------------------------------

/// Encode a dense vector as a little-endian `f32` byte array (4 bytes per element).
/// The element count is recoverable as `bytes.len() / 4`.
pub fn encode_dense(dense: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(dense.len() * 4);
    for v in dense {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Encode a sparse vector as `u32_le count` followed by `count` x (`u32_le term`,
/// `f32_le weight`) pairs. Terms are sorted so the encoding is deterministic for a
/// given map (a `HashMap` has no inherent order).
pub fn encode_sparse(sparse: &HashMap<u32, f32>) -> Vec<u8> {
    let mut entries: Vec<(&u32, &f32)> = sparse.iter().collect();
    entries.sort_unstable_by_key(|(term, _)| **term);
    let mut out = Vec::with_capacity(4 + entries.len() * 8);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (term, weight) in entries {
        out.extend_from_slice(&term.to_le_bytes());
        out.extend_from_slice(&weight.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::embed::mock::MockEmbedder;
    use crate::blob::test_support::{add_branch, fixture, TENANT};
    use sqlx::SqlitePool;

    const BRANCH_A: &str = "branch-a";
    const BRANCH_B: &str = "branch-b";
    const COLLECTION: &str = "projects";

    /// Insert a `files` row so the ladder's `blob_refs` FK is satisfiable, returning
    /// its file_id. The ladder itself does not touch `files` (that is `dedup.rs`).
    async fn make_file(pool: &SqlitePool, branch: &str, path: &str) -> i64 {
        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
             VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
        )
        .bind(branch)
        .bind(path)
        .execute(pool)
        .await
        .expect("file insert");
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(branch)
            .bind(path)
            .fetch_one(pool)
            .await
            .expect("file_id")
    }

    fn ctx<'a>(branch: &'a str, file_id: i64) -> ChunkContext<'a> {
        ChunkContext {
            tenant_id: TENANT,
            branch_id: branch,
            collection_id: COLLECTION,
            file_id,
            content_key_version: 4,
        }
    }

    fn key_for(c: &ChunkContext<'_>, text: &str, chunk_index: u32) -> ChunkKey {
        let chunk_content_hash = wqm_common::hashing::compute_content_hash(text);
        ChunkKey {
            content_key: chunk_key(c, &chunk_content_hash),
            chunk_content_hash,
            chunk_index,
            text: text.into(),
        }
    }

    // AC-F6.3: a content_key MISS embeds, inserts the blob, and enqueues an Upsert with
    // the single current branch as the initial membership.
    #[tokio::test]
    async fn miss_embeds_inserts_and_upserts() {
        let fx = fixture(BRANCH_A).await;
        let file_id = make_file(&fx.pool, BRANCH_A, "a.rs").await;
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();
        let c = ctx(BRANCH_A, file_id);
        let k = key_for(&c, "fn main() {}", 0);

        let outcome = ingest_chunk(&fx.pool, &*embedder, &mut sink, &c, &k)
            .await
            .expect("ingest miss");

        assert_eq!(outcome, ChunkOutcome::BlobCreated);
        assert_eq!(embedder.call_count(), 1, "a miss embeds exactly once");

        // The blob landed in SQLite (FP-1: before any flush).
        let blob_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE content_key = ?")
                .bind(&k.content_key)
                .fetch_one(&fx.pool)
                .await
                .unwrap();
        assert_eq!(blob_count, 1);

        // Exactly one Upsert op with single-branch payload carrying all three fields.
        assert_eq!(sink.ops.len(), 1);
        match &sink.ops[0] {
            QdrantOp::Upsert { payload, .. } => {
                assert_eq!(payload.tenant_id, TENANT);
                assert_eq!(payload.branch_id, vec![BRANCH_A.to_string()]);
                assert_eq!(payload.collection_id, COLLECTION);
            }
            other => panic!("expected Upsert, got {other:?}"),
        }
    }

    // AC-F6.3: "byte-identical content" re-ingested is a HIT, not a third case — it
    // reuses the blob and never re-embeds.
    #[tokio::test]
    async fn byte_identical_content_is_a_hit_not_a_new_blob() {
        let fx = fixture(BRANCH_A).await;
        let file1 = make_file(&fx.pool, BRANCH_A, "a.rs").await;
        let file2 = make_file(&fx.pool, BRANCH_A, "b.rs").await;
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();

        let text = "let x = 1;";
        // First ingest (file a) -> miss.
        let c1 = ctx(BRANCH_A, file1);
        let k1 = key_for(&c1, text, 0);
        let o1 = ingest_chunk(&fx.pool, &*embedder, &mut sink, &c1, &k1)
            .await
            .unwrap();
        assert_eq!(o1, ChunkOutcome::BlobCreated);

        // Second ingest of the SAME bytes in another file -> hit (same content_key).
        let c2 = ctx(BRANCH_A, file2);
        let k2 = key_for(&c2, text, 0);
        assert_eq!(
            k1.content_key, k2.content_key,
            "identical bytes -> same content_key"
        );
        let o2 = ingest_chunk(&fx.pool, &*embedder, &mut sink, &c2, &k2)
            .await
            .unwrap();
        assert_eq!(o2, ChunkOutcome::BlobReused, "byte-identical is a HIT");

        assert_eq!(embedder.call_count(), 1, "the hit must NOT re-embed");
        let blob_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs")
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(blob_count, 1, "one blob, deduped across files");
    }

    // AC-F6.2: the HIT path recomputes the FULL branch_id[] from SQLite and PUTs it.
    // After two branches reference the same content, the PUT carries BOTH branches.
    #[tokio::test]
    async fn hit_recomputes_full_membership_and_puts() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let file_a = make_file(&fx.pool, BRANCH_A, "a.rs").await;
        let file_b = make_file(&fx.pool, BRANCH_B, "a.rs").await;
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();

        let text = "shared chunk body";
        let ca = ctx(BRANCH_A, file_a);
        let ka = key_for(&ca, text, 0);
        ingest_chunk(&fx.pool, &*embedder, &mut sink, &ca, &ka)
            .await
            .unwrap();

        // Branch B references the same content -> hit, full membership = {A, B}.
        let cb = ctx(BRANCH_B, file_b);
        let kb = key_for(&cb, text, 0);
        ingest_chunk(&fx.pool, &*embedder, &mut sink, &cb, &kb)
            .await
            .unwrap();

        // The last op is an OverwritePayload (PUT) with both branches.
        match sink.ops.last().expect("an op") {
            QdrantOp::OverwritePayload { payload, .. } => {
                let mut branches = payload.branch_id.clone();
                branches.sort();
                assert_eq!(branches, vec![BRANCH_A.to_string(), BRANCH_B.to_string()]);
                assert_eq!(
                    payload.tenant_id, TENANT,
                    "PUT carries tenant_id (never dropped)"
                );
                assert_eq!(payload.collection_id, COLLECTION);
            }
            other => panic!("expected OverwritePayload PUT, got {other:?}"),
        }
    }

    // AC-F6.2 / DATA-05 / SEC-4: the HIT path PUTs against the STORED point_id, even
    // when that point_id was salted to a value != point_id(content_key, 0).
    #[tokio::test]
    async fn hit_targets_stored_salted_point_id() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let file_a = make_file(&fx.pool, BRANCH_A, "a.rs").await;
        let file_b = make_file(&fx.pool, BRANCH_B, "a.rs").await;
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();

        let text = "rekeyed chunk";
        let ca = ctx(BRANCH_A, file_a);
        let ka = key_for(&ca, text, 0);
        ingest_chunk(&fx.pool, &*embedder, &mut sink, &ca, &ka)
            .await
            .unwrap();

        // Force a re-key: overwrite the stored point_id with a salted derivation, as
        // the SEC-4 guard would (AC-F5.5). The natural derivation is point_id(ck, 0).
        let natural = point_id(&ka.content_key, 0).to_string();
        let salted =
            wqm_common::hashing::salted_point_id(&ka.content_key, &[0xde, 0xad], 0).to_string();
        assert_ne!(
            natural, salted,
            "the salted id differs from the natural one"
        );
        sqlx::query("UPDATE blobs SET point_id = ? WHERE content_key = ?")
            .bind(&salted)
            .bind(&ka.content_key)
            .execute(&fx.pool)
            .await
            .unwrap();

        // Now a hit from branch B must PUT against the STORED salted id, not `natural`.
        sink.ops.clear();
        let cb = ctx(BRANCH_B, file_b);
        let kb = key_for(&cb, text, 0);
        ingest_chunk(&fx.pool, &*embedder, &mut sink, &cb, &kb)
            .await
            .unwrap();

        match sink.ops.last().expect("a PUT") {
            QdrantOp::OverwritePayload { point_id: pid, .. } => {
                assert_eq!(*pid, salted, "PUT targets the STORED salted point_id");
                assert_ne!(*pid, natural, "NOT the freshly-derived point_id");
            }
            other => panic!("expected OverwritePayload, got {other:?}"),
        }
    }

    // FP-1: durable vectors persist in SQLite BEFORE any Qdrant op is enqueued. We
    // assert that on a miss the blob row (with non-empty vectors) exists and the sink
    // op references the same point_id (i.e. SQLite truth precedes the enqueue).
    #[tokio::test]
    async fn fp1_sqlite_vectors_precede_qdrant_enqueue() {
        let fx = fixture(BRANCH_A).await;
        let file_id = make_file(&fx.pool, BRANCH_A, "a.rs").await;
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();
        let c = ctx(BRANCH_A, file_id);
        let k = key_for(&c, "vectorized", 0);

        ingest_chunk(&fx.pool, &*embedder, &mut sink, &c, &k)
            .await
            .unwrap();

        // SQLite has the durable vectors.
        let row =
            sqlx::query("SELECT point_id, dense_vec, sparse_vec FROM blobs WHERE content_key = ?")
                .bind(&k.content_key)
                .fetch_one(&fx.pool)
                .await
                .unwrap();
        let stored_pid: String = row.get("point_id");
        let dense: Vec<u8> = row.get("dense_vec");
        let sparse: Vec<u8> = row.get("sparse_vec");
        assert!(!dense.is_empty(), "dense vector persisted before enqueue");
        assert!(!sparse.is_empty(), "sparse vector persisted before enqueue");

        // The enqueued op points at the SAME persisted point_id (truth-then-product).
        match &sink.ops[0] {
            QdrantOp::Upsert { point_id: pid, .. } => assert_eq!(*pid, stored_pid),
            other => panic!("expected Upsert, got {other:?}"),
        }
    }

    // AC-F6.5 (F04 race) at the SQLite layer: two ingests of the SAME content_key —
    // the second finds the blob present and takes the hit (membership) path. (The lock
    // that serializes the two is unit-tested in `lock.rs`; here we prove the second
    // call's BEHAVIOR is the hit path once the blob exists.)
    #[tokio::test]
    async fn second_ingest_of_same_content_key_takes_hit_path() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let file_a = make_file(&fx.pool, BRANCH_A, "a.rs").await;
        let file_b = make_file(&fx.pool, BRANCH_B, "a.rs").await;
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();
        let text = "racing content";

        let ca = ctx(BRANCH_A, file_a);
        let ka = key_for(&ca, text, 0);
        let first = ingest_chunk(&fx.pool, &*embedder, &mut sink, &ca, &ka)
            .await
            .unwrap();
        let cb = ctx(BRANCH_B, file_b);
        let kb = key_for(&cb, text, 0);
        let second = ingest_chunk(&fx.pool, &*embedder, &mut sink, &cb, &kb)
            .await
            .unwrap();

        assert_eq!(first, ChunkOutcome::BlobCreated);
        assert_eq!(
            second,
            ChunkOutcome::BlobReused,
            "second takes the membership path"
        );
    }

    // AC-F6.2: blob points always use chunk_index 0 in the point_id; the positional
    // index lives only in blob_refs.chunk_index. Two chunks at indices 3 and 7 of the
    // same content still map to the SAME single point_id.
    #[tokio::test]
    async fn point_id_ignores_chunk_index_position() {
        let fx = fixture(BRANCH_A).await;
        let file_id = make_file(&fx.pool, BRANCH_A, "a.rs").await;
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();
        let c = ctx(BRANCH_A, file_id);

        let text = "same bytes at two positions";
        let k_at_3 = key_for(&c, text, 3);
        ingest_chunk(&fx.pool, &*embedder, &mut sink, &c, &k_at_3)
            .await
            .unwrap();

        let stored: String = sqlx::query_scalar("SELECT point_id FROM blobs WHERE content_key = ?")
            .bind(&k_at_3.content_key)
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(
            stored,
            point_id(&k_at_3.content_key, 0).to_string(),
            "point_id uses index 0"
        );

        // blob_refs preserves the positional index 3.
        let stored_index: i64 = sqlx::query_scalar(
            "SELECT chunk_index FROM blob_refs WHERE branch_id = ? AND file_id = ?",
        )
        .bind(BRANCH_A)
        .bind(file_id)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
        assert_eq!(stored_index, 3, "positional index lives in blob_refs");
    }

    #[test]
    fn dense_encoding_round_trips_count() {
        let bytes = encode_dense(&[1.0, 2.5, -3.0]);
        assert_eq!(bytes.len(), 12, "3 f32 -> 12 bytes");
    }

    #[test]
    fn sparse_encoding_is_deterministic() {
        let mut m = HashMap::new();
        m.insert(7u32, 1.0f32);
        m.insert(3u32, 2.0f32);
        // Same map encodes identically regardless of insertion order.
        assert_eq!(encode_sparse(&m), encode_sparse(&m));
        // Count prefix = 2.
        assert_eq!(&encode_sparse(&m)[..4], &2u32.to_le_bytes());
    }
}
