//! File-level orchestration of the dedup ladder (arch §4.1, AC-F6.1).
//!
//! File: `wqm-storage-write/src/blob/dedup.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: The entry point of one file's ingest. It performs the file-level work
//!   ONCE (UPSERT `files` + `concrete`), then runs the per-chunk ladder
//!   ([`crate::blob::ladder`]) under per-content_key locks
//!   ([`crate::blob::lock`]). The split keeps this file small (AC-F6.6): file-level
//!   responsibility here, per-chunk write cycle in `ladder.rs`, the embed seam in
//!   `embed.rs`, the lock manager in `lock.rs`.
//!
//!   Two transactions per file (arch §5.5 additive): the file-level UPSERT commits
//!   first, then each chunk's blob/blob_refs/membership writes commit inside the
//!   per-content_key lock. The per-chunk lock covers ONLY the blob and its membership
//!   rows — never the `files`/`concrete` rows (AC-F6.1).
//!
//!   Deadlock freedom (AC-F6.8): all of a file's content_key locks are acquired in one
//!   sorted batch ([`crate::blob::lock::ContentKeyLockManager::lock_many`]) before the
//!   chunk loop, so two files sharing chunks acquire the shared locks in the same order.
//!
//! Neighbors: [`crate::blob::ladder`] (per-chunk cycle), [`crate::blob::lock`] (locks),
//!   [`crate::blob::embed`] (embed seam). The §8 nexus invariant — no blob/Qdrant blob
//!   point written outside a content_key lock — is realized by this orchestration.

use std::sync::Arc;

use sqlx::SqlitePool;
use wqm_common::error::StorageError;
use wqm_common::timestamps::now_utc;
use wqm_storage::types::requests::IngestFileRequest;
use wqm_storage::types::stats::IngestOutcome;

use crate::blob::embed::Embedder;
use crate::blob::ladder::{self, ChunkContext, ChunkKey, ChunkOutcome, QdrantSink};
use crate::blob::lock::ContentKeyLockManager;

/// The resolved per-file identifiers the ladder needs that are NOT in the request.
///
/// `tenant_id`/`branch_id`/`collection_id` come from the calling session; the file_id
/// is minted by the file-level UPSERT here. `content_key_version` is the once-per-session
/// cached `projects.content_key_version` flag (PERF-R4-N1 — see [`ingest_file`]).
pub struct IngestParams<'a> {
    pub tenant_id: &'a str,
    pub branch_id: &'a str,
    pub collection_id: &'a str,
    /// `projects.content_key_version` — read ONCE per session and cached by the caller,
    /// then passed here (never re-read per chunk). AC-F5.8 wiring deferred from F5.
    pub content_key_version: i64,
    /// The file's content hash at this ingest (written to `concrete.file_hash`).
    pub file_hash: &'a str,
}

/// Ingest one file's chunk batch through the dedup ladder (arch §4.1).
///
/// Sequence:
///   1. UPSERT the `files` row, minting/looking up `file_id` (file level, AC-F6.1).
///   2. UPSERT the `concrete` row (file level, AC-F6.1).
///   3. Compute each chunk's content_key; acquire ALL the file's locks in sorted order.
///   4. For each chunk, run the ladder under its held lock (hit -> membership/PUT;
///      miss -> embed/insert/upsert).
///
/// Returns the [`IngestOutcome`] counting created vs reused blobs. Qdrant ops are
/// ENQUEUED into `sink`; the actual batch flush runs OUTSIDE the locks (GP-6) and is
/// the caller's responsibility (the locks are released when the returned guards drop,
/// which happens before this function returns).
pub async fn ingest_file(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    params: &IngestParams<'_>,
    file: &IngestFileRequest,
) -> Result<IngestOutcome, StorageError> {
    // Step 1+2 (file level, ONE upsert each, before the chunk loop — AC-F6.1).
    let file_id = upsert_file_row(pool, params, file).await?;
    upsert_concrete_row(pool, params, file_id).await?;

    // Step 3: compute every chunk's content_key and the keys to lock. The lock set is
    // the chunks' content_keys; lock_many sorts + dedups them (AC-F6.8).
    let chunk_keys: Vec<ChunkKey> = file
        .chunks
        .iter()
        .map(|c| ChunkKey {
            content_key: ladder::chunk_key(
                &ChunkContext {
                    tenant_id: params.tenant_id,
                    branch_id: params.branch_id,
                    collection_id: params.collection_id,
                    file_id,
                    content_key_version: params.content_key_version,
                },
                &c.content_hash,
            ),
            chunk_content_hash: c.content_hash.clone(),
            chunk_index: c.chunk_index,
            text: c.text.clone(),
        })
        .collect();

    let lock_set: Vec<String> = chunk_keys.iter().map(|k| k.content_key.clone()).collect();
    // Hold ALL the file's locks for the chunk loop; dropped (released) on return.
    let _held = locks.lock_many(&lock_set).await;

    // Step 4: run each chunk through the ladder. The locks are already held; the ladder
    // does the per-chunk SQLite writes and enqueues exactly one Qdrant op per chunk.
    let ctx = ChunkContext {
        tenant_id: params.tenant_id,
        branch_id: params.branch_id,
        collection_id: params.collection_id,
        file_id,
        content_key_version: params.content_key_version,
    };

    let mut outcome = IngestOutcome::default();
    for key in &chunk_keys {
        match ladder::ingest_chunk(pool, embedder, sink, &ctx, key).await? {
            ChunkOutcome::BlobCreated => outcome.blobs_created += 1,
            ChunkOutcome::BlobReused => outcome.blobs_reused += 1,
        }
        outcome.chunks_ingested += 1;
    }
    Ok(outcome)
}

/// UPSERT the `files` row for `(branch_id, relative_path)` and return its `file_id`
/// (AC-F6.1). Idempotent: re-ingesting the same file updates `updated_at` and reuses
/// the existing row.
async fn upsert_file_row(
    pool: &SqlitePool,
    params: &IngestParams<'_>,
    file: &IngestFileRequest,
) -> Result<i64, StorageError> {
    let now = now_utc();
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, language, collection, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?) \
         ON CONFLICT(branch_id, relative_path) DO UPDATE SET \
           language = excluded.language, updated_at = excluded.updated_at",
    )
    .bind(params.branch_id)
    .bind(&file.path)
    .bind(file.language.as_deref())
    .bind(params.collection_id)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("upsert files row: {e}")))?;

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(params.branch_id)
            .bind(&file.path)
            .fetch_one(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("read back file_id: {e}")))?;
    Ok(file_id)
}

/// UPSERT the `concrete` row for `(branch_id, file_id)` (AC-F6.1). Records the file's
/// mtime/hash; ingest-status columns keep their defaults for F6 (LSP/tree-sitter
/// enrichment is a later feature).
async fn upsert_concrete_row(
    pool: &SqlitePool,
    params: &IngestParams<'_>,
    file_id: i64,
) -> Result<(), StorageError> {
    let now = now_utc();
    sqlx::query(
        "INSERT INTO concrete(branch_id, file_id, file_mtime, file_hash, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?) \
         ON CONFLICT(branch_id, file_id) DO UPDATE SET \
           file_mtime = excluded.file_mtime, file_hash = excluded.file_hash, \
           updated_at = excluded.updated_at",
    )
    .bind(params.branch_id)
    .bind(file_id)
    .bind(&now)
    .bind(params.file_hash)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("upsert concrete row: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::embed::mock::MockEmbedder;
    use crate::blob::ladder::CaptureSink;
    use crate::blob::test_support::{fixture, TENANT};
    use std::time::Duration;
    use wqm_common::hashing::compute_content_hash;
    use wqm_storage::types::requests::ChunkInput;

    const BRANCH_A: &str = "branch-a";
    const COLLECTION: &str = "projects";

    fn chunk(text: &str, idx: u32) -> ChunkInput {
        ChunkInput {
            chunk_index: idx,
            text: text.into(),
            content_hash: compute_content_hash(text),
        }
    }

    fn params<'a>(branch: &'a str, version: i64) -> IngestParams<'a> {
        IngestParams {
            tenant_id: TENANT,
            branch_id: branch,
            collection_id: COLLECTION,
            content_key_version: version,
            file_hash: "deadbeef",
        }
    }

    // AC-F6.1: the file-level `files` + `concrete` rows are upserted ONCE, and the
    // per-chunk writes land in blobs/blob_refs/fts_branch_membership.
    #[tokio::test]
    async fn file_level_rows_upserted_once_then_chunks() {
        let fx = fixture(BRANCH_A).await;
        let locks = ContentKeyLockManager::with_defaults();
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();

        let req = IngestFileRequest::new(
            "src/main.rs",
            vec![chunk("alpha", 0), chunk("beta", 1), chunk("gamma", 2)],
        );
        let p = params(BRANCH_A, 4);
        let outcome = ingest_file(&fx.pool, &locks, &*embedder, &mut sink, &p, &req)
            .await
            .expect("ingest_file");

        assert_eq!(outcome.chunks_ingested, 3);
        assert_eq!(
            outcome.blobs_created, 3,
            "three distinct chunks -> three blobs"
        );
        assert_eq!(outcome.blobs_reused, 0);

        // Exactly one files row and one concrete row for this (branch, path).
        let files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files")
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(files, 1, "files upserted once");
        let concrete: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM concrete")
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(concrete, 1, "concrete upserted once");

        // Per-chunk rows present.
        let blobs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs")
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(blobs, 3);
        let refs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs")
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(refs, 3);
        assert_eq!(sink.ops.len(), 3, "one Qdrant op per chunk");
    }

    // AC-F6.1: re-ingesting the SAME file does NOT create a second files/concrete row
    // (the file-level UPSERT is idempotent) and reuses the blobs (hit path).
    #[tokio::test]
    async fn reingest_same_file_is_idempotent_and_reuses_blobs() {
        let fx = fixture(BRANCH_A).await;
        let locks = ContentKeyLockManager::with_defaults();
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();

        let req = IngestFileRequest::new("src/lib.rs", vec![chunk("one", 0), chunk("two", 1)]);
        let p = params(BRANCH_A, 4);

        ingest_file(&fx.pool, &locks, &*embedder, &mut sink, &p, &req)
            .await
            .unwrap();
        let second = ingest_file(&fx.pool, &locks, &*embedder, &mut sink, &p, &req)
            .await
            .unwrap();

        assert_eq!(second.blobs_created, 0, "re-ingest creates no new blobs");
        assert_eq!(second.blobs_reused, 2, "both chunks hit");
        assert_eq!(embedder.call_count(), 2, "only the first pass embedded");

        let files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files")
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(files, 1, "no duplicate files row on re-ingest");
    }

    // AC-F5.8 wiring (PERF-R4-N1): the content_key_version flag is passed in already
    // cached — it is read ZERO times from the DB per chunk by the ladder. A file with
    // many chunks issues no `content_key_version` read at all (the value is a plain i64
    // field on IngestParams, consumed per chunk without a query).
    #[tokio::test]
    async fn version_flag_is_cached_not_read_per_chunk() {
        let fx = fixture(BRANCH_A).await;
        // There is no `projects` table in store.db at all — if the ladder tried to read
        // content_key_version per chunk it would error. A clean 50-chunk ingest proves
        // the flag is consumed from the cached field, never queried.
        let no_projects: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='projects'",
        )
        .fetch_one(&fx.pool)
        .await
        .unwrap();
        assert_eq!(
            no_projects, 0,
            "store.db has no projects table; flag must be cached"
        );

        let locks = ContentKeyLockManager::with_defaults();
        let embedder = MockEmbedder::new();
        let mut sink = CaptureSink::default();
        let chunks: Vec<ChunkInput> = (0..50)
            .map(|i| chunk(&format!("chunk body {i}"), i))
            .collect();
        let req = IngestFileRequest::new("big.rs", chunks);
        let p = params(BRANCH_A, 4);

        let outcome = ingest_file(&fx.pool, &locks, &*embedder, &mut sink, &p, &req)
            .await
            .expect("ingest must not read content_key_version per chunk");
        assert_eq!(outcome.chunks_ingested, 50);
    }

    // AC-F6.8 end-to-end: two files sharing >=2 chunks ingested concurrently in opposite
    // chunk orders do not deadlock — ingest_file acquires the shared locks in sorted
    // order via lock_many.
    #[tokio::test]
    async fn concurrent_shared_chunk_files_do_not_deadlock() {
        let fx = fixture(BRANCH_A).await;
        let pool = fx.pool.clone();
        let locks = ContentKeyLockManager::with_defaults();
        let embedder = MockEmbedder::new();

        // Two shared chunks ("shared-1", "shared-2") plus a file-unique one each. The
        // two files list the shared chunks in OPPOSITE order.
        let file1 = IngestFileRequest::new(
            "f1.rs",
            vec![
                chunk("shared-1", 0),
                chunk("shared-2", 1),
                chunk("uniq-1", 2),
            ],
        );
        let file2 = IngestFileRequest::new(
            "f2.rs",
            vec![
                chunk("shared-2", 0),
                chunk("shared-1", 1),
                chunk("uniq-2", 2),
            ],
        );

        let run = async {
            let p1 = pool.clone();
            let p2 = pool.clone();
            let l1 = locks.clone();
            let l2 = locks.clone();
            let e1 = embedder.clone();
            let e2 = embedder.clone();
            let h1 = tokio::spawn(async move {
                let mut s = CaptureSink::default();
                let pr = IngestParams {
                    tenant_id: TENANT,
                    branch_id: BRANCH_A,
                    collection_id: COLLECTION,
                    content_key_version: 4,
                    file_hash: "h1",
                };
                ingest_file(&p1, &l1, &*e1, &mut s, &pr, &file1)
                    .await
                    .unwrap();
            });
            let h2 = tokio::spawn(async move {
                let mut s = CaptureSink::default();
                let pr = IngestParams {
                    tenant_id: TENANT,
                    branch_id: BRANCH_A,
                    collection_id: COLLECTION,
                    content_key_version: 4,
                    file_hash: "h2",
                };
                ingest_file(&p2, &l2, &*e2, &mut s, &pr, &file2)
                    .await
                    .unwrap();
            });
            h1.await.unwrap();
            h2.await.unwrap();
        };
        tokio::time::timeout(Duration::from_secs(15), run)
            .await
            .expect("sorted multi-lock ingest must not deadlock");
    }
}
