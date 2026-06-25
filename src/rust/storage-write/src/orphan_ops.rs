//! Implementation helpers for AC-F16.5 orphan migration (probe, drop, re-home).
//!
//! File: `wqm-storage-write/src/orphan_ops.rs`
//! Location: `src/rust/storage-write/src/` (write-crate)
//! Context: Split from `orphan.rs` for codesize compliance (coding.md §X,
//!   500-line file limit). This module is `pub(super)` — only `orphan.rs`
//!   calls into it. All public surface for F18 lives in `orphan.rs`.
//!
//! ## Function map
//!
//! - `collect_library_docs` / `fetch_doc_chunks` — Step 1: enumerate docs.
//! - `probe_global_for_equal_set`                — Step 2: equal-cardinality probe.
//! - `drop_project_doc`                          — Step 3a: DROP (products-then-truth).
//! - `rehome_doc_to_global`                      — Step 3b: RE-HOME orchestrator.
//! - `copy_chunks_to_global`                     — RE-HOME step 1 (recovery anchor).
//! - `enqueue_tenant_updates`                    — RE-HOME step 2 (payload PUT + audit).
//! - `delete_source_rows` / `gc_orphaned_blobs`  — RE-HOME step 3 (source cleanup).
//! - `insert_global_file`                        — helper: files row in global store.

use sqlx::SqlitePool;
use tracing::info;
use wqm_common::error::StorageError;
use wqm_common::hashing::{bucket, content_key_v4, point_id as derive_point_id};
use wqm_common::timestamps::now_utc;

use crate::blob::ladder::{QdrantOp, QdrantSink};
use crate::library::LIBRARY_SENTINEL_BRANCH_ID;
use crate::orphan::{ChunkRecord, LibraryDoc};

// ---------------------------------------------------------------------------
// Step 1: collect library docs from the project store
// ---------------------------------------------------------------------------

/// Collect all library-collection docs from the project store.
///
/// A "library doc" is a `files` row whose `collection` is NOT 'projects'.
pub(super) async fn collect_library_docs(
    pool: &SqlitePool,
) -> Result<Vec<LibraryDoc>, StorageError> {
    let file_rows = sqlx::query_as::<_, (i64, String)>(
        "SELECT file_id, collection FROM files WHERE collection != 'projects'",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("orphan: list library files: {e}")))?;

    let mut docs = Vec::with_capacity(file_rows.len());
    for (file_id, collection) in file_rows {
        let chunks = fetch_doc_chunks(pool, file_id).await?;
        if !chunks.is_empty() {
            docs.push(LibraryDoc {
                file_id,
                collection,
                chunks,
            });
        }
    }
    Ok(docs)
}

/// Fetch all chunk records for `file_id` via `blobs JOIN blob_refs`.
async fn fetch_doc_chunks(
    pool: &SqlitePool,
    file_id: i64,
) -> Result<Vec<ChunkRecord>, StorageError> {
    let rows = sqlx::query_as::<_, (i64, String, String, String, Vec<u8>, Vec<u8>, i64)>(
        "SELECT b.blob_id, b.chunk_content_hash, b.point_id, b.raw_text, \
                b.dense_vec, b.sparse_vec, r.chunk_index \
         FROM blobs b \
         JOIN blob_refs r ON r.blob_id = b.blob_id \
         WHERE r.file_id = ?",
    )
    .bind(file_id)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("orphan: fetch chunks file={file_id}: {e}")))?;

    Ok(rows
        .into_iter()
        .map(
            |(
                blob_id,
                chunk_content_hash,
                point_id,
                raw_text,
                dense_vec,
                sparse_vec,
                chunk_index,
            )| {
                ChunkRecord {
                    blob_id,
                    chunk_content_hash,
                    point_id,
                    raw_text,
                    dense_vec,
                    sparse_vec,
                    chunk_index,
                }
            },
        )
        .collect())
}

// ---------------------------------------------------------------------------
// Step 2: probe global store for equal-cardinality chunk-hash match
// ---------------------------------------------------------------------------

/// Return true iff the global store contains a doc whose chunk-hash set equals
/// `project_hashes` exactly (same cardinality AND same members).
///
/// DOM-R8-N1 directional-subset hazard: "all project hashes present in global"
/// is one-directional. Equal cardinality AND full membership is required to
/// prevent a strict-subset project doc from being silently dropped.
pub(super) async fn probe_global_for_equal_set(
    global_pool: &SqlitePool,
    project_hashes: &[&str],
) -> Result<bool, StorageError> {
    let n = project_hashes.len() as i64;
    if n == 0 {
        return Ok(false);
    }

    // Dynamic IN-list: one '?' per hash, then two HAVING parameters.
    let placeholders = project_hashes
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT r.file_id \
         FROM blob_refs r \
         JOIN blobs b ON b.blob_id = r.blob_id \
         WHERE b.chunk_content_hash IN ({placeholders}) \
         GROUP BY r.file_id \
         HAVING COUNT(DISTINCT b.chunk_content_hash) = ? \
            AND (SELECT COUNT(*) FROM blob_refs r2 WHERE r2.file_id = r.file_id) = ? \
         LIMIT 1"
    );

    let mut q = sqlx::query_scalar::<_, i64>(&sql);
    for hash in project_hashes {
        q = q.bind(*hash);
    }
    q = q.bind(n).bind(n);

    let result = q
        .fetch_optional(global_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("orphan: probe global hash set: {e}")))?;

    Ok(result.is_some())
}

// ---------------------------------------------------------------------------
// Step 3a: DROP (match found — products-then-truth FP-1)
// ---------------------------------------------------------------------------

/// DROP the project copy of a library doc whose global duplicate was found.
///
/// FP-1 products-then-truth: enqueue `QdrantOp::Delete` FIRST, then delete rows.
pub(super) async fn drop_project_doc<S>(
    project_pool: &SqlitePool,
    sink: &mut S,
    doc: &LibraryDoc,
    collection: &str,
) -> Result<(), StorageError>
where
    S: QdrantSink,
{
    // Enqueue Delete for each blob's Qdrant point (data product first).
    for chunk in &doc.chunks {
        sink.enqueue(QdrantOp::Delete {
            point_id: chunk.point_id.clone(),
            collection: collection.to_owned(),
        });
    }
    // Remove blob_refs + files rows (truth rows after data product).
    sqlx::query("DELETE FROM blob_refs WHERE file_id = ?")
        .bind(doc.file_id)
        .execute(project_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("orphan drop: delete blob_refs: {e}")))?;
    sqlx::query("DELETE FROM files WHERE file_id = ?")
        .bind(doc.file_id)
        .execute(project_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("orphan drop: delete files: {e}")))?;
    // GC blobs with no remaining referrers.
    gc_orphaned_blobs(project_pool, doc).await
}

// ---------------------------------------------------------------------------
// Step 3b: RE-HOME (no match — truth-first FP-1, Cluster B ordering)
// ---------------------------------------------------------------------------

/// Re-home a unique project library doc to the global library store.
///
/// Cluster B ordering (FP-1 truth-first):
///   1. INSERT destination blob rows in global store (recovery anchor).
///   2. Enqueue `QdrantOp::OverwritePayload` (tenant_id → global) + audit log.
///   3. Delete source rows LAST (crash leaves source intact → case5 heals).
pub(super) async fn rehome_doc_to_global<S>(
    project_pool: &SqlitePool,
    global_pool: &SqlitePool,
    sink: &mut S,
    doc: &LibraryDoc,
    project_tenant_id: &str,
    global_tenant_id: &str,
    collection_id: &str,
) -> Result<(), StorageError>
where
    S: QdrantSink,
{
    let now = now_utc();
    copy_chunks_to_global(global_pool, doc, global_tenant_id, &now).await?;
    enqueue_tenant_updates(
        sink,
        doc,
        project_tenant_id,
        global_tenant_id,
        collection_id,
    );
    delete_source_rows(project_pool, doc).await
}

/// Step 1 of RE-HOME: INSERT destination blob rows in global store.
///
/// Each blob gets a new content_key for the global tenant (arch §5.4).
/// The point_id is derived from the global content_key (the Qdrant point
/// payload is updated in step 2 via `OverwritePayload`).
async fn copy_chunks_to_global(
    global_pool: &SqlitePool,
    doc: &LibraryDoc,
    global_tenant_id: &str,
    now: &str,
) -> Result<(), StorageError> {
    let global_file_id = insert_global_file(global_pool, &doc.collection, now).await?;
    for chunk in &doc.chunks {
        let global_ck = content_key_v4(
            global_tenant_id,
            bucket::CODE,
            &chunk.chunk_content_hash,
            "",
        );
        let global_pid = derive_point_id(&global_ck, 0).to_string();
        sqlx::query(
            "INSERT OR IGNORE INTO blobs(content_key, chunk_content_hash, point_id, \
             tenant_id, raw_text, dense_vec, sparse_vec, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&global_ck)
        .bind(&chunk.chunk_content_hash)
        .bind(&global_pid)
        .bind(global_tenant_id)
        .bind(&chunk.raw_text)
        .bind(&chunk.dense_vec)
        .bind(&chunk.sparse_vec)
        .bind(now)
        .execute(global_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("orphan re-home: global blob insert: {e}")))?;

        let global_blob_id: i64 =
            sqlx::query_scalar("SELECT blob_id FROM blobs WHERE content_key = ?")
                .bind(&global_ck)
                .fetch_one(global_pool)
                .await
                .map_err(|e| StorageError::Sqlite(format!("orphan re-home: fetch blob_id: {e}")))?;
        sqlx::query(
            "INSERT OR IGNORE INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
             VALUES (?, ?, ?, ?)",
        )
        .bind(LIBRARY_SENTINEL_BRANCH_ID)
        .bind(global_file_id)
        .bind(chunk.chunk_index)
        .bind(global_blob_id)
        .execute(global_pool)
        .await
        .map_err(|e| {
            StorageError::Sqlite(format!("orphan re-home: global blob_ref insert: {e}"))
        })?;
    }
    Ok(())
}

/// Step 2 of RE-HOME: enqueue `OverwritePayload` for each point + emit audit.
///
/// The audit event (SEC-F16-01) is emitted here so scope promotion to the
/// global tenant is never silent.
fn enqueue_tenant_updates<S>(
    sink: &mut S,
    doc: &LibraryDoc,
    project_tenant_id: &str,
    global_tenant_id: &str,
    collection_id: &str,
) where
    S: QdrantSink,
{
    for chunk in &doc.chunks {
        sink.enqueue(QdrantOp::OverwritePayload {
            point_id: chunk.point_id.clone(),
            payload: crate::blob::ladder::BlobPayload {
                tenant_id: global_tenant_id.to_owned(),
                branch_id: vec![LIBRARY_SENTINEL_BRANCH_ID.to_owned()],
                collection_id: collection_id.to_owned(),
            },
        });
    }
    info!(
        orphan_migrated = true,
        doc_file_id = doc.file_id,
        project_tenant = %project_tenant_id,
        global_tenant = %global_tenant_id,
        "orphan-migrated: doc {} (project {}) -> global tenant {}",
        doc.file_id, project_tenant_id, global_tenant_id,
    );
}

/// Step 3 of RE-HOME (LAST): delete source blob_refs, files, and GC blobs.
///
/// Executes AFTER steps 1+2; a crash here leaves source rows intact and
/// case5 (`reconcile/case5.rs`) heals the tenant-mismatch without culling.
async fn delete_source_rows(pool: &SqlitePool, doc: &LibraryDoc) -> Result<(), StorageError> {
    sqlx::query("DELETE FROM blob_refs WHERE file_id = ?")
        .bind(doc.file_id)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("orphan re-home: delete src blob_refs: {e}")))?;
    sqlx::query("DELETE FROM files WHERE file_id = ?")
        .bind(doc.file_id)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("orphan re-home: delete src files: {e}")))?;
    gc_orphaned_blobs(pool, doc).await
}

/// GC blobs from `doc` that have no remaining `blob_refs` referrers.
async fn gc_orphaned_blobs(pool: &SqlitePool, doc: &LibraryDoc) -> Result<(), StorageError> {
    for chunk in &doc.chunks {
        let ref_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?")
            .bind(chunk.blob_id)
            .fetch_one(pool)
            .await
            .map_err(|e| {
                StorageError::Sqlite(format!("orphan gc: ref_count blob {}: {e}", chunk.blob_id))
            })?;
        if ref_count == 0 {
            sqlx::query("DELETE FROM blobs WHERE blob_id = ?")
                .bind(chunk.blob_id)
                .execute(pool)
                .await
                .map_err(|e| {
                    StorageError::Sqlite(format!("orphan gc: delete blob {}: {e}", chunk.blob_id))
                })?;
        }
    }
    Ok(())
}

/// Insert a `files` row in the global store and return its `file_id`.
///
/// Uses a UUID-based `relative_path` to avoid uniqueness conflicts across
/// concurrent migrations to the same global store.
async fn insert_global_file(
    global_pool: &SqlitePool,
    collection: &str,
    now: &str,
) -> Result<i64, StorageError> {
    let path = format!("orphan-migration/{}", uuid::Uuid::new_v4());
    let row = sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?)",
    )
    .bind(LIBRARY_SENTINEL_BRANCH_ID)
    .bind(&path)
    .bind(collection)
    .bind(now)
    .bind(now)
    .execute(global_pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("orphan re-home: global files insert: {e}")))?;
    Ok(row.last_insert_rowid())
}
