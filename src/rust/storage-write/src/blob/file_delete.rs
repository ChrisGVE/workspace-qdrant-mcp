//! Single-file delete from a branch (arch §4.3 single-file variant, AC-F9.6).
//!
//! File: `wqm-storage-write/src/blob/file_delete.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: Implements `delete_file_from_branch` — the scoped variant of the FP-1
//!   delete sequence that removes ONE file from ONE branch without tearing down
//!   the whole branch (arch §4.3 "Single-file delete variant", DATA-06/AC-F9.6).
//!
//!   The same FP-1 ordering applies (data products first, truth rows last):
//!     1. Pre-select (blob_id, point_id, content_key) via a JOIN scoped to
//!        (branch_id, file_id) — read-only, outside any transaction.
//!     2. DELETE blob_refs/fts_branch_membership/concrete for (branch_id, file_id)
//!        inside one BEGIN IMMEDIATE transaction.
//!     3. For each blob from Step 1, test refcount:
//!        - count 0  -> orphan: enqueue Qdrant Delete, then delete blobs row
//!                      with ABA re-verify (BEGIN IMMEDIATE).
//!        - count > 0 -> survivor: recompute membership + enqueue PUT in
//!                      MembershipPutBatch (AC-F19.3 strategy, INSIDE lock).
//!     4. DELETE the files row for (branch_id, file_id) LAST only if no blob_refs
//!        remain for this (branch_id, file_id) — crash-recovery anchor.
//!        The branches row is UNTOUCHED (AC-F9.6-e).
//!
//!   The blob_refs.blob_id ON DELETE RESTRICT FK means the blobs DELETE in step 3
//!   only succeeds AFTER the blob_refs DELETE has committed (step 2).
//!
//! Neighbors: [`crate::branch::delete`] (whole-branch delete), [`crate::blob::gc`]
//!   (refcount helper), [`crate::blob::membership`] (single SELECT DISTINCT producer),
//!   [`crate::qdrant::membership_batch`] (survivor PUT batch).

use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

use crate::blob::ladder::{QdrantOp, QdrantSink};
use crate::blob::lock::ContentKeyLockManager;
use crate::qdrant::membership::build_membership_payload;
use crate::qdrant::membership_batch::{MembershipPutBatch, PendingMembershipPut};

/// One blob referenced by the target (branch_id, file_id).
#[derive(Debug, Clone)]
struct FileBlob {
    blob_id: i64,
    point_id: String,
    content_key: String,
}

/// Step 1: pre-select all (blob_id, point_id, content_key) for (branch_id, file_id).
async fn preselect_file_blobs(
    pool: &SqlitePool,
    branch_id: &str,
    file_id: i64,
) -> Result<Vec<FileBlob>, StorageError> {
    let rows = sqlx::query(
        "SELECT DISTINCT br.blob_id, b.point_id, b.content_key \
         FROM blob_refs br \
         JOIN blobs b ON b.blob_id = br.blob_id \
         WHERE br.branch_id = ? AND br.file_id = ?",
    )
    .bind(branch_id)
    .bind(file_id)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("file_delete preselect: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| FileBlob {
            blob_id: r.get("blob_id"),
            point_id: r.get("point_id"),
            content_key: r.get("content_key"),
        })
        .collect())
}

/// Step 2: delete junction rows scoped to (branch_id, file_id) inside one transaction.
///
/// fts_branch_membership is removed only for blob_ids that have LOST their last
/// membership row for this branch after the blob_refs delete — we delete it
/// unconditionally here and rely on the FK cascade semantics not applying (it's
/// not a cascade FK on blob_refs; we handle it explicitly).
async fn delete_file_junction_rows(
    pool: &SqlitePool,
    branch_id: &str,
    file_id: i64,
    blob_ids: &[i64],
) -> Result<(), StorageError> {
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| StorageError::Sqlite(format!("file_delete begin tx: {e}")))?;

    // Delete blob_refs for this (branch_id, file_id).
    sqlx::query("DELETE FROM blob_refs WHERE branch_id = ? AND file_id = ?")
        .bind(branch_id)
        .bind(file_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::Sqlite(format!("file_delete blob_refs: {e}")))?;

    // Delete concrete for this (branch_id, file_id).
    sqlx::query("DELETE FROM concrete WHERE branch_id = ? AND file_id = ?")
        .bind(branch_id)
        .bind(file_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::Sqlite(format!("file_delete concrete: {e}")))?;

    // Delete fts_branch_membership for any blob that no longer has any blob_refs
    // on this branch (not just this file — a blob could be referenced by another
    // file on the same branch). We re-check per blob_id.
    for &blob_id in blob_ids {
        let still_on_branch: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM blob_refs WHERE blob_id = ? AND branch_id = ?",
        )
        .bind(blob_id)
        .bind(branch_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| StorageError::Sqlite(format!("file_delete fts recheck: {e}")))?;

        if still_on_branch == 0 {
            sqlx::query("DELETE FROM fts_branch_membership WHERE blob_id = ? AND branch_id = ?")
                .bind(blob_id)
                .bind(branch_id)
                .execute(&mut *tx)
                .await
                .map_err(|e| {
                    StorageError::Sqlite(format!("file_delete fts_branch_membership: {e}"))
                })?;
        }
    }

    tx.commit()
        .await
        .map_err(|e| StorageError::Sqlite(format!("file_delete commit junction: {e}")))?;

    Ok(())
}

/// Step 3 (orphan path): ABA-guarded re-verify and delete orphan blobs.
/// Returns the set of blob_ids that were confirmed orphans and deleted.
async fn delete_confirmed_orphan_blobs(
    pool: &SqlitePool,
    sink: &mut dyn QdrantSink,
    candidates: &[&FileBlob],
    collection_name: &str,
) -> Result<Vec<i64>, StorageError> {
    let mut deleted_ids = Vec::new();

    for candidate in candidates {
        // BEGIN IMMEDIATE: re-verify and delete are atomic (ABA guard).
        let mut tx = pool
            .begin()
            .await
            .map_err(|e| StorageError::Sqlite(format!("orphan blob begin immediate: {e}")))?;

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?")
            .bind(candidate.blob_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| StorageError::Sqlite(format!("orphan re-verify: {e}")))?;

        if count == 0 {
            // Confirmed orphan: enqueue Qdrant Delete FIRST (FP-1: data product before truth).
            sink.enqueue(QdrantOp::Delete {
                point_id: candidate.point_id.clone(),
                collection: collection_name.to_string(),
            });

            sqlx::query(
                "DELETE FROM blobs WHERE rowid IN \
                 (SELECT rowid FROM blobs WHERE blob_id = ?)",
            )
            .bind(candidate.blob_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| StorageError::Sqlite(format!("orphan blob delete: {e}")))?;

            deleted_ids.push(candidate.blob_id);
        }

        tx.commit()
            .await
            .map_err(|e| StorageError::Sqlite(format!("orphan blob commit: {e}")))?;
    }

    Ok(deleted_ids)
}

/// Delete one file from one branch (arch §4.3 single-file variant, AC-F9.6).
///
/// The `branches` row is UNTOUCHED (AC-F9.6-e). The orphaned Qdrant points are
/// captured in `sink`; survivor membership PUTs are accumulated in `batch` for
/// flushing outside all locks.
///
/// No embedding call is made at any point (AC-F9.8).
///
/// `file_id` is the `files.file_id` (integer primary key) of the file being removed.
pub async fn delete_file_from_branch(
    pool: &SqlitePool,
    lock_mgr: &Arc<ContentKeyLockManager>,
    sink: &mut dyn QdrantSink,
    batch: &mut MembershipPutBatch,
    branch_id: &str,
    file_id: i64,
    tenant_id: &str,
    collection_id: &str,
    collection_name: &str,
) -> Result<(), StorageError> {
    // Step 1: pre-select file's blobs (read-only, outside transaction).
    let file_blobs = preselect_file_blobs(pool, branch_id, file_id).await?;
    if file_blobs.is_empty() {
        // No blobs for this file on this branch; delete the files row if it exists.
        sqlx::query("DELETE FROM files WHERE branch_id = ? AND file_id = ?")
            .bind(branch_id)
            .bind(file_id)
            .execute(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("file_delete files row (empty): {e}")))?;
        return Ok(());
    }

    let all_blob_ids: Vec<i64> = file_blobs.iter().map(|b| b.blob_id).collect();

    // Step 2: delete junction rows (blob_refs / concrete / fts_branch_membership).
    delete_file_junction_rows(pool, branch_id, file_id, &all_blob_ids).await?;

    // Step 3: for each blob, check refcount and act.
    let orphan_candidates: Vec<&FileBlob> = file_blobs.iter().collect::<Vec<_>>();

    // Separate into orphans (count == 0) and survivors (count > 0).
    let mut orphans = Vec::new();
    let mut survivors = Vec::new();
    for fb in &orphan_candidates {
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?")
            .bind(fb.blob_id)
            .fetch_one(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("refcount check: {e}")))?;
        if count == 0 {
            orphans.push(*fb);
        } else {
            survivors.push(*fb);
        }
    }

    // Orphan path: ABA-guarded delete (Qdrant Delete enqueued inside).
    delete_confirmed_orphan_blobs(pool, sink, &orphans, collection_name).await?;

    // Survivor path: recompute membership + enqueue PUT (INSIDE lock, batch flush outside).
    for fb in survivors {
        let _guard = lock_mgr.lock(&fb.content_key).await;

        let payload = build_membership_payload(pool, fb.blob_id, tenant_id, collection_id).await?;

        batch.push(
            collection_name,
            PendingMembershipPut {
                point_id: fb.point_id.clone(),
                payload,
            },
        );
        // _guard drops here.
    }

    // Step 4: delete the files row LAST (crash-recovery anchor).
    // Only if no blob_refs remain for this (branch_id, file_id).
    let remaining_refs: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE branch_id = ? AND file_id = ?")
            .bind(branch_id)
            .bind(file_id)
            .fetch_one(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("file_delete final ref check: {e}")))?;

    if remaining_refs == 0 {
        sqlx::query("DELETE FROM files WHERE branch_id = ? AND file_id = ?")
            .bind(branch_id)
            .bind(file_id)
            .execute(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("file_delete files row: {e}")))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (extracted to sibling file to keep this file within codesize budget)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "file_delete_tests.rs"]
mod tests;
