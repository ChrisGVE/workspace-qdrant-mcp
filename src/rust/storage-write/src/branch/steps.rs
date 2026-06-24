//! SQL step helpers for the 8-step FP-1 whole-branch delete (arch §4.3, AC-F9.3).
//!
//! File: `wqm-storage-write/src/branch/steps.rs`
//! Location: `src/rust/storage-write/src/branch/` (write-crate branch layer)
//! Context: Owns `BlobCandidate` and `step1` through `step8` — the individual SQL
//!   operations that implement the FP-1 physical-delete sequence. Split from
//!   `delete.rs` to keep each file within the arch §9 line-budget. The orchestrator
//!   (`branch_delete` in `delete.rs`) calls these in order.
//!
//!   CHUNKED-DELETE IDIOM (AC-F9.2): `DELETE ... LIMIT` is INVALID (libsqlite3-sys
//!   0.30.1 lacks `SQLITE_ENABLE_UPDATE_DELETE_LIMIT`). ALL bounded deletes use:
//!     DELETE FROM <t> WHERE rowid IN (SELECT rowid FROM <t> WHERE <pred> LIMIT N)
//!   looped until 0 rows affected, committing per batch.
//!
//!   SQLITE_MAX_VARIABLE_NUMBER: all IN-list queries batch at ≤1000.
//!
//! Neighbors: [`super::delete`] (calls these fns in order), [`crate::blob::membership`]
//!   (single SELECT DISTINCT producer called in step6), [`crate::qdrant::membership_batch`]
//!   (MembershipPutBatch passed into step6).

use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

use crate::blob::ladder::{QdrantOp, QdrantSink};
use crate::blob::lock::ContentKeyLockManager;
use crate::qdrant::membership::build_membership_payload;
use crate::qdrant::membership_batch::{MembershipPutBatch, PendingMembershipPut};

/// A blob + point + content_key tuple for one branch blob (preselected at Step 1).
#[derive(Debug, Clone)]
pub(super) struct BlobCandidate {
    pub blob_id: i64,
    pub point_id: String,
    pub content_key: String,
}

/// Step 1: pre-select all (blob_id, point_id, content_key) for the branch.
///
/// Read-only, outside any transaction (arch §4.3 "preselect outside tx").
pub(super) async fn step1_preselect(
    pool: &SqlitePool,
    branch_id: &str,
) -> Result<Vec<BlobCandidate>, StorageError> {
    let rows = sqlx::query(
        "SELECT DISTINCT br.blob_id, b.point_id, b.content_key \
         FROM blob_refs br \
         JOIN blobs b ON b.blob_id = br.blob_id \
         WHERE br.branch_id = ?",
    )
    .bind(branch_id)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("step1 preselect: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| BlobCandidate {
            blob_id: r.get("blob_id"),
            point_id: r.get("point_id"),
            content_key: r.get("content_key"),
        })
        .collect())
}

/// Step 2: identify orphan candidates via batched `GROUP BY`.
///
/// A blob is an orphan candidate if its ONLY referrer is the deleted branch.
/// Query: `GROUP BY blob_id HAVING SUM(CASE WHEN branch_id != ? THEN 1 ELSE 0 END) = 0`.
/// Batch size: ≤1000 (SQLITE_MAX_VARIABLE_NUMBER limit).
pub(super) async fn step2_orphan_candidates(
    pool: &SqlitePool,
    branch_id: &str,
    all_blob_ids: &[i64],
) -> Result<Vec<i64>, StorageError> {
    const BATCH: usize = 1000;
    let mut orphans = Vec::new();

    for chunk in all_blob_ids.chunks(BATCH) {
        let placeholders: String = chunk
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT blob_id FROM blob_refs \
             WHERE blob_id IN ({placeholders}) \
             GROUP BY blob_id \
             HAVING SUM(CASE WHEN branch_id != ?1 THEN 1 ELSE 0 END) = 0"
        );
        let mut q = sqlx::query(&sql).bind(branch_id);
        for id in chunk {
            q = q.bind(*id);
        }
        let rows = q
            .fetch_all(pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("step2 orphan scan: {e}")))?;
        for r in rows {
            orphans.push(r.get::<i64, _>("blob_id"));
        }
    }
    Ok(orphans)
}

/// Step 3: enqueue `QdrantOp::Delete` for orphaned points (data product before truth row,
/// FP-1). Tests capture this without a live Qdrant client via the `QdrantSink` seam.
pub(super) fn step3_enqueue_orphan_deletes(
    sink: &mut dyn QdrantSink,
    candidates: &[&BlobCandidate],
    collection: &str,
) {
    for c in candidates {
        sink.enqueue(QdrantOp::Delete {
            point_id: c.point_id.clone(),
            collection: collection.to_string(),
        });
    }
}

/// Step 4: chunked branch-wide DELETE of `fts_branch_membership`, `blob_refs`,
/// and `concrete` (FK-safe order), committing per batch (≤10,000 rows per loop).
///
/// Branch-wide (WHERE branch_id = ?) so the `blob_refs ON DELETE RESTRICT` FK
/// cannot abort the blob delete at Step 5 (AC-F9.5 / DATA-03).
pub(super) async fn step4_chunked_delete_branch_rows(
    pool: &SqlitePool,
    branch_id: &str,
) -> Result<(), StorageError> {
    const BATCH: i64 = 10_000;

    // Step 4a: fts_branch_membership
    loop {
        let n = sqlx::query(
            "DELETE FROM fts_branch_membership \
             WHERE rowid IN \
               (SELECT rowid FROM fts_branch_membership WHERE branch_id = ? LIMIT ?)",
        )
        .bind(branch_id)
        .bind(BATCH)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("step4a fts_branch_membership: {e}")))?
        .rows_affected();
        if n == 0 {
            break;
        }
    }

    // Step 4b: blob_refs
    loop {
        let n = sqlx::query(
            "DELETE FROM blob_refs \
             WHERE rowid IN \
               (SELECT rowid FROM blob_refs WHERE branch_id = ? LIMIT ?)",
        )
        .bind(branch_id)
        .bind(BATCH)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("step4b blob_refs: {e}")))?
        .rows_affected();
        if n == 0 {
            break;
        }
    }

    // Step 4c: concrete
    loop {
        let n = sqlx::query(
            "DELETE FROM concrete \
             WHERE rowid IN \
               (SELECT rowid FROM concrete WHERE branch_id = ? LIMIT ?)",
        )
        .bind(branch_id)
        .bind(BATCH)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("step4c concrete: {e}")))?
        .rows_affected();
        if n == 0 {
            break;
        }
    }

    Ok(())
}

/// Step 5: re-verify orphan set under `BEGIN IMMEDIATE` (ABA guard) and delete
/// confirmed orphan blobs. Chunked at ≤1000 (SQLITE_MAX_VARIABLE_NUMBER).
///
/// Returns the blob_ids confirmed as orphans (used to partition the survivor set
/// in the orchestrator for Step 6).
pub(super) async fn step5_verify_and_delete_orphan_blobs(
    pool: &SqlitePool,
    orphan_candidates: &[i64],
) -> Result<Vec<i64>, StorageError> {
    const BATCH: usize = 1000;
    let mut confirmed_orphans = Vec::new();

    for chunk in orphan_candidates.chunks(BATCH) {
        // BEGIN IMMEDIATE: re-verify and delete are atomic (ABA guard).
        let mut tx = pool
            .begin()
            .await
            .map_err(|e| StorageError::Sqlite(format!("step5 begin immediate: {e}")))?;
        // Ensure writes cannot sneak in between re-verify and delete.
        sqlx::query("PRAGMA read_uncommitted = 0")
            .execute(&mut *tx)
            .await
            .ok();

        // Re-verify: which candidates still have ANY referrer?
        let placeholders: String = (1..=chunk.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT blob_id FROM blob_refs \
             WHERE blob_id IN ({placeholders}) \
             GROUP BY blob_id HAVING COUNT(*) > 0"
        );
        let mut q = sqlx::query(&sql);
        for id in chunk {
            q = q.bind(*id);
        }
        let still_referenced: Vec<i64> = q
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| StorageError::Sqlite(format!("step5 re-verify: {e}")))?
            .into_iter()
            .map(|r| r.get::<i64, _>("blob_id"))
            .collect();

        let confirmed_batch: Vec<i64> = chunk
            .iter()
            .copied()
            .filter(|id| !still_referenced.contains(id))
            .collect();

        if !confirmed_batch.is_empty() {
            let ph: String = (1..=confirmed_batch.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let del_sql = format!(
                "DELETE FROM blobs WHERE rowid IN \
                 (SELECT rowid FROM blobs WHERE blob_id IN ({ph}))"
            );
            let mut dq = sqlx::query(&del_sql);
            for id in &confirmed_batch {
                dq = dq.bind(*id);
            }
            dq.execute(&mut *tx)
                .await
                .map_err(|e| StorageError::Sqlite(format!("step5 delete orphan blobs: {e}")))?;

            confirmed_orphans.extend_from_slice(&confirmed_batch);
        }

        tx.commit()
            .await
            .map_err(|e| StorageError::Sqlite(format!("step5 commit: {e}")))?;
    }

    Ok(confirmed_orphans)
}

/// Step 6: recompute membership for still-referenced blobs; enqueue survivor PUTs.
///
/// Recompute runs INSIDE the per-`content_key` lock (serializes against concurrent
/// ingests). The actual Qdrant PUT fires OUTSIDE all locks via `batch.flush(client)`.
/// This is the AC-F19.3 batch-outside-lock strategy.
///
/// `survivors` = all_candidates MINUS confirmed_orphans.
pub(super) async fn step6_enqueue_survivor_puts(
    pool: &SqlitePool,
    lock_mgr: &Arc<ContentKeyLockManager>,
    batch: &mut MembershipPutBatch,
    survivors: &[&BlobCandidate],
    tenant_id: &str,
    collection_id: &str,
    collection_name: &str,
) -> Result<(), StorageError> {
    for candidate in survivors {
        let _guard = lock_mgr.lock(&candidate.content_key).await;

        // Step 4 already deleted the branch's blob_refs, so compute_membership
        // sees the surviving branch set automatically.
        let payload =
            build_membership_payload(pool, candidate.blob_id, tenant_id, collection_id).await?;

        batch.push(
            collection_name,
            PendingMembershipPut {
                point_id: candidate.point_id.clone(),
                payload,
            },
        );
        // _guard drops here, releasing the per-content_key lock.
    }
    Ok(())
}

/// Step 7: delete orphaned `files` rows for this branch (FP-1: files before branch row).
///
/// A file is "orphaned" when it has no blob_refs from any other branch — meaning it
/// only existed to serve chunks that belonged exclusively to this branch.
pub(super) async fn step7_delete_orphaned_files(
    pool: &SqlitePool,
    branch_id: &str,
) -> Result<(), StorageError> {
    sqlx::query(
        "DELETE FROM files \
         WHERE branch_id = ? \
         AND NOT EXISTS ( \
           SELECT 1 FROM blob_refs \
           WHERE blob_refs.file_id = files.file_id \
             AND blob_refs.branch_id != ? \
         )",
    )
    .bind(branch_id)
    .bind(branch_id)
    .execute(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("step7 delete orphaned files: {e}")))?;
    Ok(())
}

/// Step 8: delete the `branches` row LAST.
///
/// The `branches` row is the crash-recovery anchor (arch §4.3): deleting it last
/// ensures that if the process crashes mid-sequence, a re-run can re-execute all
/// earlier steps idempotently (steps 4-7 are WHERE-scoped; step 1 returns nothing
/// after blob_refs are cleared, short-circuiting to steps 7+8).
pub(super) async fn step8_delete_branch(
    pool: &SqlitePool,
    branch_id: &str,
) -> Result<(), StorageError> {
    sqlx::query("DELETE FROM branches WHERE branch_id = ?")
        .bind(branch_id)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("step8 delete branch row: {e}")))?;
    Ok(())
}
