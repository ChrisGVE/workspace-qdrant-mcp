//! Case 2 -- stale Qdrant point / orphan blob prune (arch §4.7 case 2, AC-F15.3).
//!
//! File: `wqm-storage-write/src/reconcile/case2.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: Reconcile case 2 handles drift where Qdrant holds a point that has no
//!   referrers in SQLite `blob_refs` (orphan blob). Root cause: crash after §4.3
//!   Step 3 (Qdrant delete failed) but the blob row survived.
//!
//!   **MUST run AFTER case 5** (cross-DB tenant-mismatch heal). Without case 5 first,
//!   a mis-tenanted point would have zero referrers in the WRONG store and be wrongly
//!   culled here -- silent data loss (DATA-R7-04).
//!
//!   Fix:
//!     1. Find orphan candidates: blobs with `blob_id > watermark` where
//!        `(SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?) = 0`.
//!     2. ABA guard: re-verify inside `BEGIN IMMEDIATE` before deleting.
//!     3. For confirmed orphans: enqueue `QdrantOp::Delete` then delete the
//!        `blobs` row (FP-1 ordering: data product first, truth row last).
//!        The `blobs_ad` FTS5 AFTER DELETE trigger keeps `fts_content` in sync.
//!
//! The incremental window is `blob_id > watermark`; pass `watermark = 0` for FULL.
//!
//! Neighbors: [`crate::blob::gc::{blob_refcount, delete_orphan_blob_row}`] (GC helpers),
//!   [`crate::blob::ladder::{QdrantOp, QdrantSink}`] (op sink).

use sqlx::{Row, Sqlite, SqlitePool, Transaction};
use wqm_common::error::StorageError;

use crate::blob::ladder::{QdrantOp, QdrantSink};

/// One orphan candidate: a blob with no referrers in the scan window.
pub(super) struct OrphanCandidate {
    blob_id: i64,
    point_id: String,
}

/// Scan the watermark window for blobs that currently have zero referrers.
///
/// This is the initial candidate scan (outside a transaction). ABA re-verification
/// under `BEGIN IMMEDIATE` happens in [`confirm_and_delete_orphan`].
/// `pub(super)` so ABA tests can call scan then confirm separately.
pub(super) async fn scan_orphan_candidates(
    pool: &SqlitePool,
    watermark: i64,
) -> Result<Vec<OrphanCandidate>, StorageError> {
    let rows = sqlx::query(
        "SELECT b.blob_id, b.point_id \
         FROM blobs b \
         WHERE b.blob_id > ? \
           AND (SELECT COUNT(*) FROM blob_refs r WHERE r.blob_id = b.blob_id) = 0 \
         ORDER BY b.blob_id",
    )
    .bind(watermark)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("case2 scan_orphans: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| OrphanCandidate {
            blob_id: r.get("blob_id"),
            point_id: r.get("point_id"),
        })
        .collect())
}

/// ABA-guarded re-verify + delete for one orphan candidate.
///
/// Holds a `pool.begin()` transaction for the entire re-verify + delete sequence
/// so the single pool connection is kept for the duration of the RMW. This means
/// no concurrent ingest can insert a `blob_ref` between the count check and the
/// DELETE -- preserving still-referenced blobs (AC-F15.3).
///
/// FP-1 ordering: the `QdrantOp::Delete` is enqueued BEFORE the `blobs` row is
/// deleted (data product before truth row).
///
/// Returns true if the orphan was confirmed and deleted; false if ABA survivor.
/// `pub(super)` so ABA tests can call scan then confirm separately.
pub(super) async fn confirm_and_delete_orphan<S: QdrantSink>(
    pool: &SqlitePool,
    sink: &mut S,
    candidate: &OrphanCandidate,
    collection_name: &str,
) -> Result<bool, StorageError> {
    // pool.begin() acquires the single connection and holds it for the whole RMW.
    // A concurrent ingest trying pool.begin() or pool.acquire() blocks until we
    // commit or drop -- this is the ABA serialization barrier (DATA-NIT-02).
    let mut tx: Transaction<'_, Sqlite> = pool
        .begin()
        .await
        .map_err(|e| StorageError::Sqlite(format!("case2 begin: {e}")))?;

    // Re-read refcount INSIDE the held transaction (ABA guard).
    let refcount: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?")
        .bind(candidate.blob_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| StorageError::Sqlite(format!("case2 refcount: {e}")))?;

    if refcount == 0 {
        // Confirmed orphan: enqueue Qdrant Delete FIRST (FP-1: data product before truth).
        sink.enqueue(QdrantOp::Delete {
            point_id: candidate.point_id.clone(),
            collection: collection_name.to_owned(),
        });

        // Delete the blobs row INSIDE the held transaction.
        sqlx::query(
            "DELETE FROM blobs WHERE rowid IN \
             (SELECT rowid FROM blobs WHERE blob_id = ?)",
        )
        .bind(candidate.blob_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::Sqlite(format!("case2 blob delete: {e}")))?;

        tx.commit()
            .await
            .map_err(|e| StorageError::Sqlite(format!("case2 commit: {e}")))?;

        Ok(true)
    } else {
        // ABA survivor: a new ref appeared after the candidate scan; keep blob alive.
        // tx drops here, rolling back automatically.
        Ok(false)
    }
}

/// Reconcile case 2: prune orphan blobs (zero referrers) with ABA guard.
///
/// Scans `blobs` in the watermark window (`blob_id > watermark`), identifies
/// zero-referrer candidates, then ABA-re-verifies each under `BEGIN IMMEDIATE`
/// before deleting. Returns the count of confirmed-deleted orphans.
///
/// **This function must be called AFTER `run_case5`** so mis-tenanted points
/// are healed before this prune path could wrongly cull them.
pub async fn run_case2<S: QdrantSink>(
    pool: &SqlitePool,
    sink: &mut S,
    watermark: i64,
    collection_name: &str,
) -> Result<u64, StorageError> {
    let candidates = scan_orphan_candidates(pool, watermark).await?;
    let mut deleted = 0u64;

    for candidate in &candidates {
        if confirm_and_delete_orphan(pool, sink, candidate, collection_name).await? {
            deleted += 1;
        }
    }

    Ok(deleted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "case2_tests.rs"]
mod tests;
