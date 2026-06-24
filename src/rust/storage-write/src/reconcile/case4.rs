//! Case 4 -- FTS branch-membership drift (arch §4.7 case 4, AC-F15.1 / AC-F15.4).
//!
//! File: `wqm-storage-write/src/reconcile/case4.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: Reconcile case 4 fixes drift where `blob_refs` has a `(blob_id, branch_id)`
//!   row but `fts_branch_membership` lacks the corresponding entry. Root cause: crash
//!   after `blob_refs` write but before `fts_branch_membership` write in §4.1.
//!
//!   Fix (additive-first, AC-F15.4 watermark-scoped):
//!     1. Scan `blob_refs` rows with `blob_id > watermark` (incremental window).
//!     2. For each `(blob_id, branch_id)` pair, INSERT INTO `fts_branch_membership`
//!        ON CONFLICT IGNORE -- harmless if the row already exists.
//!     3. Returns the maximum `blob_id` touched for watermark advancement.
//!
//!   In FULL mode (`watermark = 0`) the scan covers all `blob_refs` rows.
//!   The query is keyset-paginated by `blob_id` (ascending order) so the
//!   watermark advancement tracks exactly the highest blob_id processed.
//!
//! Neighbors: [`super::watermark`] (watermark advancement), schema `fts.rs`
//!   (`fts_branch_membership` table definition).

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

/// One `(blob_id, branch_id)` pair from `blob_refs` in the scan window.
struct BlobRefRow {
    blob_id: i64,
    branch_id: String,
}

/// Fetch `(blob_id, branch_id)` pairs above the watermark from `blob_refs`.
async fn fetch_refs_above_watermark(
    pool: &SqlitePool,
    watermark: i64,
) -> Result<Vec<BlobRefRow>, StorageError> {
    let rows = sqlx::query(
        "SELECT DISTINCT blob_id, branch_id FROM blob_refs \
         WHERE blob_id > ? ORDER BY blob_id, branch_id",
    )
    .bind(watermark)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("case4 fetch_refs: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| BlobRefRow {
            blob_id: r.get("blob_id"),
            branch_id: r.get("branch_id"),
        })
        .collect())
}

/// Reconcile case 4: repair FTS branch-membership drift.
///
/// For every `(blob_id, branch_id)` pair in `blob_refs` above the watermark,
/// inserts into `fts_branch_membership ON CONFLICT IGNORE`. This is purely
/// additive (FP-1) and idempotent: existing rows are silently skipped.
///
/// Returns the maximum `blob_id` processed (for watermark advancement), or
/// `watermark` if no rows were found in the window.
pub async fn run_case4(pool: &SqlitePool, watermark: i64) -> Result<i64, StorageError> {
    let refs = fetch_refs_above_watermark(pool, watermark).await?;
    let mut max_blob_id = watermark;

    for row in &refs {
        if row.blob_id > max_blob_id {
            max_blob_id = row.blob_id;
        }
        sqlx::query(
            "INSERT INTO fts_branch_membership(blob_id, branch_id) \
             VALUES (?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(row.blob_id)
        .bind(&row.branch_id)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("case4 fts insert: {e}")))?;
    }

    Ok(max_blob_id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "case4_tests.rs"]
mod tests;
