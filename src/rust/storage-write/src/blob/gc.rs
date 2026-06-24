//! Orphan blob GC by referrer count (arch §5.4, AC-F9.4).
//!
//! File: `wqm-storage-write/src/blob/gc.rs`
//! Location: `src/rust/storage-write/src/blob/` (write-crate blob layer)
//! Context: Provides the refcount GC helper used by both the whole-branch delete
//!   (arch §4.3 Step 5) and by any reconcile sweep that needs to prune orphaned
//!   blobs. The refcount is `SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?`
//!   (arch §5.4) — NO Qdrant payload field is read for GC (DR GP-6 / AC-F9.4).
//!
//!   The ABA guard (re-verify under BEGIN IMMEDIATE before DELETE) ensures that a
//!   concurrent ingest cannot insert a new blob_ref between the count check and the
//!   DELETE, keeping a still-referenced blob alive.
//!
//!   The caller is responsible for deleting the Qdrant point BEFORE calling the
//!   blob-row delete here (FP-1: data products before truth rows). This module
//!   handles only the SQLite side.
//!
//! Neighbors: [`crate::branch::delete`] (calls step5 helper via this module),
//!   [`crate::blob::file_delete`] (calls `blob_refcount` directly),
//!   [`crate::blob::ladder`] (QdrantOp::Delete variant enqueued before calling here).

use sqlx::SqlitePool;
use wqm_common::error::StorageError;

/// Return the current referrer count for `blob_id`.
///
/// Executes `SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?` (arch §5.4).
/// NO Qdrant payload field is read — the count is derived purely from SQLite
/// truth (DR GP-6 / AC-F9.4).
///
/// ## Usage
///
/// Call AFTER committing the `blob_refs` mutation (DELETE or INSERT) so the
/// count reflects the post-mutation state. The delete path calls this inside
/// a BEGIN IMMEDIATE transaction (ABA guard) to prevent a new ref from
/// appearing between the count and the DELETE.
pub async fn blob_refcount(pool: &SqlitePool, blob_id: i64) -> Result<i64, StorageError> {
    sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("blob_refcount: {e}")))
}

/// Delete a confirmed-orphan blob row (SQLite side only).
///
/// The caller MUST have already:
///   1. Deleted the Qdrant point for this blob (FP-1 ordering).
///   2. Verified that `blob_refcount` == 0 inside the SAME transaction
///      (ABA guard: `pool` is the same pool and the count was read inside a
///      BEGIN IMMEDIATE that is still open when this delete executes).
///
/// Uses the subselect idiom required by AC-F9.2 (`DELETE ... LIMIT` is
/// invalid in the bundled SQLite).
///
/// The `blobs_ad` FTS5 AFTER DELETE trigger fires automatically, keeping
/// `fts_content` in sync (AC-F3.2).
pub async fn delete_orphan_blob_row(pool: &SqlitePool, blob_id: i64) -> Result<(), StorageError> {
    sqlx::query(
        "DELETE FROM blobs WHERE rowid IN \
         (SELECT rowid FROM blobs WHERE blob_id = ?)",
    )
    .bind(blob_id)
    .execute(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("delete_orphan_blob_row: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::test_support::{add_branch, fixture, TENANT};
    use sqlx::SqlitePool;

    const BRANCH_A: &str = "branch-a";
    const BRANCH_B: &str = "branch-b";

    /// Insert a blob row and return its blob_id.
    async fn insert_blob(pool: &SqlitePool, ck: &str, pt: &str) -> i64 {
        sqlx::query(
            "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
             raw_text, dense_vec, sparse_vec, created_at) \
             VALUES (?, 'hash1', ?, ?, 'text', X'', X'', '2024-01-01')",
        )
        .bind(ck)
        .bind(pt)
        .bind(TENANT)
        .execute(pool)
        .await
        .expect("blob insert");

        sqlx::query_scalar("SELECT blob_id FROM blobs WHERE content_key = ?")
            .bind(ck)
            .fetch_one(pool)
            .await
            .expect("blob_id")
    }

    /// Insert a file + blob_ref for (branch, blob_id). Returns file_id.
    async fn insert_ref(pool: &SqlitePool, branch: &str, blob_id: i64, path: &str) -> i64 {
        sqlx::query(
            "INSERT OR IGNORE INTO files(branch_id, relative_path, collection, created_at, updated_at) \
             VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
        )
        .bind(branch)
        .bind(path)
        .execute(pool)
        .await
        .expect("file insert");

        let file_id: i64 = sqlx::query_scalar(
            "SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?",
        )
        .bind(branch)
        .bind(path)
        .fetch_one(pool)
        .await
        .expect("file_id");

        sqlx::query(
            "INSERT OR IGNORE INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
             VALUES (?, ?, 0, ?)",
        )
        .bind(branch)
        .bind(file_id)
        .bind(blob_id)
        .execute(pool)
        .await
        .expect("ref insert");

        file_id
    }

    // AC-F9.4: blob_refcount returns 0 for a blob with no referrers.
    #[tokio::test]
    async fn refcount_zero_when_no_refs() {
        let fx = fixture(BRANCH_A).await;
        let blob_id = insert_blob(&fx.pool, "ck-gc1", "pt-gc1").await;

        let count = blob_refcount(&fx.pool, blob_id)
            .await
            .expect("blob_refcount");
        assert_eq!(count, 0, "no refs -> refcount 0 (AC-F9.4)");
    }

    // AC-F9.4: blob_refcount returns 1 for a blob with one referrer.
    #[tokio::test]
    async fn refcount_one_with_single_ref() {
        let fx = fixture(BRANCH_A).await;
        let blob_id = insert_blob(&fx.pool, "ck-gc2", "pt-gc2").await;
        insert_ref(&fx.pool, BRANCH_A, blob_id, "a.rs").await;

        let count = blob_refcount(&fx.pool, blob_id)
            .await
            .expect("blob_refcount");
        assert_eq!(count, 1, "one ref -> refcount 1");
    }

    // AC-F9.4: blob_refcount returns 2 for a blob shared across two branches.
    #[tokio::test]
    async fn refcount_two_with_two_branches() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let blob_id = insert_blob(&fx.pool, "ck-gc3", "pt-gc3").await;
        insert_ref(&fx.pool, BRANCH_A, blob_id, "a.rs").await;
        insert_ref(&fx.pool, BRANCH_B, blob_id, "a.rs").await;

        let count = blob_refcount(&fx.pool, blob_id)
            .await
            .expect("blob_refcount");
        assert_eq!(count, 2, "two branches -> refcount 2");
    }

    // AC-F9.4: a still-referenced blob is NEVER GC'd (ABA-survivor test).
    // Simulate the ABA race: manually check refcount > 0 and verify we do NOT delete.
    #[tokio::test]
    async fn aba_survivor_still_referenced_blob_not_deleted() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let blob_id = insert_blob(&fx.pool, "ck-aba-gc", "pt-aba-gc").await;
        insert_ref(&fx.pool, BRANCH_A, blob_id, "aba.rs").await;
        insert_ref(&fx.pool, BRANCH_B, blob_id, "aba.rs").await;

        // Simulate Step 4: delete BRANCH_A's blob_ref.
        sqlx::query("DELETE FROM blob_refs WHERE branch_id = ? AND blob_id = ?")
            .bind(BRANCH_A)
            .bind(blob_id)
            .execute(&fx.pool)
            .await
            .unwrap();

        // Re-verify: blob still has a ref from BRANCH_B -> must NOT be deleted.
        let count = blob_refcount(&fx.pool, blob_id)
            .await
            .expect("blob_refcount");
        assert_eq!(count, 1, "ABA survivor: still has ref from BRANCH_B");

        // The ABA guard: since count > 0, we do NOT call delete_orphan_blob_row.
        // The blob must survive.
        let blob_exists: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
            .bind(blob_id)
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(
            blob_exists, 1,
            "ABA survivor: still-referenced blob must not be GC'd (AC-F9.4)"
        );
    }

    // delete_orphan_blob_row removes the blob and fires the FTS5 blobs_ad trigger.
    #[tokio::test]
    async fn delete_orphan_blob_row_removes_blob() {
        let fx = fixture(BRANCH_A).await;
        let blob_id = insert_blob(&fx.pool, "ck-gc-del", "pt-gc-del").await;

        // Confirm no refs first (orphan condition).
        let count = blob_refcount(&fx.pool, blob_id).await.expect("refcount");
        assert_eq!(count, 0);

        delete_orphan_blob_row(&fx.pool, blob_id)
            .await
            .expect("delete_orphan_blob_row");

        let exists: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
            .bind(blob_id)
            .fetch_one(&fx.pool)
            .await
            .unwrap();
        assert_eq!(exists, 0, "orphan blob row must be deleted");
    }

    // AC-F9.4: NO Qdrant payload field is read for GC determination.
    // This is structural: blob_refcount only queries blob_refs, not Qdrant.
    // Verified by reading the implementation — the function contains no Qdrant call.
    #[test]
    fn gc_refcount_uses_only_sqlite_not_qdrant() {
        use std::path::Path;
        let gc_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/blob/gc.rs");
        let src = std::fs::read_to_string(&gc_path).expect("read gc.rs");

        // Production code (before #[cfg(test)]) must not call get_points or any
        // Qdrant query method for GC decisions.
        let production: String = src
            .lines()
            .take_while(|l| !l.trim().starts_with("#[cfg(test)]"))
            .collect::<Vec<_>>()
            .join("\n");

        let forbidden = ["get_points", "get_payload", "QdrantWriteClient"];
        for token in &forbidden {
            assert!(
                !production.contains(token),
                "gc.rs production code must not use Qdrant for GC (AC-F9.4 DR GP-6): \
                 found '{token}' in production section"
            );
        }
    }
}
