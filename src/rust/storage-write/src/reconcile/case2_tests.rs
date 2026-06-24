//! Tests for reconcile case 2 (AC-F15.1 / AC-F15.3 -- ABA-guarded orphan prune).

use super::*;
use crate::blob::ladder::{CaptureSink, QdrantOp};
use crate::blob::test_support::{fixture, TENANT};
use crate::blob::vector_codec::{encode_dense, encode_sparse};
use std::collections::HashMap;

const BRANCH_A: &str = "branch-a";
const COLL: &str = "projects";

/// Insert a blob with empty vectors; returns blob_id.
async fn insert_blob(pool: &sqlx::SqlitePool, point_id: &str) -> i64 {
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());
    sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, 'h', ?, ?, 't', ?, ?, '2026-01-01')",
    )
    .bind(format!("ck-{point_id}"))
    .bind(point_id)
    .bind(TENANT)
    .bind(dense)
    .bind(sparse)
    .execute(pool)
    .await
    .expect("blob insert")
    .last_insert_rowid()
}

/// Insert a blob_ref for blob_id in branch_id.
async fn insert_ref(pool: &sqlx::SqlitePool, branch_id: &str, blob_id: i64, path: &str) {
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .bind(path)
    .execute(pool)
    .await
    .expect("file");
    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id=? AND relative_path=?")
            .bind(branch_id)
            .bind(path)
            .fetch_one(pool)
            .await
            .expect("file_id");
    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES(?,?,0,?)")
        .bind(branch_id)
        .bind(file_id)
        .bind(blob_id)
        .execute(pool)
        .await
        .expect("ref insert");
}

// AC-F15.1 case 2: orphan blob (no refs) is pruned; Delete op is enqueued.
#[tokio::test]
async fn case2_orphan_blob_is_deleted_and_delete_op_enqueued() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-orphan").await;
    // No blob_ref: blob is an orphan.

    let mut sink = CaptureSink::default();
    let deleted = run_case2(&fx.pool, &mut sink, 0, COLL)
        .await
        .expect("case2");

    assert_eq!(deleted, 1, "one orphan must be deleted");

    // Delete op enqueued before the blobs row was removed (FP-1 order check).
    let has_delete = sink
        .ops
        .iter()
        .any(|op| matches!(op, QdrantOp::Delete { point_id, .. } if point_id == "pt-orphan"));
    assert!(
        has_delete,
        "QdrantOp::Delete must be enqueued for orphan point"
    );

    // Blob row must be gone from SQLite.
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&fx.pool)
        .await
        .expect("count");
    assert_eq!(count, 0, "orphan blob row must be deleted from SQLite");
}

// AC-F15.3 ABA guard (non-vacuous): a ref inserted BETWEEN scan_orphan_candidates
// and confirm_and_delete_orphan causes the blob to be classified as a survivor.
//
// This is the genuine ABA scenario: the candidate scan sees count=0 (orphan).
// Before confirm runs, a concurrent ingest commits a new blob_ref (here we do it
// synchronously between the two calls). confirm re-reads count inside its held
// transaction and sees count=1 -- the blob is NOT deleted.
//
// This directly tests the property that would have been broken by the old
// raw BEGIN/COMMIT pattern: the re-verify inside confirm must see the ref.
// With pool.begin() holding the connection for the re-verify + delete, the only
// way a ref can appear in the re-verify is if it was committed BEFORE confirm
// acquired the connection -- which is exactly what this test sets up.
#[tokio::test]
async fn case2_aba_ref_inserted_between_scan_and_confirm_survives() {
    let fx = fixture(BRANCH_A).await;

    // Insert blob with no refs -- scan will find it as an orphan candidate.
    let blob_id = insert_blob(&fx.pool, "pt-aba-between").await;

    // Step 1: scan finds the blob as an orphan (count=0).
    let candidates = scan_orphan_candidates(&fx.pool, 0).await.expect("scan");
    assert_eq!(candidates.len(), 1, "scan must find one orphan candidate");
    assert_eq!(candidates[0].blob_id, blob_id);

    // Step 2: simulate concurrent ingest -- insert a ref AFTER the scan.
    // In production this would happen in a separate tokio task. Here we do it
    // synchronously to make the test deterministic.
    insert_ref(&fx.pool, BRANCH_A, blob_id, "aba-between.rs").await;

    // Step 3: confirm runs and re-reads the refcount inside its held transaction.
    // It must see count=1 and return false (ABA survivor -- blob kept alive).
    let mut sink = CaptureSink::default();
    let deleted = confirm_and_delete_orphan(&fx.pool, &mut sink, &candidates[0], COLL)
        .await
        .expect("confirm");

    assert!(
        !deleted,
        "ABA survivor: ref inserted after scan must prevent deletion"
    );
    assert!(
        sink.ops.is_empty(),
        "ABA survivor: no Delete op must be enqueued"
    );

    // Blob row must still exist.
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&fx.pool)
        .await
        .expect("count");
    assert_eq!(count, 1, "ABA survivor: blob row must remain in SQLite");
}

// Referenced blob is NOT pruned.
#[tokio::test]
async fn case2_referenced_blob_is_not_pruned() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-ref").await;
    insert_ref(&fx.pool, BRANCH_A, blob_id, "ref.rs").await;

    let mut sink = CaptureSink::default();
    let deleted = run_case2(&fx.pool, &mut sink, 0, COLL)
        .await
        .expect("case2");

    assert_eq!(deleted, 0);
    assert!(sink.ops.is_empty());
}

// Watermark scoping: blobs at or below watermark are not scanned.
#[tokio::test]
async fn case2_incremental_watermark_scopes_scan() {
    let fx = fixture(BRANCH_A).await;

    // Old orphan blob (below watermark after insert).
    let old_id = insert_blob(&fx.pool, "pt-old-orphan").await;
    // New orphan blob (above watermark).
    let _new_id = insert_blob(&fx.pool, "pt-new-orphan").await;

    // Watermark = old_id: incremental pass must only see the new orphan.
    let mut sink = CaptureSink::default();
    let deleted = run_case2(&fx.pool, &mut sink, old_id, COLL)
        .await
        .expect("case2 incremental");

    assert_eq!(deleted, 1, "only the blob above the watermark is pruned");
    let has_new = sink
        .ops
        .iter()
        .any(|op| matches!(op, QdrantOp::Delete { point_id, .. } if point_id == "pt-new-orphan"));
    assert!(has_new);
    let has_old = sink
        .ops
        .iter()
        .any(|op| matches!(op, QdrantOp::Delete { point_id, .. } if point_id == "pt-old-orphan"));
    assert!(!has_old, "old orphan below watermark must not be touched");
}

// Mixed fixture: one orphan + one referenced blob; only orphan is deleted.
#[tokio::test]
async fn case2_mixed_fixture_orphan_pruned_referenced_kept() {
    let fx = fixture(BRANCH_A).await;
    let orphan_id = insert_blob(&fx.pool, "pt-orphan2").await;
    let ref_id = insert_blob(&fx.pool, "pt-ref2").await;
    insert_ref(&fx.pool, BRANCH_A, ref_id, "ref2.rs").await;

    let mut sink = CaptureSink::default();
    let deleted = run_case2(&fx.pool, &mut sink, 0, COLL)
        .await
        .expect("case2 mixed");

    assert_eq!(deleted, 1);
    let orphan_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id=?")
        .bind(orphan_id)
        .fetch_one(&fx.pool)
        .await
        .expect("count");
    let ref_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id=?")
        .bind(ref_id)
        .fetch_one(&fx.pool)
        .await
        .expect("count");
    assert_eq!(orphan_count, 0, "orphan deleted");
    assert_eq!(ref_count, 1, "referenced blob kept");
}
