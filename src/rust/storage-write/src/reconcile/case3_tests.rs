//! Tests for reconcile case 3 (AC-F15.1 -- missed branch-topology event).
//!
//! Uses MockGitRefReader (no live git repo) and CaptureSink (no live Qdrant).

use super::*;
use crate::blob::ladder::{CaptureSink, QdrantOp};
use crate::blob::lock::ContentKeyLockManager;
use crate::blob::test_support::{add_branch, fixture, TENANT};
use crate::blob::vector_codec::{encode_dense, encode_sparse};
use crate::qdrant::membership_batch::MembershipPutBatch;
use crate::reconcile::seams::MockGitRefReader;
use std::collections::HashMap;

const BRANCH_A: &str = "branch-a";
const BRANCH_B: &str = "branch-b";
const COLL_ID: &str = "projects";
const COLL_NAME: &str = "projects";

/// Insert a minimal blob with no refs (orphan) in `branch_id`.
async fn insert_blob_with_ref(
    pool: &sqlx::SqlitePool,
    branch_id: &str,
    point_id: &str,
    path: &str,
) -> i64 {
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());
    let blob_id = sqlx::query(
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
    .last_insert_rowid();

    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .bind(path)
    .execute(pool)
    .await
    .expect("file insert");
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
    blob_id
}

// AC-F15.1 case 3: branch absent from git is deleted by branch_delete.
#[tokio::test]
async fn case3_deleted_git_branch_triggers_branch_delete() {
    let fx = fixture(BRANCH_A).await;
    insert_blob_with_ref(&fx.pool, BRANCH_A, "pt-deleted", "del.rs").await;

    // branch-a is gone from git.
    let git = MockGitRefReader::all_deleted();
    let lock_mgr = ContentKeyLockManager::with_defaults();
    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    let deleted = run_case3(
        &fx.pool, &lock_mgr, &mut sink, &mut batch, &git, TENANT, COLL_ID, COLL_NAME,
    )
    .await
    .expect("case3");

    assert_eq!(deleted, 1, "one deleted branch must be processed");

    // branch_delete enqueues a QdrantOp::Delete for the orphaned blob.
    let has_delete = sink
        .ops
        .iter()
        .any(|op| matches!(op, QdrantOp::Delete { .. }));
    assert!(has_delete, "branch_delete must enqueue QdrantOp::Delete");

    // Branch row must be gone.
    let branch_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM branches WHERE branch_id=?")
        .bind(BRANCH_A)
        .fetch_one(&fx.pool)
        .await
        .expect("branch count");
    assert_eq!(branch_count, 0, "deleted branch row must be removed");
}

// AC-F15.1 case 3: live branch is NOT deleted.
#[tokio::test]
async fn case3_live_git_branch_is_kept() {
    let fx = fixture(BRANCH_A).await;
    insert_blob_with_ref(&fx.pool, BRANCH_A, "pt-live", "live.rs").await;

    // branch-a still exists in git.
    let git = MockGitRefReader::with_live([BRANCH_A]);
    let lock_mgr = ContentKeyLockManager::with_defaults();
    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    let deleted = run_case3(
        &fx.pool, &lock_mgr, &mut sink, &mut batch, &git, TENANT, COLL_ID, COLL_NAME,
    )
    .await
    .expect("case3 live");

    assert_eq!(deleted, 0, "live branch must not be deleted");
    // No deletes enqueued.
    let has_delete = sink
        .ops
        .iter()
        .any(|op| matches!(op, QdrantOp::Delete { .. }));
    assert!(!has_delete);

    // Branch row must still exist.
    let branch_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM branches WHERE branch_id=?")
        .bind(BRANCH_A)
        .fetch_one(&fx.pool)
        .await
        .expect("branch count");
    assert_eq!(branch_count, 1, "live branch row must remain");
}

// Mixed: two branches, one deleted and one live.
#[tokio::test]
async fn case3_mixed_deleted_and_live_branches() {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;
    insert_blob_with_ref(&fx.pool, BRANCH_A, "pt-a-del", "a.rs").await;
    insert_blob_with_ref(&fx.pool, BRANCH_B, "pt-b-live", "b.rs").await;

    // branch-b is still in git; branch-a is gone.
    let git = MockGitRefReader::with_live([BRANCH_B]);
    let lock_mgr = ContentKeyLockManager::with_defaults();
    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    let deleted = run_case3(
        &fx.pool, &lock_mgr, &mut sink, &mut batch, &git, TENANT, COLL_ID, COLL_NAME,
    )
    .await
    .expect("case3 mixed");

    assert_eq!(deleted, 1);

    let a_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM branches WHERE branch_id=?")
        .bind(BRANCH_A)
        .fetch_one(&fx.pool)
        .await
        .expect("count a");
    let b_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM branches WHERE branch_id=?")
        .bind(BRANCH_B)
        .fetch_one(&fx.pool)
        .await
        .expect("count b");
    assert_eq!(a_count, 0, "deleted branch-a row must be gone");
    assert_eq!(b_count, 1, "live branch-b must remain");
}
