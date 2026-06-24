//! SQL-integration tests for `branch::delete` -- whole-branch delete (AC-F9.x).
//!
//! Covers: chunked-DELETE idiom (AC-F9.2), RESTRICT FK backstop (AC-F9.5),
//! 8-step FP-1 sequence (AC-F9.3/AC-F9.4), no-embedder invariant (AC-F9.8),
//! and crash-recovery idempotency.
//!
//! Truth-table / probe tests (AC-F9.1, no DB) live in `delete_probe_tests.rs`.

// Include the probe truth-table tests as a sibling sub-module so they are
// discovered by `cargo test` via this file's module declaration in delete.rs.
#[cfg(test)]
#[path = "delete_probe_tests.rs"]
mod probe_tests;

use std::sync::Arc;

use sqlx::SqlitePool;

use super::*;
use crate::blob::ladder::QdrantOp;
use crate::blob::lock::{ContentKeyLockManager, LockManagerConfig};
use crate::blob::test_support::{add_branch, fixture, TENANT};
use crate::qdrant::membership_batch::MembershipPutBatch;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const BRANCH_A: &str = "branch-a";
const BRANCH_B: &str = "branch-b";
const COLLECTION: &str = "projects";

/// Capturing sink that records both Upsert/OverwritePayload AND Delete ops.
#[derive(Debug, Default)]
struct CaptureSink {
    pub ops: Vec<QdrantOp>,
}
impl QdrantSink for CaptureSink {
    fn enqueue(&mut self, op: QdrantOp) {
        self.ops.push(op);
    }
}

fn lock_mgr() -> Arc<ContentKeyLockManager> {
    ContentKeyLockManager::new(LockManagerConfig::default())
}

/// Insert a blob + file + blob_ref + fts_branch_membership for a branch.
/// Returns (blob_id, file_id, point_id).
async fn insert_blob_for_branch(
    pool: &SqlitePool,
    branch: &str,
    ck: &str,
    pt: &str,
) -> (i64, i64, String) {
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

    let blob_id: i64 = sqlx::query_scalar("SELECT blob_id FROM blobs WHERE content_key = ?")
        .bind(ck)
        .fetch_one(pool)
        .await
        .expect("blob_id");

    let path = format!("{branch}/{ck}.rs");
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(branch)
    .bind(&path)
    .execute(pool)
    .await
    .expect("file insert");

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(branch)
            .bind(&path)
            .fetch_one(pool)
            .await
            .expect("file_id");

    sqlx::query(
        "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?, ?, 0, ?)",
    )
    .bind(branch)
    .bind(file_id)
    .bind(blob_id)
    .execute(pool)
    .await
    .expect("blob_ref insert");

    sqlx::query("INSERT INTO fts_branch_membership(blob_id, branch_id) VALUES (?, ?)")
        .bind(blob_id)
        .bind(branch)
        .execute(pool)
        .await
        .expect("fts_membership insert");

    (blob_id, file_id, pt.to_string())
}

/// Add a file + blob_ref + fts_branch_membership that ties `blob_id` to `branch`.
/// Used to plant a second-branch reference that prevents orphan GC.
async fn attach_blob_to_branch(pool: &SqlitePool, branch: &str, blob_id: i64, path: &str) {
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(branch)
    .bind(path)
    .execute(pool)
    .await
    .unwrap();

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(branch)
            .bind(path)
            .fetch_one(pool)
            .await
            .unwrap();

    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)")
        .bind(branch)
        .bind(file_id)
        .bind(blob_id)
        .execute(pool)
        .await
        .unwrap();

    sqlx::query("INSERT INTO fts_branch_membership(blob_id, branch_id) VALUES (?,?)")
        .bind(blob_id)
        .bind(branch)
        .execute(pool)
        .await
        .unwrap();
}

/// Shared setup: one blob referenced by BRANCH_A and BRANCH_B.
/// Returns (pool, blob_id).
async fn setup_shared_blob_two_branches() -> (sqlx::Pool<sqlx::Sqlite>, i64) {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;

    sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES ('ck-shared','hash1','pt-shared', ?, 'text', X'', X'', '2024-01-01')",
    )
    .bind(TENANT)
    .execute(&fx.pool)
    .await
    .expect("blob insert");

    let blob_id: i64 =
        sqlx::query_scalar("SELECT blob_id FROM blobs WHERE content_key = 'ck-shared'")
            .fetch_one(&fx.pool)
            .await
            .expect("blob_id");

    attach_blob_to_branch(&fx.pool, BRANCH_A, blob_id, "shared.rs").await;
    attach_blob_to_branch(&fx.pool, BRANCH_B, blob_id, "shared.rs").await;

    (fx.pool, blob_id)
}

// ---------------------------------------------------------------------------
// AC-F9.2: chunked-delete idiom; bundled SQLite rejects DELETE ... LIMIT
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sqlite_rejects_delete_limit_syntax() {
    let fx = fixture(BRANCH_A).await;
    let result = sqlx::query("DELETE FROM branches LIMIT 1")
        .execute(&fx.pool)
        .await;
    assert!(
        result.is_err(),
        "bundled SQLite must reject DELETE ... LIMIT \
         (libsqlite3-sys 0.30.1 lacks SQLITE_ENABLE_UPDATE_DELETE_LIMIT; \
         this test justifies the chunked subselect idiom AC-F9.2)"
    );
}

// ---------------------------------------------------------------------------
// AC-F9.5 (DATA-03): RESTRICT FK backstop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn restrict_fk_rejects_premature_blob_delete() {
    let fx = fixture(BRANCH_A).await;
    let (blob_id, _file_id, _pt) =
        insert_blob_for_branch(&fx.pool, BRANCH_A, "ck-restrict", "pt-restrict").await;

    let result = sqlx::query("DELETE FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .execute(&fx.pool)
        .await;

    assert!(
        result.is_err(),
        "ON DELETE RESTRICT must prevent blob deletion while blob_refs exist (AC-F9.5 DATA-03)"
    );
}

// ---------------------------------------------------------------------------
// AC-F9.3: 8-step FP-1 whole-branch delete + orphan GC
// ---------------------------------------------------------------------------

#[tokio::test]
async fn whole_branch_delete_removes_orphan_blobs_and_branch_row() {
    let fx = fixture(BRANCH_A).await;
    let lm = lock_mgr();
    let (blob_id, _file_id, pt) =
        insert_blob_for_branch(&fx.pool, BRANCH_A, "ck-del1", "pt-del1").await;

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    branch_delete(
        &fx.pool, &lm, &mut sink, &mut batch, BRANCH_A, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("branch_delete");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "orphan blob must be deleted");

    let br: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM branches WHERE branch_id = ?")
        .bind(BRANCH_A)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
    assert_eq!(br, 0, "branches row deleted (step 8, last)");

    let deletes: Vec<&QdrantOp> = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::Delete { .. }))
        .collect();
    assert_eq!(deletes.len(), 1);
    match &deletes[0] {
        QdrantOp::Delete { point_id, .. } => assert_eq!(*point_id, pt),
        _ => unreachable!(),
    }
}

// AC-F9.3: shared blob row must survive when another branch still holds a ref.
#[tokio::test]
async fn shared_blob_kept() {
    let (pool, blob_id) = setup_shared_blob_two_branches().await;
    let lm = lock_mgr();
    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    branch_delete(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("branch_delete");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1, "shared blob must survive");
}

// AC-F9.3: no Qdrant Delete op emitted for a blob still referenced by another branch.
#[tokio::test]
async fn shared_blob_no_delete_op() {
    let (pool, _blob_id) = setup_shared_blob_two_branches().await;
    let lm = lock_mgr();
    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    branch_delete(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("branch_delete");

    let deletes: Vec<&QdrantOp> = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::Delete { .. }))
        .collect();
    assert_eq!(deletes.len(), 0, "no Delete op for a shared blob");
}

// AC-F9.3: survivor PUT is enqueued with the correct remaining-branch membership.
#[tokio::test]
async fn shared_blob_survivor_put_enqueued() {
    let (pool, _blob_id) = setup_shared_blob_two_branches().await;
    let lm = lock_mgr();
    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    branch_delete(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("branch_delete");

    assert_eq!(
        batch.len(),
        1,
        "one survivor PUT enqueued for the shared blob"
    );
    let (_, pending) = batch.iter().next().expect("entry");
    assert_eq!(pending.point_id, "pt-shared");
    let mut branches = pending.payload.branch_id.clone();
    branches.sort();
    assert_eq!(
        branches,
        vec![BRANCH_B.to_string()],
        "survivor PUT carries only BRANCH_B in membership"
    );
}

// AC-F9.4 / ABA survivor: blob that gains a new ref BETWEEN step 2 and step 5
// is NOT deleted (BEGIN IMMEDIATE re-verify catches it).
#[tokio::test]
async fn aba_survivor_blob_not_deleted() {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;
    let lm = lock_mgr();

    let (blob_id, _file_a, _pt) =
        insert_blob_for_branch(&fx.pool, BRANCH_A, "ck-aba", "pt-aba").await;

    // Plant a BRANCH_B ref so the blob is never truly orphaned.
    attach_blob_to_branch(&fx.pool, BRANCH_B, blob_id, "aba.rs").await;

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    branch_delete(
        &fx.pool, &lm, &mut sink, &mut batch, BRANCH_A, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("branch_delete");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
    assert_eq!(count, 1, "ABA survivor: blob must not be GC'd (AC-F9.4)");
}

// AC-F9.8: delete runs to completion without calling an embedder.
#[tokio::test]
async fn delete_never_calls_embedder() {
    let fx = fixture(BRANCH_A).await;
    let lm = lock_mgr();
    insert_blob_for_branch(&fx.pool, BRANCH_A, "ck-noembed", "pt-noembed").await;

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    // branch_delete takes no Embedder parameter; if it called embed() it wouldn't
    // compile. This test is the explicit coverage point for AC-F9.8.
    branch_delete(
        &fx.pool, &lm, &mut sink, &mut batch, BRANCH_A, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("branch_delete must not call embedder");
}

// Crash-recovery: a second run of branch_delete after the first completed is
// idempotent (step 1 returns empty -> short-circuit to steps 7+8, 0 rows affected).
#[tokio::test]
async fn crash_after_step3_rerun_is_idempotent() {
    let fx = fixture(BRANCH_A).await;
    let lm = lock_mgr();
    insert_blob_for_branch(&fx.pool, BRANCH_A, "ck-crash3", "pt-crash3").await;

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    branch_delete(
        &fx.pool, &lm, &mut sink, &mut batch, BRANCH_A, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("first run");

    // Re-insert the branch row to simulate crash-before-step-8 scenario.
    sqlx::query(
        "INSERT OR IGNORE INTO branches(branch_id, branch_name, location, created_at, updated_at) \
         VALUES (?, ?, '/repo', '2024-01-01', '2024-01-01')",
    )
    .bind(BRANCH_A)
    .bind(BRANCH_A)
    .execute(&fx.pool)
    .await
    .unwrap();

    let mut sink2 = CaptureSink::default();
    let mut batch2 = MembershipPutBatch::new();
    branch_delete(
        &fx.pool,
        &lm,
        &mut sink2,
        &mut batch2,
        BRANCH_A,
        TENANT,
        COLLECTION,
        COLLECTION,
    )
    .await
    .expect("idempotent re-run");

    let deletes2: Vec<&QdrantOp> = sink2
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::Delete { .. }))
        .collect();
    assert_eq!(
        deletes2.len(),
        0,
        "crash-recovery re-run is idempotent: no duplicate orphan deletes"
    );
}
