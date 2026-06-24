//! Tests for `blob::file_delete` -- single-file delete from a branch (AC-F9.6).
//!
//! Covers: orphan GC, shared-blob survival (within-branch and cross-branch),
//! no-embedder invariant (AC-F9.8), and the `branches` row untouched invariant
//! (AC-F9.6-e).

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

#[derive(Default)]
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

/// Insert a blob + file + blob_ref + fts_branch_membership for the given branch.
/// Returns (blob_id, file_id).
async fn insert_file_blob(
    pool: &SqlitePool,
    branch: &str,
    ck: &str,
    pt: &str,
    path: &str,
) -> (i64, i64) {
    // Insert blob if not already present.
    sqlx::query(
        "INSERT OR IGNORE INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
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

    sqlx::query(
        "INSERT OR IGNORE INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(branch)
    .bind(path)
    .execute(pool)
    .await
    .expect("file insert");

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
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
    .expect("blob_ref insert");

    sqlx::query("INSERT OR IGNORE INTO fts_branch_membership(blob_id, branch_id) VALUES (?, ?)")
        .bind(blob_id)
        .bind(branch)
        .execute(pool)
        .await
        .expect("fts membership");

    (blob_id, file_id)
}

/// Shared fixture for the "delete file A from branch A" scenario.
///
/// Sets up three blobs:
/// - `blob_a_only`: referenced only by file A on branch A (should be orphaned).
/// - `blob_shared`: referenced by file A AND a second file B on branch A (survivor).
/// - `blob_y`: referenced by file A on branch A AND a file on branch B (survivor).
struct SharedBlobFixture {
    pool: sqlx::Pool<sqlx::Sqlite>,
    blob_a_only: i64,
    blob_shared: i64,
    blob_y: i64,
    file_a: i64,
}

/// Insert a raw blob row with the given content_key / point_id / hash.
/// Returns the new blob_id.
async fn insert_raw_blob(pool: &SqlitePool, ck: &str, pt: &str, hash: &str) -> i64 {
    sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, ?, ?, ?, 'text', X'', X'', '2024-01-01')",
    )
    .bind(ck)
    .bind(hash)
    .bind(pt)
    .bind(TENANT)
    .execute(pool)
    .await
    .unwrap();

    sqlx::query_scalar("SELECT blob_id FROM blobs WHERE content_key = ?")
        .bind(ck)
        .fetch_one(pool)
        .await
        .unwrap()
}

/// Add a blob_ref (at `chunk_index`) + fts_branch_membership tying `blob_id` to
/// `(branch, file_id)`. Does not insert the file row -- caller owns that.
async fn add_blob_ref_and_membership(
    pool: &SqlitePool,
    branch: &str,
    file_id: i64,
    blob_id: i64,
    chunk_index: i64,
) {
    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,?,?)")
        .bind(branch)
        .bind(file_id)
        .bind(chunk_index)
        .bind(blob_id)
        .execute(pool)
        .await
        .unwrap();

    sqlx::query("INSERT OR IGNORE INTO fts_branch_membership(blob_id, branch_id) VALUES (?,?)")
        .bind(blob_id)
        .bind(branch)
        .execute(pool)
        .await
        .unwrap();
}

/// Insert a standalone file row for `(branch, path)` and return its file_id.
async fn insert_file_row(pool: &SqlitePool, branch: &str, path: &str) -> i64 {
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(branch)
    .bind(path)
    .execute(pool)
    .await
    .unwrap();

    sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
        .bind(branch)
        .bind(path)
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn setup_shared_blob_fixture() -> SharedBlobFixture {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;

    // Blob 1: referenced ONLY by file A on branch A (orphan candidate).
    let (blob_a_only, file_a) =
        insert_file_blob(&fx.pool, BRANCH_A, "ck-a-only", "pt-a-only", "a.rs").await;

    // Blob 2: referenced by file A AND a separate file B on branch A (within-branch survivor).
    let blob_shared = insert_raw_blob(&fx.pool, "ck-shared-bx", "pt-shared-bx", "h2").await;
    add_blob_ref_and_membership(&fx.pool, BRANCH_A, file_a, blob_shared, 1).await;
    let file_b_a = insert_file_row(&fx.pool, BRANCH_A, "b.rs").await;
    add_blob_ref_and_membership(&fx.pool, BRANCH_A, file_b_a, blob_shared, 0).await;

    // Blob 3: referenced by file A on branch A AND a file on branch B (cross-branch survivor).
    let blob_y = insert_raw_blob(&fx.pool, "ck-shared-y", "pt-shared-y", "h3").await;
    add_blob_ref_and_membership(&fx.pool, BRANCH_A, file_a, blob_y, 2).await;
    insert_file_blob(&fx.pool, BRANCH_B, "ck-shared-y", "pt-shared-y", "y.rs").await;

    SharedBlobFixture {
        pool: fx.pool,
        blob_a_only,
        blob_shared,
        blob_y,
        file_a,
    }
}

// ---------------------------------------------------------------------------
// AC-F9.6: single-file delete -- orphan and survivor behaviour
// ---------------------------------------------------------------------------

// Only-file's blob is orphaned and removed.
#[tokio::test]
async fn delete_file_orphan_blob_removed() {
    let SharedBlobFixture {
        pool,
        blob_a_only,
        file_a,
        ..
    } = setup_shared_blob_fixture().await;
    let lm = lock_mgr();

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    delete_file_from_branch(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, file_a, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("delete_file_from_branch");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_a_only)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "blob_a_only must be deleted (orphan)");
}

// Blob shared with another file on the same branch survives.
#[tokio::test]
async fn delete_file_shared_within_branch_survives() {
    let SharedBlobFixture {
        pool,
        blob_shared,
        file_a,
        ..
    } = setup_shared_blob_fixture().await;
    let lm = lock_mgr();

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    delete_file_from_branch(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, file_a, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("delete_file_from_branch");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_shared)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1, "blob shared with file B on branch A must survive");
}

// Blob shared with another branch survives.
#[tokio::test]
async fn delete_file_shared_cross_branch_survives() {
    let SharedBlobFixture {
        pool,
        blob_y,
        file_a,
        ..
    } = setup_shared_blob_fixture().await;
    let lm = lock_mgr();

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    delete_file_from_branch(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, file_a, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("delete_file_from_branch");

    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_y)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 1, "blob shared with branch Y must survive");
}

// Exactly one Qdrant Delete op is emitted (only for the orphan blob).
#[tokio::test]
async fn delete_file_emits_one_delete_op_for_orphan() {
    let SharedBlobFixture { pool, file_a, .. } = setup_shared_blob_fixture().await;
    let lm = lock_mgr();

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    delete_file_from_branch(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, file_a, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("delete_file_from_branch");

    let deletes: Vec<&QdrantOp> = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::Delete { .. }))
        .collect();
    assert_eq!(deletes.len(), 1, "only one orphan deleted");
    match &deletes[0] {
        QdrantOp::Delete { point_id, .. } => assert_eq!(point_id, "pt-a-only"),
        _ => unreachable!(),
    }
}

// Survivor PUTs are enqueued for shared blobs.
#[tokio::test]
async fn delete_file_enqueues_survivor_puts() {
    let SharedBlobFixture { pool, file_a, .. } = setup_shared_blob_fixture().await;
    let lm = lock_mgr();

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    delete_file_from_branch(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, file_a, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("delete_file_from_branch");

    assert!(
        !batch.is_empty(),
        "survivor PUTs must be enqueued for shared blobs"
    );
}

// AC-F9.6-e: the branches row is UNTOUCHED after a single-file delete.
#[tokio::test]
async fn delete_file_branches_row_untouched() {
    let SharedBlobFixture { pool, file_a, .. } = setup_shared_blob_fixture().await;
    let lm = lock_mgr();

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();
    delete_file_from_branch(
        &pool, &lm, &mut sink, &mut batch, BRANCH_A, file_a, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("delete_file_from_branch");

    let br: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM branches WHERE branch_id = ?")
        .bind(BRANCH_A)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(br, 1, "branches row must be untouched (AC-F9.6-e)");
}

// ---------------------------------------------------------------------------
// AC-F9.8: single-file delete never calls embedder
// ---------------------------------------------------------------------------

#[tokio::test]
async fn file_delete_never_calls_embedder() {
    let fx = fixture(BRANCH_A).await;
    let lm = lock_mgr();
    let (_blob_id, file_id) = insert_file_blob(
        &fx.pool,
        BRANCH_A,
        "ck-noembed-file",
        "pt-noembed-file",
        "noembed.rs",
    )
    .await;

    let mut sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    // If embedder is called, the test panics (PanicEmbedder not needed here;
    // we simply assert no panic occurs).
    delete_file_from_branch(
        &fx.pool, &lm, &mut sink, &mut batch, BRANCH_A, file_id, TENANT, COLLECTION, COLLECTION,
    )
    .await
    .expect("file_delete must not call embedder");
}
