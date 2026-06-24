//! AC-F8.6 and AC-F8.7 rename tests (codesize split from onboard_tests.rs).
//!
//! AC-F8.6: same-content rename with panicking embedder -- must take HIT path.
//! AC-F8.7: crash-between-halves invariant -- blob row survives; reconcile converges.

use std::sync::atomic::Ordering;

// `super::*` brings everything from onboard_tests.rs's `use super::*` (which
// imports from onboard.rs), so apply_git_diff, apply_one_change, etc. are in scope.
use super::*;
use crate::blob::file_delete::delete_file_from_branch;
use crate::blob::ladder::CaptureSink;
use crate::blob::test_support::{fixture, TENANT};
use crate::qdrant::membership_batch::MembershipPutBatch;
use wqm_common::git::file_change::{FileChange, FileChangeStatus};

// Re-use test doubles from sibling module (onboard_tests.rs).
use super::{lock_mgr, seed_branch_files, MockEmbedder, MockProvider, PanickingEmbedder};

// ---------------------------------------------------------------------------
// AC-F8.6: same-content RENAME = zero re-embed (panicking embedder must be silent)
// ---------------------------------------------------------------------------

/// Setup half: onboard old.rs so the blob exists in the DB.
async fn setup_rename_test(branch_id: &str) -> crate::blob::test_support::Fixture {
    let f = fixture(branch_id).await;
    let locks = lock_mgr();
    seed_branch_files(&f.pool, &locks, branch_id, &["old.rs:shared content"]).await;
    f
}

/// Assert half: blob still exists and refcount >= 1 after rename.
async fn assert_blob_survived_rename(pool: &sqlx::SqlitePool) {
    let blobs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs")
        .fetch_one(pool)
        .await
        .unwrap();
    assert_eq!(blobs, 1, "blob must survive same-content rename");
    let refs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs")
        .fetch_one(pool)
        .await
        .unwrap();
    assert!(
        refs >= 1,
        "blob refcount must stay >= 1 after rename (DOM-02)"
    );
}

#[tokio::test]
async fn test_rename_same_content_no_reembed() {
    let f = setup_rename_test("br-e").await;
    let pool = &f.pool;

    // Verify pre-condition: exactly one blob exists from setup.
    let pre_blobs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs")
        .fetch_one(pool)
        .await
        .unwrap();
    assert_eq!(pre_blobs, 1, "setup: expected one blob before rename");

    // Apply the rename with identical content via a panicking embedder.
    // ingest(new) must take the HIT path (blob already exists), so embed() is never called.
    let mut rename_provider = MockProvider::new();
    rename_provider.add_file("new.rs", vec![(0, "shared content")]); // same text as old.rs

    let rename_diff = vec![FileChange {
        status: FileChangeStatus::Renamed {
            old_path: "old.rs".into(),
            similarity: 100,
        },
        path: "new.rs".into(),
    }];

    let locks = lock_mgr();
    let rename_embedder = PanickingEmbedder;
    let mut rename_sink = CaptureSink::default();
    let mut batch = MembershipPutBatch::new();

    // If PanickingEmbedder::embed() is called, the test panics here -- that IS the assertion.
    apply_git_diff(
        pool,
        &locks,
        &rename_embedder,
        &mut rename_sink,
        &mut batch,
        &rename_provider,
        TENANT,
        "br-e",
        "projects",
        "projects",
        &rename_diff,
    )
    .await
    .unwrap();

    assert_blob_survived_rename(pool).await;
}

// ---------------------------------------------------------------------------
// AC-F8.7: crash-between-halves -- blob row survives; delete-old completes cleanly
// ---------------------------------------------------------------------------

/// Setup half: onboard old.rs and return (fixture, blob_id).
async fn setup_crash_test(branch_id: &str) -> (crate::blob::test_support::Fixture, i64) {
    let f = fixture(branch_id).await;
    let locks = lock_mgr();
    seed_branch_files(&f.pool, &locks, branch_id, &["old.rs:crash test content"]).await;
    let blob_id: i64 = sqlx::query_scalar("SELECT blob_id FROM blobs LIMIT 1")
        .fetch_one(&f.pool)
        .await
        .unwrap();
    (f, blob_id)
}

/// Execute only the ingest(new) half -- simulates crash before delete(old).
async fn simulate_crash_after_ingest(
    pool: &sqlx::SqlitePool,
    branch_id: &str,
    new_path: &str,
    content: &str,
) -> (i64, usize) {
    let mut crash_provider = MockProvider::new();
    crash_provider.add_file(new_path, vec![(0, content)]);
    let (crash_embedder, crash_calls) = MockEmbedder::new();
    let locks = lock_mgr();
    let mut crash_sink = CaptureSink::default();
    let ingest_change = FileChange {
        status: FileChangeStatus::Added,
        path: new_path.into(),
    };
    let mut batch = MembershipPutBatch::new();
    apply_one_change(
        pool,
        &locks,
        crash_embedder.as_ref(),
        &mut crash_sink,
        &mut batch,
        &crash_provider,
        TENANT,
        branch_id,
        "projects",
        "projects",
        &ingest_change,
    )
    .await
    .unwrap();
    let calls = crash_calls.load(Ordering::Acquire);
    (batch.len() as i64, calls)
}

/// Complete the delete(old) half -- reconcile step.
async fn complete_delete_old(pool: &sqlx::SqlitePool, branch_id: &str, old_path: &str) {
    let old_fid: Option<i64> =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(branch_id)
            .bind(old_path)
            .fetch_optional(pool)
            .await
            .unwrap();

    if let Some(fid) = old_fid {
        let locks = lock_mgr();
        let mut del_sink = CaptureSink::default();
        let mut del_batch = MembershipPutBatch::new();
        delete_file_from_branch(
            pool,
            &locks,
            &mut del_sink,
            &mut del_batch,
            branch_id,
            fid,
            TENANT,
            "projects",
            "projects",
        )
        .await
        .unwrap();
    }
}

#[tokio::test]
async fn test_crash_between_halves_blob_survives() {
    let (f, blob_id) = setup_crash_test("br-f").await;
    let pool = &f.pool;

    // Simulate crash: only ingest(new) runs, delete(old) is skipped.
    let (_batch_len, embed_calls) =
        simulate_crash_after_ingest(pool, "br-f", "new.rs", "crash test content").await;

    // Same content -> HIT path: embed must NOT be called.
    assert_eq!(
        embed_calls, 0,
        "same-content ingest(new) must use HIT path, no re-embed"
    );

    // blob row must still exist post-crash.
    let mid_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(pool)
        .await
        .unwrap();
    assert_eq!(
        mid_count, 1,
        "blob must survive crash between ingest(new) and delete(old)"
    );

    // Both old and new refs present -> refcount >= 2.
    let total_refs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(pool)
        .await
        .unwrap();
    assert!(
        total_refs >= 2,
        "both old+new blob_refs must exist after ingest(new)-only crash"
    );

    // Complete the rename (reconcile step: run delete(old)).
    complete_delete_old(pool, "br-f", "old.rs").await;

    // After full rename: blob still alive because new ref keeps it.
    let final_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(pool)
        .await
        .unwrap();
    assert_eq!(
        final_count, 1,
        "blob must survive complete rename (new ref prevents GC)"
    );
}
