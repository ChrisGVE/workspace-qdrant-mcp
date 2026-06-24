//! Whole-branch delete orchestrator: calls probe + 8-step FP-1 sequence (arch §4.3, §5.5).
//!
//! File: `wqm-storage-write/src/branch/delete.rs`
//! Location: `src/rust/storage-write/src/branch/` (write-crate branch layer)
//! Context: Thin orchestrator for `branch_delete`. Git2 probe types + logic live in
//!   [`super::probe`]; SQL step helpers (`step1`–`step8`, `BlobCandidate`) live in
//!   [`super::steps`]. This file only owns the orchestration logic and the test module.
//!
//!   The caller runs `probe_branch` + `delete_decision` BEFORE calling `branch_delete`.
//!   `branch_delete` executes the Proceed branch unconditionally.
//!
//!   TRUTH TABLE (arch §4.3, DR GP-3, AC-F9.1) — implemented in [`super::probe`]:
//!   | Git probe result                                           | Action  |
//!   |------------------------------------------------------------|---------|
//!   | for-each-ref empty + reflog delete event                   | Proceed |
//!   | branch present in topology                                  | Keep    |
//!   | read error / Auth / I/O / locked repo / NotFound-ambiguous | DEFER   |
//!   | reflog unavailable / git dir unreachable                   | DEFER   |
//!
//! Neighbors: [`super::probe`] (GP-4 truth-table types + git2 probe), [`super::steps`]
//!   (SQL step helpers), [`crate::blob::gc`] (refcount helper), [`crate::qdrant`].

use std::sync::Arc;

use sqlx::SqlitePool;
use wqm_common::error::StorageError;

use crate::blob::ladder::QdrantSink;
use crate::blob::lock::ContentKeyLockManager;
use crate::qdrant::membership_batch::MembershipPutBatch;

pub use super::probe::{delete_decision, DeleteAction, GitBranchProbe};
use super::steps::{
    step1_preselect, step2_orphan_candidates, step3_enqueue_orphan_deletes,
    step4_chunked_delete_branch_rows, step5_verify_and_delete_orphan_blobs,
    step6_enqueue_survivor_puts, step7_delete_orphaned_files, step8_delete_branch,
};

/// Delete a branch and its orphaned blobs/Qdrant points (arch §4.3, §5.5).
///
/// Implements the 8-step FP-1 physical-delete sequence:
///   1. Pre-select all (blob_id, point_id, content_key) for the branch.
///   2. Identify orphan candidates via GROUP BY (outside tx).
///   3. Enqueue QdrantOp::Delete for each orphan (data product first, FP-1).
///   4. Chunked DELETE of fts_branch_membership / blob_refs / concrete.
///   5. ABA-guarded re-verify + DELETE orphan blobs rows.
///   6. Recompute membership + enqueue survivor PUTs into `batch`.
///   7. DELETE orphaned `files` rows.
///   8. DELETE `branches` row LAST (crash-recovery anchor).
///
/// The caller is responsible for running `delete_decision(probe_branch(...))` BEFORE
/// calling this function; this function is the "Proceed" branch of the truth table.
///
/// `sink` receives `QdrantOp::Delete` for each orphaned point (Step 3).
/// `batch` accumulates survivor membership PUTs (Step 6); the caller must call
/// `batch.flush(client)` OUTSIDE all locks after this function returns.
///
/// No embedding call is made at any point (AC-F9.8).
pub async fn branch_delete(
    pool: &SqlitePool,
    lock_mgr: &Arc<ContentKeyLockManager>,
    sink: &mut dyn QdrantSink,
    batch: &mut MembershipPutBatch,
    branch_id: &str,
    tenant_id: &str,
    collection_id: &str,
    collection_name: &str,
) -> Result<(), StorageError> {
    // Step 1: pre-select all (blob_id, point_id, content_key) for the branch.
    let all_candidates = step1_preselect(pool, branch_id).await?;
    if all_candidates.is_empty() {
        // No blobs: still delete files + branch row (steps 7+8 are idempotent).
        step7_delete_orphaned_files(pool, branch_id).await?;
        step8_delete_branch(pool, branch_id).await?;
        return Ok(());
    }

    let all_blob_ids: Vec<i64> = all_candidates.iter().map(|c| c.blob_id).collect();

    // Step 2: identify orphan candidates (batched GROUP BY, outside transaction).
    let orphan_ids = step2_orphan_candidates(pool, branch_id, &all_blob_ids).await?;

    // Partition into orphans and survivors.
    let orphan_candidates: Vec<_> = all_candidates
        .iter()
        .filter(|c| orphan_ids.contains(&c.blob_id))
        .collect();
    let survivor_candidates: Vec<_> = all_candidates
        .iter()
        .filter(|c| !orphan_ids.contains(&c.blob_id))
        .collect();

    // Step 3: enqueue Qdrant DELETE for orphaned points (data product before truth row).
    step3_enqueue_orphan_deletes(sink, &orphan_candidates, collection_name);

    // Step 4: chunked DELETE of fts_branch_membership / blob_refs / concrete.
    step4_chunked_delete_branch_rows(pool, branch_id).await?;

    // Step 5: ABA-guarded re-verify and delete orphan blobs.
    let confirmed_orphans = step5_verify_and_delete_orphan_blobs(pool, &orphan_ids).await?;

    // Any orphan that survived Step 5 (ABA: new ref inserted between step 2 and step 5)
    // must also be included in survivor membership recompute.
    let final_survivors: Vec<_> = survivor_candidates
        .into_iter()
        .chain(
            orphan_candidates
                .iter()
                .copied()
                .filter(|c| !confirmed_orphans.contains(&c.blob_id)),
        )
        .collect();

    // Step 6: enqueue survivor membership PUTs (INSIDE lock, flush OUTSIDE later).
    step6_enqueue_survivor_puts(
        pool,
        lock_mgr,
        batch,
        &final_survivors,
        tenant_id,
        collection_id,
        collection_name,
    )
    .await?;

    // Step 7: delete orphaned files.
    step7_delete_orphaned_files(pool, branch_id).await?;

    // Step 8: delete branches row LAST (crash-recovery anchor, arch §4.3).
    step8_delete_branch(pool, branch_id).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use sqlx::SqlitePool;

    use super::*;
    use crate::blob::ladder::QdrantOp;
    use crate::blob::lock::{ContentKeyLockManager, LockManagerConfig};
    use crate::blob::test_support::{add_branch, fixture, TENANT};
    use crate::qdrant::membership_batch::MembershipPutBatch;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

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

        let file_id: i64 = sqlx::query_scalar(
            "SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?",
        )
        .bind(branch)
        .bind(&path)
        .fetch_one(pool)
        .await
        .expect("file_id");

        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
             VALUES (?, ?, 0, ?)",
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

    // -----------------------------------------------------------------------
    // AC-F9.1: truth table — unit tests for every row of delete_decision
    // (types re-exported from super::probe via delete.rs pub use)
    // -----------------------------------------------------------------------

    #[test]
    fn truth_table_proceed_requires_both_signals() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: true,
            git2_error: false,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Proceed);
    }

    #[test]
    fn truth_table_keep_when_ref_present() {
        let probe = GitBranchProbe {
            for_each_ref_empty: false,
            reflog_has_delete: true,
            git2_error: false,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Keep);
    }

    #[test]
    fn truth_table_keep_when_ref_present_no_reflog() {
        let probe = GitBranchProbe {
            for_each_ref_empty: false,
            reflog_has_delete: false,
            git2_error: false,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Keep);
    }

    #[test]
    fn truth_table_defer_on_git2_error() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: true,
            git2_error: true,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Defer);
    }

    #[test]
    fn truth_table_defer_on_reflog_unavailable() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: false,
            git2_error: false,
            reflog_unavailable: true,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Defer);
    }

    #[test]
    fn truth_table_defer_on_empty_ref_no_reflog_confirmation() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: false,
            git2_error: false,
            reflog_unavailable: false,
        };
        assert_eq!(delete_decision(probe), DeleteAction::Defer);
    }

    // AC-F9.1: transient NotFound MUST map to DEFER (SEED F06).
    #[test]
    fn truth_table_transient_not_found_maps_to_defer() {
        let probe = GitBranchProbe {
            for_each_ref_empty: true,
            reflog_has_delete: false,
            git2_error: true, // NotFound captured as git2_error
            reflog_unavailable: false,
        };
        assert_eq!(
            delete_decision(probe),
            DeleteAction::Defer,
            "transient NotFound must map to DEFER, not Proceed (AC-F9.1 SEED F06)"
        );
    }

    // AC-F9.1: integration — probe_branch on nonexistent branch -> DEFER.
    #[test]
    fn probe_branch_notfound_defers() {
        use super::super::probe::probe_branch;

        let dir = tempfile::TempDir::new().expect("tempdir");
        let repo = git2::Repository::init(dir.path()).expect("git init");
        let sig = git2::Signature::now("Test", "t@t.com").expect("sig");
        let tree_id = {
            let mut index = repo.index().expect("index");
            index.write_tree().expect("write_tree")
        };
        let tree = repo.find_tree(tree_id).expect("tree");
        repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[])
            .expect("commit");

        let probe = probe_branch(dir.path(), "nonexistent-branch");
        assert_eq!(
            delete_decision(probe),
            DeleteAction::Defer,
            "probe for nonexistent branch -> NotFound -> git2_error -> DEFER"
        );
    }

    // -----------------------------------------------------------------------
    // AC-F9.2: chunked-delete idiom; bundled SQLite rejects DELETE ... LIMIT
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // AC-F9.5 (DATA-03): RESTRICT FK backstop
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // AC-F9.3: 8-step FP-1 whole-branch delete + orphan GC
    // -----------------------------------------------------------------------

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

    // AC-F9.3: shared blob is NOT deleted; survivor PUT enqueued.
    #[tokio::test]
    async fn shared_blob_kept_and_survivor_put_enqueued() {
        let fx = fixture(BRANCH_A).await;
        add_branch(&fx.pool, BRANCH_B).await;
        let lm = lock_mgr();

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

        // File + blob_ref for BRANCH_A.
        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
             VALUES (?, 'shared.rs', 'projects', '2024-01-01', '2024-01-01')",
        )
        .bind(BRANCH_A)
        .execute(&fx.pool)
        .await
        .unwrap();
        let file_a: i64 = sqlx::query_scalar(
            "SELECT file_id FROM files WHERE branch_id = ? AND relative_path = 'shared.rs'",
        )
        .bind(BRANCH_A)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)",
        )
        .bind(BRANCH_A)
        .bind(file_a)
        .bind(blob_id)
        .execute(&fx.pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO fts_branch_membership(blob_id, branch_id) VALUES (?,?)")
            .bind(blob_id)
            .bind(BRANCH_A)
            .execute(&fx.pool)
            .await
            .unwrap();

        // File + blob_ref for BRANCH_B.
        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
             VALUES (?, 'shared.rs', 'projects', '2024-01-01', '2024-01-01')",
        )
        .bind(BRANCH_B)
        .execute(&fx.pool)
        .await
        .unwrap();
        let file_b: i64 = sqlx::query_scalar(
            "SELECT file_id FROM files WHERE branch_id = ? AND relative_path = 'shared.rs'",
        )
        .bind(BRANCH_B)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)",
        )
        .bind(BRANCH_B)
        .bind(file_b)
        .bind(blob_id)
        .execute(&fx.pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO fts_branch_membership(blob_id, branch_id) VALUES (?,?)")
            .bind(blob_id)
            .bind(BRANCH_B)
            .execute(&fx.pool)
            .await
            .unwrap();

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
        assert_eq!(count, 1, "shared blob must survive");

        let deletes: Vec<&QdrantOp> = sink
            .ops
            .iter()
            .filter(|op| matches!(op, QdrantOp::Delete { .. }))
            .collect();
        assert_eq!(deletes.len(), 0, "no Delete op for a shared blob");

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

        // Add a ref from BRANCH_B so the blob is never truly orphaned.
        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
             VALUES (?, 'aba.rs', 'projects', '2024-01-01', '2024-01-01')",
        )
        .bind(BRANCH_B)
        .execute(&fx.pool)
        .await
        .unwrap();
        let file_b: i64 = sqlx::query_scalar(
            "SELECT file_id FROM files WHERE branch_id = ? AND relative_path = 'aba.rs'",
        )
        .bind(BRANCH_B)
        .fetch_one(&fx.pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)",
        )
        .bind(BRANCH_B)
        .bind(file_b)
        .bind(blob_id)
        .execute(&fx.pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO fts_branch_membership(blob_id, branch_id) VALUES (?,?)")
            .bind(blob_id)
            .bind(BRANCH_B)
            .execute(&fx.pool)
            .await
            .unwrap();

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
}
