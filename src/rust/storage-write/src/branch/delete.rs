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
// Tests (extracted to sibling file to keep this file within codesize budget)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "delete_tests.rs"]
mod tests;
