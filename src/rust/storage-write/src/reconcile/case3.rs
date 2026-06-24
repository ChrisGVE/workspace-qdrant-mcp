//! Case 3 -- missed branch-topology event (arch §4.7 case 3, AC-F15.1).
//!
//! File: `wqm-storage-write/src/reconcile/case3.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: Reconcile case 3 handles branches that were deleted from git while the
//!   daemon was down, so the daemon never received the delete event and the `branches`
//!   row plus all descendant data still exist in `store.db`.
//!
//!   Fix:
//!     1. Enumerate every `branches` row in the store.
//!     2. For each branch, ask `GitRefReader.branch_exists(branch_name)`.
//!     3. If the branch no longer exists in git, run `branch_delete` (the same
//!        8-step FP-1 sequence from arch §4.3 / F9).
//!
//!   **Git read is daemon-side** -- the `GitRefReader` trait (seam) is injected so
//!   offline tests use `MockGitRefReader` without a real git repository.
//!   The LIVE implementation RIDES #175.
//!
//! Neighbors: [`crate::branch::delete::branch_delete`] (the 8-step delete path),
//!   [`super::seams::GitRefReader`] (injectable git probe seam).

use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

use crate::blob::ladder::QdrantSink;
use crate::blob::lock::ContentKeyLockManager;
use crate::branch::delete::branch_delete;
use crate::qdrant::membership_batch::MembershipPutBatch;

use super::seams::GitRefReader;

/// One branch row from `store.db`.
struct BranchRow {
    branch_id: String,
    branch_name: String,
}

/// Enumerate all branch rows in the store.
async fn list_all_branches(pool: &SqlitePool) -> Result<Vec<BranchRow>, StorageError> {
    let rows = sqlx::query("SELECT branch_id, branch_name FROM branches ORDER BY branch_id")
        .fetch_all(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("case3 list_branches: {e}")))?;
    Ok(rows
        .into_iter()
        .map(|r| BranchRow {
            branch_id: r.get("branch_id"),
            branch_name: r.get("branch_name"),
        })
        .collect())
}

/// Reconcile case 3: delete branches confirmed absent from git topology.
///
/// For each branch in the store, asks `GitRefReader.branch_exists(branch_name)`.
/// If the branch no longer exists in git, runs `branch_delete` (the full 8-step
/// FP-1 sequence). Returns the count of branches deleted.
///
/// `lock_mgr` and `batch` are forwarded to `branch_delete` for its internal lock
/// acquisition and survivor membership PUTs. The caller must flush `batch` after
/// this function returns.
pub async fn run_case3<S, G>(
    pool: &SqlitePool,
    lock_mgr: &Arc<ContentKeyLockManager>,
    sink: &mut S,
    batch: &mut MembershipPutBatch,
    git_reader: &G,
    tenant_id: &str,
    collection_id: &str,
    collection_name: &str,
) -> Result<u64, StorageError>
where
    S: QdrantSink,
    G: GitRefReader,
{
    let branches = list_all_branches(pool).await?;
    let mut deleted = 0u64;

    for branch in &branches {
        if !git_reader.branch_exists(&branch.branch_name).await {
            branch_delete(
                pool,
                lock_mgr,
                sink,
                batch,
                &branch.branch_id,
                tenant_id,
                collection_id,
                collection_name,
            )
            .await?;
            deleted += 1;
        }
    }

    Ok(deleted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "case3_tests.rs"]
mod tests;
