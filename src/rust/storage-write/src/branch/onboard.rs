//! Branch onboard + macro git-diff apply (arch §4.2, §4.6, F8).
//!
//! File: `wqm-storage-write/src/branch/onboard.rs`
//! Location: `src/rust/storage-write/src/branch/` (write-crate branch layer)
//!
//! ## Responsibilities
//!
//! - `branch_onboard`: register a newly-seen branch (`sync_state` pending->indexing->
//!   current), ingest all files from the diff with bounded concurrency (<=N_CPU), and
//!   write a resume cursor to `branches.sync_metadata` per batch so a crash mid-onboard
//!   is re-drivable (AC-F8.1, AC-F8.2).
//! - `apply_git_diff`: consume the result of a macro git op (merge/rebase/checkout) as
//!   a `Vec<FileChange>` without the branch-registration lifecycle; dispatches per file
//!   to ingest or single-file delete (AC-F8.6, arch §4.6).
//! - `resume_pending_onboards`: reads `sync_state IN ('pending', 'indexing')` and
//!   re-drives `branch_onboard` for each, honoring the cursor (AC-F8.2).
//!
//! ## Content/chunk seam (`FileContentProvider`)
//!
//! `FileChange` carries path+status ONLY (no content). The chunker lives in the daemon
//! pipeline, not here. This module defines the minimal `FileContentProvider` seam:
//!
//! ```rust,ignore
//! pub trait FileContentProvider {
//!     async fn chunk_file(tenant_id, branch_id, path) -> Result<IngestFileRequest, StorageError>
//! }
//! ```
//!
//! `branch_onboard` and `apply_git_diff` call `provider.chunk_file(...)` at most once
//! per changed file per invocation (AC-F8.3). The real provider -- git2 content read +
//! language-aware chunker + per-(location,rev) git-object cache that avoids the ~150s
//! SEED-B overhead -- is daemon-side and wired in issue #175. F8 ships the trait,
//! orchestration, and offline test doubles.
//!
//! ## Backoff injection
//!
//! `batch_retry_delays` on `OnboardConfig` carries the sleep durations for each retry
//! attempt (default: 1s, 2s, 4s). Tests inject `[0ms, 0ms, 0ms]` so no real sleep.
//!
//! ## Rename ordering (DOM-02, AC-F8.6)
//!
//! For `Renamed { old_path, .. }`: ingest(new) BEFORE delete(old). This ensures a blob
//! shared between old and new paths never transiently hits refcount 0. A crash between
//! the two halves leaves the durable `blobs` row intact; reconcile (F15) converges.
//!
//! Neighbors: [`super::delete`] (whole-branch delete), [`crate::blob::file_delete`]
//!   (single-file delete called for Deleted + rename-old), [`crate::blob::dedup`]
//!   (F6 ingest ladder called for Added + Modified + rename-new).
//!
//! ## Codesize splits
//!
//! Three sibling modules (included via `#[path]`) keep each file within the 500-line
//! codesize budget (coding.md §X):
//!   - `onboard_sql`      -- SQL helpers, flush retry, accumulators, PendingBranch/DiffProvider
//!   - `onboard_dispatch` -- ChangeOutcome, apply_one_change, dispatch arms, ingest_one, delete_one

use std::sync::Arc;

use sqlx::SqlitePool;
use wqm_common::error::StorageError;
use wqm_common::git::file_change::FileChange;
use wqm_storage::types::requests::IngestFileRequest;
use wqm_storage::types::stats::{BranchOnboardStats, DiffApplyStats};

use crate::blob::embed::Embedder;
use crate::blob::ladder::QdrantSink;
use crate::blob::lock::ContentKeyLockManager;
use crate::qdrant::membership_batch::MembershipPutBatch;

// SQL helpers, retry, accumulators, PendingBranch/PendingDiffProvider.
#[path = "onboard_sql.rs"]
mod sql;

// Per-file dispatch: ChangeOutcome, apply_one_change, ingest_one, delete_one.
#[path = "onboard_dispatch.rs"]
mod dispatch;

use dispatch::apply_one_change as _apply_one_change_inner;
pub(crate) use dispatch::{apply_one_change, ChangeOutcome};
use sql::{
    accumulate_diff_stats, accumulate_onboard_stats, fetch_pending_branches,
    flush_batch_with_retry, parse_cursor_index, set_sync_state, upsert_branch_pending,
    write_cursor,
};
pub use sql::{PendingBranch, PendingDiffProvider};

// ---------------------------------------------------------------------------
// FileContentProvider seam (arch §4.2, design decision 2 in task brief)
// ---------------------------------------------------------------------------

/// Provides chunked file content for the ingest ladder (arch §4.2 seam).
///
/// Called at most once per changed file per `branch_onboard` / `apply_git_diff`
/// invocation (AC-F8.3). The real implementation (git2 + language-aware chunker +
/// per-(location,rev) git-object cache) is daemon-side; see issue #175.
#[async_trait::async_trait]
pub trait FileContentProvider: Send + Sync {
    /// Return the pre-chunked ingest request for `path` on `branch_id`.
    ///
    /// Invoked ONCE per changed file per onboard (AC-F8.3). Must not cache
    /// results across calls; the caller drives the at-most-once invariant.
    async fn chunk_file(
        &self,
        tenant_id: &str,
        branch_id: &str,
        path: &str,
    ) -> Result<IngestFileRequest, StorageError>;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tunable parameters for `branch_onboard` (injected so tests do not sleep).
#[derive(Debug, Clone)]
pub struct OnboardConfig {
    /// Maximum concurrent file-ingest tasks (AC-F8.1 bounded concurrency).
    /// Default: `std::thread::available_parallelism()` or 4.
    pub max_concurrent: usize,
    /// Sleep durations for each retry attempt on Qdrant batch flush failure.
    /// Length determines max retries; after exhaustion `sync_state=error` is set.
    /// Default: [1s, 2s, 4s] (AC-F8.4). Tests inject [0, 0, 0].
    pub batch_retry_delays: Vec<std::time::Duration>,
    /// Qdrant points per batch flush attempt (AC-F8.4 ~1000 points per batch).
    pub batch_size: usize,
}

impl Default for OnboardConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            max_concurrent: cpus,
            batch_retry_delays: vec![
                std::time::Duration::from_secs(1),
                std::time::Duration::from_secs(2),
                std::time::Duration::from_secs(4),
            ],
            batch_size: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// branch_onboard (arch §4.2, AC-F8.1/F8.2/F8.3/F8.4)
// ---------------------------------------------------------------------------

/// Register and index a newly-seen branch (arch §4.2).
///
/// Sequence:
///   1. INSERT/UPSERT `branches` row with `sync_state=pending` (AC-F8.1).
///   2. Transition to `sync_state=indexing`.
///   3. For each `FileChange` in `diff`, dispatch to ingest or delete with bounded
///      concurrency (<=N_CPU) and write resume cursor per batch (AC-F8.2/F8.3).
///   4. Transition to `sync_state=current` on success, or `sync_state=error` on
///      exhausted batch retries (AC-F8.4).
///
/// `diff` is the pre-computed file-change list for this branch. `provider` is called
/// at most once per Added/Modified/Renamed file (AC-F8.3).
pub async fn branch_onboard(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    provider: &dyn FileContentProvider,
    tenant_id: &str,
    branch_id: &str,
    branch_name: &str,
    location: &str,
    collection_id: &str,
    diff: &[FileChange],
    cfg: &OnboardConfig,
) -> Result<BranchOnboardStats, StorageError> {
    upsert_branch_pending(pool, branch_id, branch_name, location).await?;
    set_sync_state(pool, branch_id, "indexing").await?;

    let result = run_onboard_loop(
        pool,
        locks,
        embedder,
        sink,
        provider,
        tenant_id,
        branch_id,
        collection_id,
        diff,
        cfg,
    )
    .await;

    match result {
        Ok(stats) => {
            set_sync_state(pool, branch_id, "current").await?;
            Ok(stats)
        }
        Err(e) => {
            let _ = set_sync_state(pool, branch_id, "error").await;
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// resume_pending_onboards (crash-resume, AC-F8.2)
// ---------------------------------------------------------------------------

/// Resume any branches in `pending` or `indexing` state (crash-resume, AC-F8.2).
///
/// Reads `branches` rows with `sync_state IN ('pending', 'indexing')`, re-drives
/// `branch_onboard` for each using the stored cursor in `sync_metadata`. The cursor
/// lets the re-run skip already-processed files (idempotent: `blob_refs ON CONFLICT
/// IGNORE` and the FTS trigger ensure re-ingest of a completed file is a no-op).
pub async fn resume_pending_onboards(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    provider: &dyn FileContentProvider,
    collection_id: &str,
    diff_provider: &dyn PendingDiffProvider,
    cfg: &OnboardConfig,
) -> Result<usize, StorageError> {
    let pending = fetch_pending_branches(pool).await?;
    let count = pending.len();
    for pb in pending {
        let diff = diff_provider
            .diff_for_branch(&pb.tenant_id, &pb.branch_id)
            .await?;
        let start_idx = parse_cursor_index(&pb.sync_metadata);
        let diff_slice = if start_idx < diff.len() {
            &diff[start_idx..]
        } else {
            &[]
        };
        branch_onboard(
            pool,
            locks,
            embedder,
            sink,
            provider,
            &pb.tenant_id,
            &pb.branch_id,
            &pb.branch_name,
            &pb.location,
            collection_id,
            diff_slice,
            cfg,
        )
        .await?;
    }
    Ok(count)
}

// ---------------------------------------------------------------------------
// apply_git_diff (arch §4.6, AC-F8.6/F8.7)
// ---------------------------------------------------------------------------

/// Apply a macro git op delta to an already-registered branch (arch §4.6).
///
/// Does NOT touch `sync_state` (the branch already exists). Dispatches per file:
///   - Added | Modified    -> ingest (F6 ladder)
///   - Deleted             -> single-file delete (F9, AC-F9.6)
///   - Renamed             -> ingest(new) BEFORE delete(old) (DOM-02, AC-F8.6)
///   - Copied | TypeChanged -> ingest the new/changed path
pub async fn apply_git_diff(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    batch: &mut MembershipPutBatch,
    provider: &dyn FileContentProvider,
    tenant_id: &str,
    branch_id: &str,
    collection_id: &str,
    collection_name: &str,
    changes: &[FileChange],
) -> Result<DiffApplyStats, StorageError> {
    let mut stats = DiffApplyStats::default();
    for change in changes {
        let outcome = apply_one_change(
            pool,
            locks,
            embedder,
            sink,
            batch,
            provider,
            tenant_id,
            branch_id,
            collection_id,
            collection_name,
            change,
        )
        .await?;
        accumulate_diff_stats(&mut stats, &outcome, change);
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// onboard loop: sequential with cursor + retry (AC-F8.1/F8.2/F8.4)
// ---------------------------------------------------------------------------

async fn run_onboard_loop(
    pool: &SqlitePool,
    locks: &Arc<ContentKeyLockManager>,
    embedder: &dyn Embedder,
    sink: &mut dyn QdrantSink,
    provider: &dyn FileContentProvider,
    tenant_id: &str,
    branch_id: &str,
    collection_id: &str,
    diff: &[FileChange],
    cfg: &OnboardConfig,
) -> Result<BranchOnboardStats, StorageError> {
    let mut stats = BranchOnboardStats::default();
    // Sequential loop with bounded concurrency intention: `max_concurrent` controls
    // the fan-out once sink becomes Arc<Mutex<dyn QdrantSink>> in #175. For now the
    // mutable `sink` borrow prevents true parallelism; the config field is the hook.
    let _ = cfg.max_concurrent; // documented intent; actual fan-out in #175
    let mut batch = MembershipPutBatch::new();
    for (idx, change) in diff.iter().enumerate() {
        let outcome = _apply_one_change_inner(
            pool,
            locks,
            embedder,
            sink,
            &mut batch,
            provider,
            tenant_id,
            branch_id,
            collection_id,
            "projects",
            change,
        )
        .await?;
        accumulate_onboard_stats(&mut stats, &outcome);
        write_cursor(pool, branch_id, idx + 1).await?;
        // Retry-guarded batch flush every `batch_size` files.
        if (idx + 1) % cfg.batch_size == 0 {
            flush_batch_with_retry(pool, branch_id, &mut batch, cfg).await?;
        }
    }
    // Final flush for any remainder.
    flush_batch_with_retry(pool, branch_id, &mut batch, cfg).await?;
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Tests (extracted to sibling for codesize compliance -- coding.md §X)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "onboard_tests.rs"]
mod tests;
