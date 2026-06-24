//! SQL helpers for branch onboard lifecycle (arch §4.2, AC-F8.1/F8.2/F8.4).
//!
//! Extracted from `onboard.rs` for codesize compliance (coding.md §X / 500-line limit).
//! Contains: branch row lifecycle SQL, resume cursor, stats accumulators, flush retry,
//! `PendingBranch` / `PendingDiffProvider` types, and `fetch_pending_branches`.
//!
//! All functions here are `pub(super)` -- visible within the `branch::onboard` module only.

use sqlx::SqlitePool;
use tracing::debug;
use wqm_common::error::StorageError;
use wqm_common::git::file_change::{FileChange, FileChangeStatus};
use wqm_common::timestamps::now_utc;
use wqm_storage::types::stats::{BranchOnboardStats, DiffApplyStats};

use super::{ChangeOutcome, OnboardConfig};
use crate::qdrant::membership_batch::MembershipPutBatch;

// ---------------------------------------------------------------------------
// Branch row lifecycle
// ---------------------------------------------------------------------------

/// INSERT/UPSERT the `branches` row with `sync_state=pending` (AC-F8.1).
///
/// If the row already exists and is `current`, it stays `current` (re-onboard of a live
/// branch keeps its state intact). Otherwise it resets to `pending`.
pub(super) async fn upsert_branch_pending(
    pool: &SqlitePool,
    branch_id: &str,
    branch_name: &str,
    location: &str,
) -> Result<(), StorageError> {
    let now = now_utc();
    sqlx::query(
        "INSERT INTO branches(branch_id, branch_name, location, active, sync_state, \
         created_at, updated_at) VALUES (?, ?, ?, 1, 'pending', ?, ?) \
         ON CONFLICT(branch_id) DO UPDATE SET \
           branch_name = excluded.branch_name, \
           location    = excluded.location, \
           sync_state  = CASE WHEN sync_state = 'current' THEN 'current' ELSE 'pending' END, \
           updated_at  = excluded.updated_at",
    )
    .bind(branch_id)
    .bind(branch_name)
    .bind(location)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("upsert branch pending: {e}")))?;
    Ok(())
}

/// Transition `branches.sync_state` for `branch_id` (AC-F8.1 / AC-F8.4).
pub(super) async fn set_sync_state(
    pool: &SqlitePool,
    branch_id: &str,
    state: &str,
) -> Result<(), StorageError> {
    let now = now_utc();
    sqlx::query("UPDATE branches SET sync_state = ?, updated_at = ? WHERE branch_id = ?")
        .bind(state)
        .bind(&now)
        .bind(branch_id)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("set_sync_state {state}: {e}")))?;
    Ok(())
}

/// Write the resume cursor to `branches.sync_metadata` as JSON (AC-F8.2).
///
/// `processed` is the 1-based count of files processed so far. On crash-resume the
/// caller reads this value and skips the already-processed head of the diff list.
pub(super) async fn write_cursor(
    pool: &SqlitePool,
    branch_id: &str,
    processed: usize,
) -> Result<(), StorageError> {
    let json = serde_json::json!({ "last_processed_chunk_index": processed }).to_string();
    sqlx::query("UPDATE branches SET sync_metadata = ?, updated_at = ? WHERE branch_id = ?")
        .bind(&json)
        .bind(now_utc())
        .bind(branch_id)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("write_cursor: {e}")))?;
    Ok(())
}

/// Parse `last_processed_chunk_index` from a `sync_metadata` JSON string.
///
/// Returns 0 on any parse error or missing field -- safe default for crash-resume.
pub(super) fn parse_cursor_index(sync_metadata: &Option<String>) -> usize {
    let json = match sync_metadata.as_deref() {
        Some(s) => s,
        None => return 0,
    };
    serde_json::from_str::<serde_json::Value>(json)
        .ok()
        .and_then(|v| v["last_processed_chunk_index"].as_u64())
        .map(|n| n as usize)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Batch flush with retry (AC-F8.4)
// ---------------------------------------------------------------------------

/// Flush the `MembershipPutBatch` with retry+backoff (AC-F8.4).
///
/// The real Qdrant client is wired by the facade in issue #175. Until then, this
/// function implements the full retry skeleton, but calls the no-op offline flush so
/// tests pass without a live Qdrant.
///
/// Retry policy: attempt 0..N (N = `cfg.batch_retry_delays.len()`). On each failure,
/// sleep `cfg.batch_retry_delays[attempt]` then retry. After exhaustion, set
/// `sync_state=error` and return `Err`. Tests inject zero-duration delays.
pub(super) async fn flush_batch_with_retry(
    pool: &SqlitePool,
    branch_id: &str,
    batch: &mut MembershipPutBatch,
    cfg: &OnboardConfig,
) -> Result<(), StorageError> {
    if batch.is_empty() {
        return Ok(());
    }
    let max_attempts = cfg.batch_retry_delays.len() + 1; // +1 for the initial attempt
    for attempt in 0..max_attempts {
        match try_flush_batch(batch).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt + 1 < max_attempts => {
                debug!(
                    "flush_batch attempt {}/{max_attempts} failed: {e}; retrying",
                    attempt + 1
                );
                tokio::time::sleep(cfg.batch_retry_delays[attempt]).await;
            }
            Err(e) => {
                let _ = set_sync_state(pool, branch_id, "error").await;
                return Err(StorageError::Sqlite(format!(
                    "branch_onboard: Qdrant batch flush exhausted {max_attempts} attempts \
                     for branch={branch_id}: {e}"
                )));
            }
        }
    }
    Ok(())
}

/// One flush attempt. Returns Ok(()) in offline mode (no live client until #175).
async fn try_flush_batch(batch: &mut MembershipPutBatch) -> Result<(), StorageError> {
    // The real flush calls `batch.flush(&qdrant_client)` -- wired in #175.
    // In offline/test mode we treat the accumulated batch as a success.
    let _ = batch.len();
    Ok(())
}

// ---------------------------------------------------------------------------
// Stats accumulators
// ---------------------------------------------------------------------------

/// Accumulate per-file ingest outcome into `BranchOnboardStats`.
pub(super) fn accumulate_onboard_stats(stats: &mut BranchOnboardStats, outcome: &ChangeOutcome) {
    stats.files_changed += 1; // every change (ingest or delete) counts as a changed file
    stats.chunks_ingested += outcome.ingest.chunks_ingested;
    stats.blobs_created += outcome.ingest.blobs_created;
    stats.blobs_reused += outcome.ingest.blobs_reused;
}

/// Accumulate per-file outcome into `DiffApplyStats`, bucketed by change kind.
pub(super) fn accumulate_diff_stats(
    stats: &mut DiffApplyStats,
    outcome: &ChangeOutcome,
    change: &FileChange,
) {
    stats.chunks_ingested += outcome.ingest.chunks_ingested;
    stats.blobs_created += outcome.ingest.blobs_created;
    stats.blobs_reused += outcome.ingest.blobs_reused;
    match &change.status {
        FileChangeStatus::Added | FileChangeStatus::Copied { .. } => stats.files_added += 1,
        FileChangeStatus::Modified | FileChangeStatus::TypeChanged => stats.files_modified += 1,
        FileChangeStatus::Deleted => stats.files_deleted += 1,
        FileChangeStatus::Renamed { .. } => stats.files_renamed += 1,
    }
}

// ---------------------------------------------------------------------------
// PendingBranch / PendingDiffProvider (crash-resume, AC-F8.2)
// ---------------------------------------------------------------------------

/// A branch row in `pending` or `indexing` state (crash-resume candidate).
pub struct PendingBranch {
    pub tenant_id: String,
    pub branch_id: String,
    pub branch_name: String,
    pub location: String,
    pub sync_metadata: Option<String>,
}

/// Provides the diff list for a branch during crash-resume (AC-F8.2).
///
/// The real implementation fetches the diff via git2 from the project location.
/// Tests supply a simple mock implementation.
#[async_trait::async_trait]
pub trait PendingDiffProvider: Send + Sync {
    async fn diff_for_branch(
        &self,
        tenant_id: &str,
        branch_id: &str,
    ) -> Result<Vec<FileChange>, StorageError>;
}

/// Fetch all branches with `sync_state IN ('pending', 'indexing')`.
pub(super) async fn fetch_pending_branches(
    pool: &SqlitePool,
) -> Result<Vec<PendingBranch>, StorageError> {
    use sqlx::Row;
    let rows = sqlx::query(
        "SELECT branch_id, branch_name, location, sync_metadata, \
         (SELECT tenant_id FROM store_meta LIMIT 1) AS tenant_id \
         FROM branches WHERE sync_state IN ('pending', 'indexing')",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("fetch_pending_branches: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| PendingBranch {
            branch_id: r.get("branch_id"),
            branch_name: r.get("branch_name"),
            location: r.get("location"),
            sync_metadata: r.get("sync_metadata"),
            tenant_id: r.get::<Option<String>, _>("tenant_id").unwrap_or_default(),
        })
        .collect())
}
