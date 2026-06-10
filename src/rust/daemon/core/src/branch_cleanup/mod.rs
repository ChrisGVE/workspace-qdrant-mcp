//! Branch cleanup: removal of branch membership data when a branch is deleted.
//!
//! When a branch is deleted (both locally and remotely), this module removes
//! the branch from all tracked_files.branches[] arrays, Qdrant point payloads,
//! and file_metadata rows. Points left with an empty branches[] array are
//! fully deleted (orphaned content).

mod db;
mod reconcile;

#[cfg(test)]
mod tests;

pub use reconcile::{reconcile_stale_branches, ReconcileStats};

use std::collections::HashSet;
use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::branch_switch::BranchUpdateContext;

/// File ids whose local rows are safe to delete: every orphaned file whose
/// base_point's Qdrant deletion did not fail (files without a base_point
/// have no Qdrant points to begin with).
fn deletable_file_ids(
    to_delete: &[&db::AffectedFile],
    failed_base_points: &HashSet<&str>,
) -> Vec<i64> {
    to_delete
        .iter()
        .filter(|f| {
            f.base_point
                .as_deref()
                .is_none_or(|bp| !failed_base_points.contains(bp))
        })
        .map(|f| f.file_id)
        .collect()
}

/// Result of a branch cleanup operation.
#[derive(Debug, Default)]
pub struct BranchCleanupResult {
    /// Files where the branch was removed but content retained (other branches still reference it).
    pub updated: u64,
    /// Files fully deleted (no branches remain).
    pub deleted: u64,
    /// Whether cleanup was skipped (branch still exists somewhere).
    pub skipped: bool,
    /// Errors encountered.
    pub errors: u64,
}

/// Result of a branch rename operation.
#[derive(Debug, Default)]
pub struct BranchRenameResult {
    /// Tracked files updated.
    pub updated: u64,
    /// Errors encountered.
    pub errors: u64,
}

/// Rename a branch across all data stores: tracked_files, Qdrant, and search.db.
pub async fn handle_branch_rename(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
    watch_folder_id: &str,
    tenant_id: &str,
    old_name: &str,
    new_name: &str,
) -> BranchRenameResult {
    let mut result = BranchRenameResult::default();

    // Hold per-tenant lock across all three stores for consistency
    let lock = branch_ctx.branch_locks.get(tenant_id);
    let _guard = lock.lock().await;

    // 1. Rename in tracked_files.branches[]
    match db::rename_branch_in_tracked_files(pool, watch_folder_id, old_name, new_name).await {
        Ok(count) => result.updated = count,
        Err(e) => {
            warn!("Failed to rename branch in tracked_files: {}", e);
            result.errors += 1;
        }
    }

    // 2. Rename in Qdrant points' branches payload
    db::rename_qdrant_branches(branch_ctx, pool, watch_folder_id, old_name, new_name).await;

    // 3. Rename in file_metadata (search.db)
    if let Some(ref sdb) = branch_ctx.search_db {
        db::rename_file_metadata_branch(sdb.pool(), tenant_id, old_name, new_name).await;
    }

    info!(
        "Branch rename '{}' -> '{}' complete: {} files updated, {} errors",
        old_name, new_name, result.updated, result.errors
    );

    result
}

/// Check whether a branch still exists locally or on a remote.
#[derive(Debug, PartialEq)]
pub enum BranchExistence {
    ExistsLocally,
    ExistsRemotely,
    Gone,
    CheckFailed(String),
}

/// Resolve the default remote for a branch: try the branch's upstream remote,
/// then fall back to git's default remote, then "origin".
fn resolve_default_remote(project_root: &Path, branch: &str) -> String {
    // Try branch-specific upstream remote
    let upstream = std::process::Command::new("git")
        .args([
            "-C",
            &project_root.to_string_lossy(),
            "config",
            &format!("branch.{}.remote", branch),
        ])
        .output();

    if let Ok(out) = upstream {
        if out.status.success() {
            let remote = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !remote.is_empty() {
                return remote;
            }
        }
    }

    // Fall back to first configured remote
    let remotes = std::process::Command::new("git")
        .args(["-C", &project_root.to_string_lossy(), "remote"])
        .output();

    if let Ok(out) = remotes {
        if out.status.success() {
            if let Some(first) = String::from_utf8_lossy(&out.stdout).lines().next() {
                let first = first.trim();
                if !first.is_empty() {
                    return first.to_string();
                }
            }
        }
    }

    "origin".to_string()
}

/// Check if a branch still exists locally or on a remote.
pub fn check_branch_existence(project_root: &Path, branch: &str) -> BranchExistence {
    // Check local refs
    let local = std::process::Command::new("git")
        .args([
            "-C",
            &project_root.to_string_lossy(),
            "branch",
            "--list",
            branch,
        ])
        .output();

    match local {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            return BranchExistence::ExistsLocally;
        }
        Err(e) => return BranchExistence::CheckFailed(format!("git branch --list: {}", e)),
        _ => {}
    }

    // Resolve the default remote (branch upstream or fallback to "origin")
    let remote_name = resolve_default_remote(project_root, branch);

    // Check remote refs
    let remote = std::process::Command::new("git")
        .args([
            "-C",
            &project_root.to_string_lossy(),
            "ls-remote",
            "--heads",
            &remote_name,
            &format!("refs/heads/{}", branch),
        ])
        .output();

    match remote {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            BranchExistence::ExistsRemotely
        }
        Ok(out) if out.status.success() => BranchExistence::Gone,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("Could not read from remote")
                || stderr.contains("does not appear to be a git repository")
            {
                // No remote configured — treat as "gone" since we already checked local
                BranchExistence::Gone
            } else {
                BranchExistence::CheckFailed(format!("git ls-remote exit={}", out.status))
            }
        }
        Err(e) => BranchExistence::CheckFailed(format!("git ls-remote: {}", e)),
    }
}

/// Clean up a deleted branch: remove from tracked_files, Qdrant, and search.db.
///
/// Only proceeds if the branch doesn't exist locally or remotely.
/// When remote check fails (network error), cleanup is deferred (no data loss).
pub async fn cleanup_deleted_branch(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
    watch_folder_id: &str,
    tenant_id: &str,
    project_root: &Path,
    deleted_branch: &str,
) -> BranchCleanupResult {
    let existence = check_branch_existence(project_root, deleted_branch);

    match existence {
        BranchExistence::ExistsLocally => {
            debug!(
                "Branch '{}' still exists locally, skipping cleanup",
                deleted_branch
            );
            return BranchCleanupResult {
                skipped: true,
                ..Default::default()
            };
        }
        BranchExistence::ExistsRemotely => {
            debug!(
                "Branch '{}' still exists on remote, skipping cleanup",
                deleted_branch
            );
            return BranchCleanupResult {
                skipped: true,
                ..Default::default()
            };
        }
        BranchExistence::CheckFailed(reason) => {
            warn!(
                "Branch existence check failed for '{}': {}. Deferring cleanup.",
                deleted_branch, reason
            );
            return BranchCleanupResult {
                skipped: true,
                errors: 1,
                ..Default::default()
            };
        }
        BranchExistence::Gone => {
            info!(
                "Branch '{}' confirmed gone (local + remote), proceeding with cleanup",
                deleted_branch
            );
        }
    }

    // Fetch all affected files
    let affected = match db::fetch_affected_files(pool, watch_folder_id, deleted_branch).await {
        Ok(files) => files,
        Err(e) => {
            warn!("Failed to fetch affected files for cleanup: {}", e);
            return BranchCleanupResult {
                errors: 1,
                ..Default::default()
            };
        }
    };

    if affected.is_empty() {
        debug!(
            "No tracked files reference branch '{}', nothing to clean up in state.db",
            deleted_branch
        );
        // Still prune search.db: file_metadata rows for this branch can
        // outlive tracked_files references (e.g. an earlier cleanup that ran
        // without a search_db handle). Leaving them makes every unfiltered
        // FTS query emit one duplicate match per stale branch row (#102).
        if let Some(ref sdb) = branch_ctx.search_db {
            db::delete_file_metadata_for_branch(sdb.pool(), tenant_id, deleted_branch).await;
        }
        return BranchCleanupResult::default();
    }

    let mut result = BranchCleanupResult::default();

    // Classify: files to update (remove branch) vs files to delete (orphaned)
    let (to_update, to_delete): (Vec<_>, Vec<_>) =
        affected.iter().partition(|f| f.remaining_branches > 0);

    // 1. Update files that still have other branches
    if !to_update.is_empty() {
        match db::remove_branch_from_tracked_files(pool, &to_update, deleted_branch).await {
            Ok(count) => result.updated = count,
            Err(e) => {
                warn!("Failed to remove branch from tracked files: {}", e);
                result.errors += 1;
            }
        }

        // Update Qdrant: set new branches array (without deleted branch)
        let lock = branch_ctx.branch_locks.get(tenant_id);
        let _guard = lock.lock().await;
        for f in &to_update {
            if let Some(ref bp) = f.base_point {
                let new_branches: Vec<&str> = f
                    .branches
                    .iter()
                    .filter(|b| b.as_str() != deleted_branch)
                    .map(|b| b.as_str())
                    .collect();
                db::update_qdrant_branches(branch_ctx, bp, &new_branches).await;
            }
        }
    }

    // 2. Delete orphaned files (no branches remain)
    if !to_delete.is_empty() {
        // Delete Qdrant points — only if no other watch folder references the
        // same base_point. A failed Qdrant deletion keeps the local rows so
        // the next cleanup/reconcile pass retries; deleting them anyway would
        // orphan the vectors with no repair path (#127).
        let mut failed_base_points: HashSet<&str> = HashSet::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for f in &to_delete {
            let Some(bp) = f.base_point.as_deref() else {
                continue;
            };
            if !seen.insert(bp) {
                continue;
            }
            if db::has_other_base_point_references(pool, bp, watch_folder_id).await {
                debug!(
                    "Keeping Qdrant points for base_point={} (referenced by other watch folders)",
                    bp
                );
            } else if let Err(e) = db::delete_qdrant_points(branch_ctx, bp).await {
                warn!("Cleanup: {} — keeping local rows for retry", e);
                failed_base_points.insert(bp);
                result.errors += 1;
            }
        }

        // Delete tracked_files and qdrant_chunks rows, except for files whose
        // Qdrant deletion failed above.
        let file_ids = deletable_file_ids(&to_delete, &failed_base_points);
        if !file_ids.is_empty() {
            match db::delete_orphaned_files(pool, &file_ids).await {
                Ok(count) => result.deleted = count,
                Err(e) => {
                    warn!("Failed to delete orphaned files: {}", e);
                    result.errors += 1;
                }
            }
        }
    }

    // 3. Delete file_metadata rows for the deleted branch
    if let Some(ref sdb) = branch_ctx.search_db {
        db::delete_file_metadata_for_branch(sdb.pool(), tenant_id, deleted_branch).await;
    }

    info!(
        "Branch '{}' cleanup complete: {} updated, {} deleted, {} errors",
        deleted_branch, result.updated, result.deleted, result.errors
    );

    result
}
