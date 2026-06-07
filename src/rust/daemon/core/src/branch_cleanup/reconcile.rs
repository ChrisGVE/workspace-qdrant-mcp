//! Stale-branch reconciliation sweep (#102).
//!
//! Branch deletions are normally handled live by the branch lifecycle event
//! consumer, but events are missed when the daemon is down (or were never
//! emitted before the lifecycle code existed). The stale rows then linger in
//! `tracked_files.branches[]`, Qdrant payloads, and search.db `file_metadata`
//! — where each stale `file_metadata` row duplicates every unfiltered grep
//! match for that file.
//!
//! This sweep compares the branches recorded in both databases against the
//! repository's actual local branches and runs the regular
//! [`cleanup_deleted_branch`] path (which re-checks local AND remote
//! existence before touching anything) for each branch that is gone.

use std::collections::BTreeSet;
use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::branch_switch::BranchUpdateContext;

use super::cleanup_deleted_branch;

/// Outcome of one reconciliation sweep.
#[derive(Debug, Default)]
pub struct ReconcileStats {
    /// Git-tracked project watch folders examined.
    pub folders_checked: u64,
    /// Branches confirmed gone and cleaned up.
    pub branches_pruned: u64,
    /// Branches skipped (still exist remotely, or existence check failed).
    pub branches_skipped: u64,
    /// Errors encountered (non-fatal; sweep continues).
    pub errors: u64,
}

/// A project watch folder eligible for reconciliation.
#[derive(sqlx::FromRow)]
struct WatchFolderRow {
    watch_id: String,
    tenant_id: String,
    path: String,
}

/// Sweep all active git-tracked project watch folders for branches that are
/// recorded in `tracked_files` / `file_metadata` but no longer exist in the
/// repository, and clean them up via [`cleanup_deleted_branch`].
pub async fn reconcile_stale_branches(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
) -> ReconcileStats {
    let mut stats = ReconcileStats::default();

    let folders: Vec<WatchFolderRow> = match sqlx::query_as(
        "SELECT watch_id, tenant_id, path FROM watch_folders \
         WHERE collection = 'projects' AND is_git_tracked = 1 AND is_archived = 0",
    )
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            warn!("Branch reconcile: failed to list watch folders: {}", e);
            stats.errors += 1;
            return stats;
        }
    };

    for folder in folders {
        let root = Path::new(&folder.path);
        // Skip folders whose checkout is missing or not a git repo — branch
        // existence cannot be determined, so pruning would be unsafe.
        if !root.join(".git").exists() {
            debug!(
                "Branch reconcile: skipping '{}' (no .git at path)",
                folder.path
            );
            continue;
        }
        stats.folders_checked += 1;

        let stored = stored_branches(pool, branch_ctx, &folder).await;
        if stored.is_empty() {
            continue;
        }
        let local = match local_branches(root) {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    "Branch reconcile: branch listing failed for '{}': {}",
                    folder.path, e
                );
                stats.errors += 1;
                continue;
            }
        };

        for branch in stored.difference(&local) {
            // cleanup_deleted_branch re-checks local AND remote existence and
            // skips (no data loss) when the branch still exists anywhere or
            // the check fails.
            let result = cleanup_deleted_branch(
                pool,
                branch_ctx,
                &folder.watch_id,
                &folder.tenant_id,
                root,
                branch,
            )
            .await;
            if result.skipped {
                stats.branches_skipped += 1;
            } else {
                info!(
                    "Branch reconcile: pruned stale branch '{}' from '{}' \
                     ({} updated, {} deleted)",
                    branch, folder.path, result.updated, result.deleted
                );
                stats.branches_pruned += 1;
            }
            stats.errors += result.errors;
        }
    }

    stats
}

/// Union of branches recorded for this watch folder in state.db
/// (`tracked_files.branches[]`) and search.db (`file_metadata.branch`).
async fn stored_branches(
    pool: &SqlitePool,
    branch_ctx: &BranchUpdateContext,
    folder: &WatchFolderRow,
) -> BTreeSet<String> {
    let mut stored: BTreeSet<String> = BTreeSet::new();

    let tracked: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT j.value FROM tracked_files tf, json_each(tf.branches) j \
         WHERE tf.watch_folder_id = ?1",
    )
    .bind(&folder.watch_id)
    .fetch_all(pool)
    .await
    .unwrap_or_else(|e| {
        warn!(
            "Branch reconcile: tracked_files branch listing failed for '{}': {}",
            folder.watch_id, e
        );
        Vec::new()
    });
    stored.extend(tracked);

    if let Some(ref sdb) = branch_ctx.search_db {
        let metadata: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT branch FROM file_metadata \
             WHERE tenant_id = ?1 AND branch IS NOT NULL",
        )
        .bind(&folder.tenant_id)
        .fetch_all(sdb.pool())
        .await
        .unwrap_or_else(|e| {
            warn!(
                "Branch reconcile: file_metadata branch listing failed for '{}': {}",
                folder.tenant_id, e
            );
            Vec::new()
        });
        stored.extend(metadata);
    }

    stored
}

/// List the repository's current local branches.
fn local_branches(root: &Path) -> Result<BTreeSet<String>, String> {
    let out = std::process::Command::new("git")
        .args([
            "-C",
            &root.to_string_lossy(),
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
        ])
        .output()
        .map_err(|e| format!("git for-each-ref: {}", e))?;
    if !out.status.success() {
        return Err(format!("git for-each-ref exit={}", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}
