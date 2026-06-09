//! Prune index documents for git branches that no longer exist.
//!
//! Branch-deletion → document purge is otherwise unwired in this deployment:
//! the file watcher excludes `.git/` (so it never observes a ref deletion), and
//! the `git::branch_lifecycle::BranchEventHandler` pipeline is test-only
//! scaffolding with no runtime consumer. Without this reconciler, every deleted
//! branch leaves its indexed documents orphaned forever — stale-branch
//! accumulation.
//!
//! This closes the gap by comparing the branches present in `tracked_files`
//! against the repository's live local branches and enqueuing `file|delete` for
//! every tracked file on a branch that no longer exists in git. It runs from the
//! same admin/startup reconciliation entry point as the ignore reconciler.
//!
//! ## Branch isolation is safe
//!
//! The file-delete processor reference-counts Qdrant points by `base_point`
//! (see `strategies/processing/file/delete.rs::check_qdrant_deletion_needed`):
//! deleting a branch's `tracked_files` rows only removes Qdrant points that no
//! other branch (e.g. `main`) still references. A shared point survives until
//! its last referencing branch is gone.
//!
//! ## Safety guards (never over-prune)
//!
//! Branch labels in `tracked_files` are NOT always real branch names: at index
//! time `get_current_branch` falls back to `"main"` when a repo can't be read,
//! so a project's whole corpus can end up labeled under a non-existent branch.
//! Treating branch-absence alone as "safe to delete" once wiped two projects
//! whose content was mislabeled under a bogus `"main"`. The guards below make
//! over-pruning structurally impossible — a lingering stale branch is benign,
//! deleting a real index is not:
//!
//! 1. Path missing, repo unopenable, or zero live branches → skip the project.
//! 2. HEAD branch not in the live set → labels untrustworthy → skip the project.
//! 3. Never prune the project's largest tracked branch (its corpus).
//! 4. Never prune a branch named `main` or `master` (default-name safety net).
//!
//! Only a branch that is absent from git AND passes all guards is pruned.

use std::collections::HashSet;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::git::BranchLifecycleDetector;
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation};
use crate::watching_queue::WatchManager;
use wqm_common::paths::RelativePath;

/// Totals returned by [`prune_orphaned_branches`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BranchPruneStats {
    /// Number of orphaned branches for which deletes were enqueued.
    pub branches_pruned: u64,
    /// Number of `file|delete` items enqueued across all pruned branches.
    pub files_enqueued: u64,
}

impl BranchPruneStats {
    /// Returns true if any branch was pruned.
    pub fn has_changes(&self) -> bool {
        self.branches_pruned > 0 || self.files_enqueued > 0
    }
}

/// Prune orphaned-branch documents for all active projects.
///
/// Iterates `watch_folders WHERE collection = 'projects' AND enabled = 1`, and
/// for each project enqueues `file|delete` for the tracked files of any branch
/// that no longer exists in the repository's local refs. Per-project failures
/// are logged and skipped — one bad repo never aborts the sweep.
pub async fn prune_orphaned_branches(
    pool: &SqlitePool,
    queue_manager: &std::sync::Arc<QueueManager>,
) -> Result<BranchPruneStats, String> {
    let rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT watch_id, tenant_id, path FROM watch_folders \
         WHERE collection = 'projects' AND enabled = 1",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("query active projects: {e}"))?;

    let mut totals = BranchPruneStats::default();
    for (watch_id, tenant_id, project_root) in &rows {
        match prune_project_branches(pool, queue_manager, watch_id, tenant_id, project_root).await {
            Ok(stats) => {
                totals.branches_pruned += stats.branches_pruned;
                totals.files_enqueued += stats.files_enqueued;
            }
            Err(e) => warn!("[branch_prune] reconciliation failed for {tenant_id}: {e}"),
        }
    }

    if totals.has_changes() {
        info!(
            "[branch_prune] Pruned {} orphaned branch(es), enqueued {} file delete(s)",
            totals.branches_pruned, totals.files_enqueued
        );
    }
    Ok(totals)
}

/// Prune orphaned branches for a single project.
async fn prune_project_branches(
    pool: &SqlitePool,
    queue_manager: &std::sync::Arc<QueueManager>,
    watch_id: &str,
    tenant_id: &str,
    project_root: &str,
) -> Result<BranchPruneStats, String> {
    let root = WatchManager::resolve_local_watch_path(project_root);
    if !root.is_dir() {
        debug!("[branch_prune] Skipping {tenant_id} — path not a directory");
        return Ok(BranchPruneStats::default());
    }

    // Ground truth: the repo's live local branches. On ANY error (not a repo,
    // unreadable, etc.) skip — pruning without a confirmed live set could wipe a
    // project's entire index.
    let detector = BranchLifecycleDetector::with_defaults(root.clone());
    let live: HashSet<String> = match detector.list_all_branches() {
        Ok(branches) => branches.into_iter().map(|(name, _, _)| name).collect(),
        Err(e) => {
            debug!("[branch_prune] Skipping {tenant_id} — cannot list git branches: {e}");
            return Ok(BranchPruneStats::default());
        }
    };
    if live.is_empty() {
        // Defensive: an empty live set would mark every tracked branch orphaned.
        debug!("[branch_prune] Skipping {tenant_id} — git reported zero local branches");
        return Ok(BranchPruneStats::default());
    }

    // Consistency proof. The repo's current HEAD branch MUST appear in the live
    // set. `get_current_branch` falls back to "main" when the repo can't be read
    // or HEAD can't be resolved — the exact failure that, at index time, labels a
    // project's corpus under a non-existent branch (observed: bws-engineer's and
    // compress-mcp's content was indexed under a bogus "main" while git only had
    // master/dev-clean). If HEAD isn't in `live`, the stored branch labels are not
    // trustworthy ground truth for this repo — skip rather than risk deleting the
    // real index.
    let head = crate::watching_queue::get_current_branch(&root);
    if !live.contains(&head) {
        debug!(
            "[branch_prune] Skipping {tenant_id} — HEAD '{head}' not among live branches; \
             branch labels not trustworthy"
        );
        return Ok(BranchPruneStats::default());
    }

    // Per-branch tracked-file counts. The branch holding the most files is the
    // project's corpus; a genuinely-deleted feature branch is a minor offshoot,
    // never the bulk. Never prune the corpus branch — this is the primary guard
    // against deleting a mislabeled main index.
    let counts: Vec<(String, i64)> = sqlx::query_as(
        "SELECT branch, COUNT(*) AS n FROM tracked_files \
         WHERE watch_folder_id = ?1 AND branch IS NOT NULL AND branch <> '' \
         GROUP BY branch",
    )
    .bind(watch_id)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("query tracked branches: {e}"))?;

    let primary = counts.iter().max_by_key(|(_, n)| *n).map(|(b, _)| b.as_str());

    let mut stats = BranchPruneStats::default();
    for (branch, _count) in &counts {
        if live.contains(branch) {
            continue; // still a real git branch
        }
        // Safety nets — never delete the corpus or a default-named branch, even
        // when absent from git. A lingering stale branch is benign; deleting a
        // real index is not. Surface the skip so mislabeled corpora are visible.
        if Some(branch.as_str()) == primary {
            info!(
                "[branch_prune] {tenant_id} — branch '{branch}' absent from git but is the \
                 project's largest tracked branch; NOT pruning (likely a mislabeled corpus)"
            );
            continue;
        }
        if branch.as_str() == "main" || branch.as_str() == "master" {
            info!(
                "[branch_prune] {tenant_id} — branch '{branch}' absent from git but is a \
                 default branch name; NOT pruning (safety net)"
            );
            continue;
        }

        let enqueued =
            enqueue_branch_deletes(pool, queue_manager, watch_id, tenant_id, branch.as_str())
                .await?;
        if enqueued > 0 {
            info!(
                "[branch_prune] {tenant_id} — branch '{branch}' no longer in git; \
                 enqueued {enqueued} file delete(s)"
            );
            stats.branches_pruned += 1;
            stats.files_enqueued += enqueued;
        }
    }
    Ok(stats)
}

/// Enqueue a `file|delete` for every tracked file on `branch`.
///
/// Mirrors the branch-switch / folder-delete enqueue path: `ItemType::File` +
/// `QueueOperation::Delete` with the branch set, so the unified queue processor
/// performs the branch-safe, reference-counted purge of Qdrant points,
/// `tracked_files`, FTS5 entries, graph edges, and keyword extractions.
async fn enqueue_branch_deletes(
    pool: &SqlitePool,
    queue_manager: &std::sync::Arc<QueueManager>,
    watch_id: &str,
    tenant_id: &str,
    branch: &str,
) -> Result<u64, String> {
    let paths: Vec<String> = sqlx::query_scalar(
        "SELECT relative_path FROM tracked_files \
         WHERE watch_folder_id = ?1 AND branch = ?2",
    )
    .bind(watch_id)
    .bind(branch)
    .fetch_all(pool)
    .await
    .map_err(|e| format!("query branch files: {e}"))?;

    let mut count = 0u64;
    for rel in paths {
        let rel_path = match RelativePath::from_user_input(&rel) {
            Ok(r) => r,
            Err(e) => {
                warn!("[branch_prune] invalid relative_path {rel:?}: {e}");
                continue;
            }
        };
        let payload = FilePayload {
            file_path: rel_path,
            file_type: None,
            file_hash: None,
            size_bytes: None,
            old_path: None,
        };
        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| format!("serialize FilePayload: {e}"))?;

        match queue_manager
            .enqueue_unified(
                ItemType::File,
                QueueOperation::Delete,
                tenant_id,
                "projects",
                &payload_json,
                Some(branch),
                None,
            )
            .await
        {
            Ok(_) => count += 1,
            Err(e) => warn!("[branch_prune] enqueue delete failed for {rel}: {e}"),
        }
    }
    Ok(count)
}
