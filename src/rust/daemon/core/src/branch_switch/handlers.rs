//! Event handlers for branch switch and commit git events.

use std::collections::HashSet;
use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::git::{diff_tree, GitEvent, GitEventType};
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::QueueOperation;
use crate::watching_queue::get_current_branch;

use super::db::{batch_update_branch, fetch_watch_folder, update_last_commit_hash};
use super::queue::{enqueue_changed_file, enqueue_tenant_scan};
use super::types::BranchSwitchStats;

/// Handle a git event by dispatching to the appropriate handler.
pub async fn handle_git_event(
    event: &GitEvent,
    pool: &SqlitePool,
    queue_manager: &QueueManager,
) -> Result<BranchSwitchStats, String> {
    let (project_root, collection, tenant_id) = fetch_watch_folder(pool, &event.watch_folder_id).await?;

    match &event.event_type {
        GitEventType::BranchSwitch => {
            handle_branch_switch(
                event, pool, queue_manager,
                &project_root, &collection, &tenant_id,
            ).await
        }
        GitEventType::Commit | GitEventType::Merge | GitEventType::Pull | GitEventType::Rebase => {
            handle_new_commit(
                event, pool, queue_manager,
                &project_root, &collection, &tenant_id,
            ).await
        }
        GitEventType::Reset => {
            // Reset can change arbitrary files — enqueue a full scan
            info!(
                "Git reset detected for {}, enqueueing full scan",
                event.watch_folder_id
            );
            enqueue_tenant_scan(queue_manager, &tenant_id, &collection, &project_root).await?;
            Ok(BranchSwitchStats::default())
        }
        GitEventType::Stash | GitEventType::Unknown => {
            debug!(
                "Ignoring git event {:?} for {}",
                event.event_type, event.watch_folder_id
            );
            Ok(BranchSwitchStats::default())
        }
    }
}

/// Handle a branch switch: diff-tree for changes, batch-update unchanged, enqueue changed.
async fn handle_branch_switch(
    event: &GitEvent,
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    project_root: &str,
    collection: &str,
    tenant_id: &str,
) -> Result<BranchSwitchStats, String> {
    let root = Path::new(project_root);
    let new_branch = event.branch.as_deref().unwrap_or("default");
    let old_branch = event.old_branch.as_deref()
        .unwrap_or_else(|| get_current_branch(root).leak());

    info!(
        "Branch switch: {} -> {} for {} (old_sha={:.8}, new_sha={:.8})",
        old_branch, new_branch, event.watch_folder_id,
        &event.old_sha, &event.new_sha
    );

    // Get changed files between old and new commits via diff-tree
    let changes = diff_tree(root, &event.old_sha, &event.new_sha)
        .map_err(|e| format!("diff_tree failed: {}", e))?;

    let changed_paths: HashSet<String> = changes.iter().map(|c| c.path.clone()).collect();

    let mut stats = BranchSwitchStats::default();

    // 1. Batch update unchanged files: update branch in tracked_files
    //    These files have identical content — only the branch metadata changes.
    match batch_update_branch(
        pool, &event.watch_folder_id, old_branch, new_branch, &changed_paths,
    ).await {
        Ok(count) => {
            stats.batch_updated = count;
            if count > 0 {
                info!(
                    "Batch-updated {} unchanged files: branch {} -> {}",
                    count, old_branch, new_branch
                );
            }
        }
        Err(e) => {
            warn!("Batch branch update failed: {}", e);
            stats.errors += 1;
        }
    }

    // 2. Enqueue changed files for re-ingestion
    for change in &changes {
        let result = enqueue_changed_file(
            queue_manager, change, tenant_id, collection, project_root, new_branch,
        ).await;
        match result {
            Ok(op) => match op {
                QueueOperation::Update => stats.enqueued_changed += 1,
                QueueOperation::Add => stats.enqueued_added += 1,
                QueueOperation::Delete => stats.enqueued_deleted += 1,
                _ => {}
            },
            Err(e) => {
                warn!("Failed to enqueue changed file {}: {}", change.path, e);
                stats.errors += 1;
            }
        }
    }

    // 3. Update last_commit_hash in watch_folders
    if let Err(e) = update_last_commit_hash(pool, &event.watch_folder_id, &event.new_sha).await {
        warn!("Failed to update last_commit_hash: {}", e);
        stats.errors += 1;
    }

    info!(
        "Branch switch complete for {}: {} batch-updated, {} changed, {} added, {} deleted, {} errors",
        event.watch_folder_id, stats.batch_updated, stats.enqueued_changed,
        stats.enqueued_added, stats.enqueued_deleted, stats.errors
    );

    Ok(stats)
}

/// Handle a new commit on the same branch: diff-tree vs parent, enqueue changed files.
async fn handle_new_commit(
    event: &GitEvent,
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    project_root: &str,
    collection: &str,
    tenant_id: &str,
) -> Result<BranchSwitchStats, String> {
    let root = Path::new(project_root);
    let branch = event.branch.as_deref().unwrap_or("default");

    info!(
        "New commit on branch {} for {} (old_sha={:.8}, new_sha={:.8})",
        branch, event.watch_folder_id,
        &event.old_sha, &event.new_sha
    );

    let changes = diff_tree(root, &event.old_sha, &event.new_sha)
        .map_err(|e| format!("diff_tree failed: {}", e))?;

    let mut stats = BranchSwitchStats::default();

    for change in &changes {
        let result = enqueue_changed_file(
            queue_manager, change, tenant_id, collection, project_root, branch,
        ).await;
        match result {
            Ok(op) => match op {
                QueueOperation::Update => stats.enqueued_changed += 1,
                QueueOperation::Add => stats.enqueued_added += 1,
                QueueOperation::Delete => stats.enqueued_deleted += 1,
                _ => {}
            },
            Err(e) => {
                warn!("Failed to enqueue changed file {}: {}", change.path, e);
                stats.errors += 1;
            }
        }
    }

    // Update last_commit_hash
    if let Err(e) = update_last_commit_hash(pool, &event.watch_folder_id, &event.new_sha).await {
        warn!("Failed to update last_commit_hash: {}", e);
        stats.errors += 1;
    }

    info!(
        "Commit processed for {}: {} changed, {} added, {} deleted, {} errors",
        event.watch_folder_id, stats.enqueued_changed,
        stats.enqueued_added, stats.enqueued_deleted, stats.errors
    );

    Ok(stats)
}
