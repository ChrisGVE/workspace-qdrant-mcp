//! Triggers branch discovery when the first file event for an unknown branch arrives.
//!
//! Checks the DiscoveryTracker to avoid re-scanning on every file event.
//! When a branch has zero tracked_files entries, runs the discovery algorithm
//! to populate shared content from existing branches.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::branch_discovery::DiscoveryScanner;
use crate::branch_switch::BranchUpdateContext;
use crate::context::ProcessingContext;

/// Check if branch discovery is needed and run it if so.
///
/// This is a no-op if:
/// - The (watch_folder_id, branch) pair has already been checked
/// - The branch already has tracked_files entries
pub(super) async fn check_and_run_discovery(
    ctx: &ProcessingContext,
    pool: &SqlitePool,
    watch_folder_id: &str,
    tenant_id: &str,
    base_path: &str,
    branch: &str,
) {
    if ctx.discovery_tracker.is_checked(watch_folder_id, branch) {
        return;
    }

    ctx.discovery_tracker.mark_checked(watch_folder_id, branch);

    let has_files = branch_has_tracked_files(pool, watch_folder_id, branch).await;
    if has_files {
        debug!(
            "Branch '{}' already has tracked files for {}, skipping discovery",
            branch, watch_folder_id
        );
        return;
    }

    info!(
        "Branch '{}' has no tracked files for {}, running discovery",
        branch, watch_folder_id
    );

    let branch_ctx = BranchUpdateContext {
        storage_client: ctx.storage_client.clone(),
        search_db: ctx.search_db.clone(),
        branch_locks: ctx.branch_locks.clone(),
    };

    let project_root = Path::new(base_path);
    match DiscoveryScanner::discover(
        pool,
        &branch_ctx,
        watch_folder_id,
        tenant_id,
        project_root,
        branch,
    )
    .await
    {
        Ok(result) => {
            info!(
                "Discovery for '{}': {} shared, {} novel, parent={:?}",
                branch,
                result.shared_count,
                result.novel_paths.len(),
                result.parent_branch
            );
        }
        Err(e) => {
            warn!(
                "Branch discovery failed for '{}' in {}: {}",
                branch, watch_folder_id, e
            );
        }
    }
}

/// Check if a branch has any tracked_files entries for this watch folder.
async fn branch_has_tracked_files(pool: &SqlitePool, watch_folder_id: &str, branch: &str) -> bool {
    let result: Option<i64> = sqlx::query_scalar(
        "SELECT 1 FROM tracked_files
         WHERE watch_folder_id = ?1
           AND EXISTS (SELECT 1 FROM json_each(branches) WHERE json_each.value = ?2)
         LIMIT 1",
    )
    .bind(watch_folder_id)
    .bind(branch)
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    result.is_some()
}
