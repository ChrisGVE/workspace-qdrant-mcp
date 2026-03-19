//! Worktree detection during project registration.
//!
//! Resolves whether a project path is a git worktree and looks up the
//! main working tree's watch_id for the `main_worktree_watch_id` FK.

use std::path::Path;

use tracing::{debug, warn};

use crate::context::ProcessingContext;
use crate::unified_queue_schema::{ProjectPayload, UnifiedQueueItem};
use wqm_common::constants::COLLECTION_PROJECTS;

/// Resolve worktree information for a project being registered.
///
/// If `git_status.is_worktree` is true, resolves the git directory, finds the
/// main worktree path, and looks up the corresponding watch_id from
/// `watch_folders`. Returns `(is_worktree, main_worktree_watch_id)`.
pub(super) async fn resolve_worktree_info(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
    git_status: &crate::git::GitStatus,
) -> (bool, Option<String>) {
    if !git_status.is_worktree {
        return (false, None);
    }

    let project_path = Path::new(&payload.project_root);
    let git_dir = match crate::git::resolve_git_dir(project_path) {
        Some(d) => d,
        None => {
            warn!(
                "Worktree detected but could not resolve git dir for {}",
                payload.project_root
            );
            return (true, None);
        }
    };

    let main_path = match crate::git::find_main_worktree_path(&git_dir) {
        Some(p) => p,
        None => {
            warn!(
                "Worktree detected but could not find main worktree path for {}",
                payload.project_root
            );
            return (true, None);
        }
    };

    let main_path_str = main_path.to_string_lossy().to_string();
    debug!(
        "Worktree {} -> main worktree at {}",
        payload.project_root, main_path_str
    );

    // Look up the watch_id of the main worktree in watch_folders
    let main_watch_id = sqlx::query_scalar::<_, String>(
        "SELECT watch_id FROM watch_folders \
         WHERE path = ?1 AND tenant_id = ?2 AND collection = ?3",
    )
    .bind(&main_path_str)
    .bind(&item.tenant_id)
    .bind(COLLECTION_PROJECTS)
    .fetch_optional(ctx.queue_manager.pool())
    .await;

    let main_watch_id = match main_watch_id {
        Ok(v) => v,
        Err(e) => {
            warn!(
                "Failed to look up main worktree watch_id for path={} tenant={}: {}",
                main_path_str, item.tenant_id, e
            );
            None
        }
    };

    if main_watch_id.is_none() {
        debug!(
            "Main worktree {} not yet registered for tenant={}; \
             main_worktree_watch_id will be NULL",
            main_path_str, item.tenant_id
        );
    }

    (true, main_watch_id)
}
