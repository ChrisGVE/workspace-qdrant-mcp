//! Worktree detection during project registration.
//!
//! Resolves whether a project path is a git worktree and looks up the
//! main working tree's watch_id for the `main_worktree_watch_id` FK.

use std::path::Path;

use tracing::{debug, warn};

use crate::context::ProcessingContext;
use crate::unified_queue_schema::{ProjectPayload, UnifiedQueueItem};
use crate::watching_queue::WatchManager;
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

    // Look up the watch_id of the main worktree in watch_folders.
    // Docker Desktop may expose the same bind mount under a different
    // container-local alias, so try the current path plus common aliases.
    let mut main_watch_id = None;
    let mut candidate_paths = Vec::with_capacity(3);
    candidate_paths.push(main_path_str.clone());
    candidate_paths.extend(
        WatchManager::watch_path_aliases(&main_path_str)
            .into_iter()
            .map(|path| path.to_string_lossy().to_string()),
    );

    for candidate_path in candidate_paths {
        match sqlx::query_scalar::<_, String>(
            "SELECT watch_id FROM watch_folders \
             WHERE path = ?1 AND tenant_id = ?2 AND collection = ?3",
        )
        .bind(&candidate_path)
        .bind(&item.tenant_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(ctx.queue_manager.pool())
        .await
        {
            Ok(Some(watch_id)) => {
                main_watch_id = Some(watch_id);
                break;
            }
            Ok(None) => continue,
            Err(e) => {
                warn!(
                    "Failed to look up main worktree watch_id for path={} tenant={} (candidate={}): {}",
                    main_path_str, item.tenant_id, candidate_path, e
                );
            }
        }
    }

    if main_watch_id.is_none() {
        debug!(
            "Main worktree {} not yet registered for tenant={}; \
             main_worktree_watch_id will be NULL",
            main_path_str, item.tenant_id
        );
    }

    (true, main_watch_id)
}
