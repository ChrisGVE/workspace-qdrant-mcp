//! Ignore file (.gitignore / .wqmignore) change detection and reconciliation.
//!
//! When the file watcher detects a Create or Modify event on .gitignore or
//! .wqmignore, this handler compares the file's mtime against the stored
//! value in `ignore_file_mtimes`, updates the stored mtime, and enqueues a
//! folder-level reconciliation scan for the project.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use crate::queue_operations::QueueManager;

use super::file_watcher::FileWatcherQueue;
use super::types::WatchConfig;

impl FileWatcherQueue {
    /// Handle a .gitignore or .wqmignore change by comparing mtime and
    /// enqueueing a folder-level reconciliation scan if the file is newer.
    pub(super) async fn handle_ignore_file_change(
        ignore_path: &Path,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        events_processed: &Arc<Mutex<u64>>,
    ) {
        let file_name = ignore_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let mtime_unix = match read_mtime_unix(ignore_path) {
            Some(t) => t,
            None => return,
        };

        let (tenant_id, project_root, collection) = {
            let c = config.read().await;
            (c.tenant_id.clone(), c.path.clone(), c.collection.clone())
        };

        let project_root_str = project_root.to_string_lossy().to_string();

        if !should_reconcile(queue_manager, &project_root_str, &file_name, mtime_unix).await {
            return;
        }

        update_stored_mtime(queue_manager, &project_root_str, &file_name, mtime_unix).await;

        run_reconciliation(
            &project_root,
            &tenant_id,
            &collection,
            queue_manager,
            events_processed,
        )
        .await;
    }
}

/// Read the file's mtime as a Unix timestamp, logging on failure.
fn read_mtime_unix(ignore_path: &Path) -> Option<i64> {
    match std::fs::metadata(ignore_path)
        .and_then(|m| m.modified())
        .map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64
        }) {
        Ok(t) => Some(t),
        Err(e) => {
            debug!(
                "Cannot read mtime for {}: {} — skipping reconciliation",
                ignore_path.display(),
                e
            );
            None
        }
    }
}

/// Check whether reconciliation is needed by comparing stored mtime.
async fn should_reconcile(
    queue_manager: &Arc<QueueManager>,
    project_root_str: &str,
    file_name: &str,
    mtime_unix: i64,
) -> bool {
    match crate::ignore_mtime::get_ignore_mtime(queue_manager.pool(), project_root_str, file_name)
        .await
    {
        Ok(Some(stored)) if stored >= mtime_unix => {
            debug!(
                "[ignore_watch] {} unchanged (stored={}, current={})",
                file_name, stored, mtime_unix
            );
            false
        }
        Ok(_) => true, // Newer or first time
        Err(e) => {
            warn!("[ignore_watch] mtime lookup failed: {e} — proceeding");
            true
        }
    }
}

/// Update the stored mtime for the ignore file.
async fn update_stored_mtime(
    queue_manager: &Arc<QueueManager>,
    project_root_str: &str,
    file_name: &str,
    mtime_unix: i64,
) {
    if let Err(e) = crate::ignore_mtime::set_ignore_mtime(
        queue_manager.pool(),
        project_root_str,
        file_name,
        mtime_unix,
    )
    .await
    {
        warn!("[ignore_watch] mtime update failed: {e}");
    }
}

/// Run reconciliation: diff tracked files vs eligible files,
/// enqueue stale deletions and missing additions.
async fn run_reconciliation(
    project_root: &Path,
    tenant_id: &str,
    collection: &str,
    queue_manager: &Arc<QueueManager>,
    events_processed: &Arc<Mutex<u64>>,
) {
    info!(
        "[ignore_watch] ignore file changed in {} — running reconciliation",
        tenant_id
    );

    // Resolve the global ignore file at call time so any edits to it via the
    // admin UI are immediately reflected on the next reconciliation trigger.
    let global_ignore_path: Option<std::path::PathBuf> = wqm_common::paths::get_database_path()
        .ok()
        .and_then(|p| p.parent().map(|dir| dir.join("global.wqmignore")));

    match crate::startup::reconciliation::ignore_sync::reconcile_ignore_rules(
        project_root,
        tenant_id,
        collection,
        queue_manager.pool(),
        queue_manager,
        global_ignore_path.as_deref(),
    )
    .await
    {
        Ok(stats) => {
            let mut count = events_processed.lock().await;
            *count += 1;
            if stats.stale_deleted > 0 || stats.missing_added > 0 {
                info!(
                    "[ignore_watch] Reconciled {}: {} stale deleted, {} missing added",
                    tenant_id, stats.stale_deleted, stats.missing_added
                );
            } else {
                debug!("[ignore_watch] Reconciled {}: no changes needed", tenant_id);
            }
        }
        Err(e) => {
            warn!(
                "[ignore_watch] Reconciliation failed for {}: {}",
                tenant_id, e
            );
        }
    }
}
