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

        // Get current mtime
        let mtime_unix = match std::fs::metadata(ignore_path)
            .and_then(|m| m.modified())
            .map(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64
            }) {
            Ok(t) => t,
            Err(e) => {
                debug!(
                    "Cannot read mtime for {}: {} — skipping reconciliation",
                    ignore_path.display(),
                    e
                );
                return;
            }
        };

        let (tenant_id, project_root, collection) = {
            let c = config.read().await;
            (c.tenant_id.clone(), c.path.clone(), c.collection.clone())
        };

        let project_root_str = project_root.to_string_lossy().to_string();

        // Compare against stored mtime
        match crate::ignore_mtime::get_ignore_mtime(
            queue_manager.pool(),
            &project_root_str,
            &file_name,
        )
        .await
        {
            Ok(Some(stored)) if stored >= mtime_unix => {
                debug!(
                    "[ignore_watch] {} unchanged (stored={}, current={})",
                    file_name, stored, mtime_unix
                );
                return;
            }
            Ok(_) => {} // Newer or first time
            Err(e) => {
                warn!("[ignore_watch] mtime lookup failed: {e} — proceeding");
            }
        }

        // Update stored mtime
        if let Err(e) = crate::ignore_mtime::set_ignore_mtime(
            queue_manager.pool(),
            &project_root_str,
            &file_name,
            mtime_unix,
        )
        .await
        {
            warn!("[ignore_watch] mtime update failed: {e}");
        }

        // Run reconciliation directly: diff tracked files vs eligible files,
        // enqueue stale deletions and missing additions
        info!(
            "[ignore_watch] {} changed in {} — running reconciliation",
            file_name, tenant_id
        );

        match crate::startup::reconciliation::ignore_sync::reconcile_ignore_rules(
            &project_root,
            &tenant_id,
            &collection,
            queue_manager.pool(),
            queue_manager,
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
}
