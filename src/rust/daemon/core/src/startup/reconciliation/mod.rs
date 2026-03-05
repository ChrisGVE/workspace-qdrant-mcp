//! Daemon Startup Reconciliation (Task 512)
//!
//! Cleans stale state from previous daemon runs and validates watch folder
//! entries on startup. This runs after schema migrations and before the
//! unified queue processor starts.
//!
//! Two main operations:
//! 1. `clean_stale_state` - Resets in-progress items, purges old completed/failed
//!    items, removes tracked files for deleted files, and cleans orphan chunks.
//! 2. `validate_watch_folders` - Deactivates watch folders whose paths no longer
//!    exist on disk.

use std::path::Path;

use sqlx::{Row, SqlitePool};
use tracing::{debug, info, warn};

use crate::lifecycle::WatchFolderLifecycle;

/// Statistics returned by `clean_stale_state`.
#[derive(Debug, Clone, Default)]
pub struct StaleCleanupStats {
    /// Queue items reset from in_progress back to pending.
    pub items_reset: u64,
    /// Old done/failed queue items removed.
    pub items_cleaned: u64,
    /// Tracked files removed because the file no longer exists on disk.
    pub tracked_files_removed: u64,
    /// Orphan qdrant_chunks removed (file_id no longer in tracked_files).
    pub orphan_chunks_removed: u64,
}

impl StaleCleanupStats {
    /// Returns true if any cleanup action was performed.
    pub fn has_changes(&self) -> bool {
        self.items_reset > 0
            || self.items_cleaned > 0
            || self.tracked_files_removed > 0
            || self.orphan_chunks_removed > 0
    }
}

/// Statistics returned by `validate_watch_folders`.
#[derive(Debug, Clone, Default)]
pub struct WatchValidationStats {
    /// Total watch folders checked.
    pub folders_checked: u64,
    /// Watch folders deactivated because their path no longer exists.
    pub folders_deactivated: u64,
    /// Watch folders with valid paths.
    pub folders_valid: u64,
}

/// Clean stale state from previous daemon runs.
///
/// Performs four cleanup operations:
/// 1. Reset in_progress queue items back to pending.
/// 2. Remove done/failed queue items older than 7 days.
/// 3. Remove tracked_files entries whose files no longer exist on disk.
/// 4. Remove orphan qdrant_chunks whose file_id is not in tracked_files.
pub async fn clean_stale_state(pool: &SqlitePool) -> Result<StaleCleanupStats, String> {
    let mut stats = StaleCleanupStats::default();

    stats.items_reset = reset_in_progress_items(pool).await?;
    stats.items_cleaned = purge_old_completed_items(pool).await?;
    stats.tracked_files_removed = remove_stale_tracked_files(pool).await?;
    stats.orphan_chunks_removed = remove_orphan_chunks(pool).await?;

    Ok(stats)
}

/// Step 1: Reset in_progress queue items back to pending.
async fn reset_in_progress_items(pool: &SqlitePool) -> Result<u64, String> {
    info!("Resetting stale in_progress queue items...");
    let result = sqlx::query(
        "UPDATE unified_queue SET status = 'pending', worker_id = NULL, lease_until = NULL, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE status = 'in_progress'",
    )
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to reset in_progress items: {}", e))?;

    let count = result.rows_affected();
    if count > 0 {
        info!("Reset {} in_progress queue items to pending", count);
    } else {
        debug!("No in_progress queue items to reset");
    }
    Ok(count)
}

/// Step 2: Remove done/failed queue items older than 7 days.
async fn purge_old_completed_items(pool: &SqlitePool) -> Result<u64, String> {
    info!("Cleaning old done/failed queue items...");
    let result = sqlx::query(
        "DELETE FROM unified_queue \
         WHERE status IN ('done', 'failed') \
         AND updated_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-7 days')",
    )
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to clean old queue items: {}", e))?;

    let count = result.rows_affected();
    if count > 0 {
        info!("Cleaned {} old done/failed queue items", count);
    } else {
        debug!("No old done/failed queue items to clean");
    }
    Ok(count)
}

/// Step 3: Remove tracked_files entries whose files no longer exist on disk.
async fn remove_stale_tracked_files(pool: &SqlitePool) -> Result<u64, String> {
    info!("Checking tracked files against filesystem...");
    let tracked_rows = sqlx::query(
        "SELECT tf.file_id, tf.file_path, wf.path AS watch_path \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("Failed to query tracked files: {}", e))?;

    let mut stale_file_ids: Vec<i64> = Vec::new();
    for row in &tracked_rows {
        let file_id: i64 = row.get("file_id");
        let file_path: String = row.get("file_path");
        let watch_path: String = row.get("watch_path");

        let abs_path = Path::new(&watch_path).join(&file_path);
        if !abs_path.exists() {
            debug!("Tracked file no longer exists: {}", abs_path.display());
            stale_file_ids.push(file_id);
        }
    }

    if !stale_file_ids.is_empty() {
        for chunk in stale_file_ids.chunks(500) {
            let placeholders: String = chunk
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let delete_sql = format!(
                "DELETE FROM tracked_files WHERE file_id IN ({})",
                placeholders
            );
            sqlx::query(&delete_sql)
                .execute(pool)
                .await
                .map_err(|e| format!("Failed to delete stale tracked files: {}", e))?;
        }
        let count = stale_file_ids.len() as u64;
        info!("Removed {} stale tracked files", count);
        Ok(count)
    } else {
        debug!("All tracked files still exist on disk");
        Ok(0)
    }
}

/// Step 4: Remove orphan qdrant_chunks (file_id not in tracked_files).
async fn remove_orphan_chunks(pool: &SqlitePool) -> Result<u64, String> {
    info!("Cleaning orphan qdrant_chunks...");
    let result = sqlx::query(
        "DELETE FROM qdrant_chunks WHERE file_id NOT IN (SELECT file_id FROM tracked_files)",
    )
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to clean orphan qdrant_chunks: {}", e))?;

    let count = result.rows_affected();
    if count > 0 {
        info!("Removed {} orphan qdrant_chunks", count);
    } else {
        debug!("No orphan qdrant_chunks found");
    }
    Ok(count)
}

/// Validate watch_folders entries against the filesystem.
///
/// For each watch folder entry, checks whether the path still exists on disk.
/// Deactivates entries whose paths have been removed.
pub async fn validate_watch_folders(pool: &SqlitePool) -> Result<WatchValidationStats, String> {
    let mut stats = WatchValidationStats::default();

    info!("Validating watch folder paths...");
    let rows = sqlx::query("SELECT watch_id, path, is_active, enabled FROM watch_folders")
        .fetch_all(pool)
        .await
        .map_err(|e| format!("Failed to query watch_folders: {}", e))?;

    stats.folders_checked = rows.len() as u64;

    for row in &rows {
        let watch_id: String = row.get("watch_id");
        let path: String = row.get("path");
        let is_active: bool = row.get("is_active");
        let _enabled: bool = row.get("enabled");

        if !Path::new(&path).exists() {
            warn!(
                "Watch folder path no longer exists, deactivating: watch_id={}, path={}",
                watch_id, path
            );
            let lifecycle = WatchFolderLifecycle::new(pool.clone());
            lifecycle
                .deactivate_by_watch_id(&watch_id)
                .await
                .map_err(|e| format!("Failed to deactivate watch folder {}: {}", watch_id, e))?;

            stats.folders_deactivated += 1;
        } else {
            debug!(
                "Watch folder valid: watch_id={}, path={}, is_active={}",
                watch_id, path, is_active
            );
            stats.folders_valid += 1;
        }
    }

    Ok(stats)
}

#[cfg(test)]
mod tests;
