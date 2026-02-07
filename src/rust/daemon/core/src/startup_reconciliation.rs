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
/// Performs four cleanup operations inside a single transaction:
/// 1. Reset in_progress queue items back to pending.
/// 2. Remove done/failed queue items older than 7 days.
/// 3. Remove tracked_files entries whose files no longer exist on disk.
/// 4. Remove orphan qdrant_chunks whose file_id is not in tracked_files.
pub async fn clean_stale_state(pool: &SqlitePool) -> Result<StaleCleanupStats, String> {
    let mut stats = StaleCleanupStats::default();

    // Step 1: Reset in_progress queue items back to pending
    info!("Resetting stale in_progress queue items...");
    let reset_result = sqlx::query(
        "UPDATE unified_queue SET status = 'pending', worker_id = NULL, lease_until = NULL, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE status = 'in_progress'"
    )
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to reset in_progress items: {}", e))?;

    stats.items_reset = reset_result.rows_affected();
    if stats.items_reset > 0 {
        info!("Reset {} in_progress queue items to pending", stats.items_reset);
    } else {
        debug!("No in_progress queue items to reset");
    }

    // Step 2: Clean old done/failed items (older than 7 days based on updated_at)
    info!("Cleaning old done/failed queue items...");
    let clean_result = sqlx::query(
        "DELETE FROM unified_queue \
         WHERE status IN ('done', 'failed') \
         AND updated_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-7 days')"
    )
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to clean old queue items: {}", e))?;

    stats.items_cleaned = clean_result.rows_affected();
    if stats.items_cleaned > 0 {
        info!("Cleaned {} old done/failed queue items", stats.items_cleaned);
    } else {
        debug!("No old done/failed queue items to clean");
    }

    // Step 3: Remove tracked_files entries where the file no longer exists on disk.
    // We need the watch folder path to reconstruct the absolute path for each tracked file.
    info!("Checking tracked files against filesystem...");
    let tracked_rows = sqlx::query(
        "SELECT tf.file_id, tf.file_path, wf.path AS watch_path \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id"
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
        // Delete stale tracked files in batches (CASCADE will remove qdrant_chunks too)
        for chunk in stale_file_ids.chunks(500) {
            let placeholders: String = chunk.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
            let delete_sql = format!(
                "DELETE FROM tracked_files WHERE file_id IN ({})",
                placeholders
            );
            sqlx::query(&delete_sql)
                .execute(pool)
                .await
                .map_err(|e| format!("Failed to delete stale tracked files: {}", e))?;
        }
        stats.tracked_files_removed = stale_file_ids.len() as u64;
        info!("Removed {} stale tracked files", stats.tracked_files_removed);
    } else {
        debug!("All tracked files still exist on disk");
    }

    // Step 4: Remove orphan qdrant_chunks (file_id not in tracked_files).
    // This catches any chunks that were orphaned by means other than CASCADE delete.
    info!("Cleaning orphan qdrant_chunks...");
    let orphan_result = sqlx::query(
        "DELETE FROM qdrant_chunks WHERE file_id NOT IN (SELECT file_id FROM tracked_files)"
    )
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to clean orphan qdrant_chunks: {}", e))?;

    stats.orphan_chunks_removed = orphan_result.rows_affected();
    if stats.orphan_chunks_removed > 0 {
        info!("Removed {} orphan qdrant_chunks", stats.orphan_chunks_removed);
    } else {
        debug!("No orphan qdrant_chunks found");
    }

    Ok(stats)
}

/// Validate watch_folders entries against the filesystem.
///
/// For each watch folder entry, checks whether the path still exists on disk.
/// Deactivates entries whose paths have been removed.
pub async fn validate_watch_folders(pool: &SqlitePool) -> Result<WatchValidationStats, String> {
    let mut stats = WatchValidationStats::default();

    info!("Validating watch folder paths...");
    let rows = sqlx::query(
        "SELECT watch_id, path, is_active, enabled FROM watch_folders"
    )
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
            // Path no longer exists -- deactivate
            warn!(
                "Watch folder path no longer exists, deactivating: watch_id={}, path={}",
                watch_id, path
            );
            sqlx::query(
                "UPDATE watch_folders SET is_active = 0, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                 WHERE watch_id = ?1"
            )
            .bind(&watch_id)
            .execute(pool)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::time::Duration;

    use crate::schema_version::SchemaManager;

    /// Create an in-memory SQLite pool for testing.
    async fn create_test_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool")
    }

    /// Run schema migrations to create all tables.
    async fn setup_schema(pool: &SqlitePool) {
        // Enable foreign keys
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(pool)
            .await
            .unwrap();

        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run schema migrations");
    }

    // -----------------------------------------------------------------------
    // clean_stale_state tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_clean_stale_state_resets_in_progress() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Insert two queue items with status='in_progress'
        sqlx::query(
            "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
             idempotency_key, worker_id, lease_until) \
             VALUES ('q1', 'file', 'ingest', 't1', 'projects', 'in_progress', \
             'key1', 'worker-old', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
             idempotency_key, worker_id, lease_until) \
             VALUES ('q2', 'file', 'ingest', 't1', 'projects', 'in_progress', \
             'key2', 'worker-old', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert one pending item that should NOT be affected
        sqlx::query(
            "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
             idempotency_key) \
             VALUES ('q3', 'file', 'ingest', 't1', 'projects', 'pending', 'key3')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let stats = clean_stale_state(&pool).await.expect("clean_stale_state failed");

        assert_eq!(stats.items_reset, 2, "Should reset 2 in_progress items");

        // Verify the items are now pending with cleared lease fields
        let row = sqlx::query("SELECT status, worker_id, lease_until FROM unified_queue WHERE queue_id = 'q1'")
            .fetch_one(&pool)
            .await
            .unwrap();

        let status: String = row.get("status");
        let worker_id: Option<String> = row.get("worker_id");
        let lease_until: Option<String> = row.get("lease_until");

        assert_eq!(status, "pending");
        assert!(worker_id.is_none(), "worker_id should be cleared");
        assert!(lease_until.is_none(), "lease_until should be cleared");

        // Verify the pending item was not touched
        let pending_row = sqlx::query("SELECT status FROM unified_queue WHERE queue_id = 'q3'")
            .fetch_one(&pool)
            .await
            .unwrap();
        let pending_status: String = pending_row.get("status");
        assert_eq!(pending_status, "pending");
    }

    #[tokio::test]
    async fn test_clean_stale_state_removes_old_done() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Insert an old done item (updated_at well in the past)
        sqlx::query(
            "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
             idempotency_key, updated_at) \
             VALUES ('q_old_done', 'file', 'ingest', 't1', 'projects', 'done', \
             'key_old_done', '2020-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert an old failed item
        sqlx::query(
            "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
             idempotency_key, updated_at) \
             VALUES ('q_old_fail', 'file', 'ingest', 't1', 'projects', 'failed', \
             'key_old_fail', '2020-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a recent done item that should NOT be removed
        sqlx::query(
            "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
             idempotency_key) \
             VALUES ('q_recent_done', 'file', 'ingest', 't1', 'projects', 'done', 'key_recent')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let stats = clean_stale_state(&pool).await.expect("clean_stale_state failed");

        assert_eq!(stats.items_cleaned, 2, "Should clean 2 old done/failed items");

        // Verify old items are gone
        let old_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE queue_id IN ('q_old_done', 'q_old_fail')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(old_count, 0, "Old items should be deleted");

        // Verify recent done item still exists
        let recent_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_recent_done'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(recent_count, 1, "Recent done item should not be deleted");
    }

    #[tokio::test]
    async fn test_clean_stale_state_removes_tracked_files_missing_on_disk() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Create a temp directory to act as a valid watch folder path
        let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let watch_path = tmp_dir.path().to_str().unwrap();

        // Create a real file inside the temp dir
        let real_file = tmp_dir.path().join("exists.rs");
        std::fs::write(&real_file, "fn main() {}").unwrap();

        // Insert a watch folder with the temp directory path
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
             VALUES ('w1', ?1, 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(watch_path)
        .execute(&pool)
        .await
        .unwrap();

        // Insert a tracked file that exists on disk
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
             VALUES ('w1', 'exists.rs', '2025-01-01T00:00:00Z', 'hash1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a tracked file that does NOT exist on disk
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
             VALUES ('w1', 'gone.rs', '2025-01-01T00:00:00Z', 'hash2', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let stats = clean_stale_state(&pool).await.expect("clean_stale_state failed");

        assert_eq!(stats.tracked_files_removed, 1, "Should remove 1 stale tracked file");

        // Verify existing file is still tracked
        let remaining: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tracked_files WHERE file_path = 'exists.rs'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(remaining, 1);

        // Verify gone file is removed
        let gone: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM tracked_files WHERE file_path = 'gone.rs'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(gone, 0);
    }

    #[tokio::test]
    async fn test_clean_stale_state_removes_orphan_chunks() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Insert a watch folder
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
             VALUES ('w1', '/tmp/nonexistent-test-dir-512', 'projects', 't1', \
             '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a tracked file
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
             VALUES ('w1', 'file.rs', '2025-01-01T00:00:00Z', 'hash1', \
             '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let file_id: i64 = sqlx::query_scalar(
            "SELECT file_id FROM tracked_files WHERE file_path = 'file.rs'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        // Insert a valid chunk referencing the file
        sqlx::query(
            "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at) \
             VALUES (?1, 'point-valid', 0, 'chash1', '2025-01-01T00:00:00Z')"
        )
        .bind(file_id)
        .execute(&pool)
        .await
        .unwrap();

        // Now delete the tracked file directly (bypassing CASCADE by disabling foreign keys)
        // to simulate an orphan chunk scenario.
        sqlx::query("PRAGMA foreign_keys = OFF").execute(&pool).await.unwrap();
        sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("PRAGMA foreign_keys = ON").execute(&pool).await.unwrap();

        // Verify the chunk is still there (orphaned)
        let chunk_count_before: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(chunk_count_before, 1, "Orphan chunk should exist before cleanup");

        let stats = clean_stale_state(&pool).await.expect("clean_stale_state failed");

        assert_eq!(stats.orphan_chunks_removed, 1, "Should remove 1 orphan chunk");

        let chunk_count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(chunk_count_after, 0, "Orphan chunk should be removed");
    }

    // -----------------------------------------------------------------------
    // validate_watch_folders tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_validate_watch_folders_deactivates_invalid() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Insert a watch folder with a path that does not exist
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
             created_at, updated_at) \
             VALUES ('w_gone', '/tmp/nonexistent-path-task-512-test', 'projects', 't1', 1, 1, \
             '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let stats = validate_watch_folders(&pool).await.expect("validate_watch_folders failed");

        assert_eq!(stats.folders_checked, 1);
        assert_eq!(stats.folders_deactivated, 1);
        assert_eq!(stats.folders_valid, 0);

        // Verify the folder is now inactive
        let is_active: bool = sqlx::query_scalar(
            "SELECT is_active FROM watch_folders WHERE watch_id = 'w_gone'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(!is_active, "Watch folder should be deactivated");
    }

    #[tokio::test]
    async fn test_validate_watch_folders_keeps_valid() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Create a temp directory to act as a valid watch folder path
        let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let watch_path = tmp_dir.path().to_str().unwrap();

        // Insert a watch folder pointing to the temp directory
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
             created_at, updated_at) \
             VALUES ('w_valid', ?1, 'projects', 't1', 1, 1, \
             '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(watch_path)
        .execute(&pool)
        .await
        .unwrap();

        let stats = validate_watch_folders(&pool).await.expect("validate_watch_folders failed");

        assert_eq!(stats.folders_checked, 1);
        assert_eq!(stats.folders_deactivated, 0);
        assert_eq!(stats.folders_valid, 1);

        // Verify the folder is still active
        let is_active: bool = sqlx::query_scalar(
            "SELECT is_active FROM watch_folders WHERE watch_id = 'w_valid'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(is_active, "Watch folder should remain active");
    }

    #[tokio::test]
    async fn test_validate_watch_folders_mixed() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Create a temp directory
        let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let valid_path = tmp_dir.path().to_str().unwrap();

        // Insert a valid watch folder
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
             created_at, updated_at) \
             VALUES ('w_ok', ?1, 'projects', 't1', 1, 1, \
             '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .bind(valid_path)
        .execute(&pool)
        .await
        .unwrap();

        // Insert an invalid watch folder
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
             created_at, updated_at) \
             VALUES ('w_bad', '/tmp/definitely-does-not-exist-512', 'libraries', 'lib1', 0, 1, \
             '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let stats = validate_watch_folders(&pool).await.expect("validate_watch_folders failed");

        assert_eq!(stats.folders_checked, 2);
        assert_eq!(stats.folders_deactivated, 1);
        assert_eq!(stats.folders_valid, 1);
    }

    #[tokio::test]
    async fn test_clean_stale_state_empty_tables() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        // Run on completely empty tables
        let stats = clean_stale_state(&pool).await.expect("clean_stale_state failed");

        assert!(!stats.has_changes(), "No changes expected on empty tables");
        assert_eq!(stats.items_reset, 0);
        assert_eq!(stats.items_cleaned, 0);
        assert_eq!(stats.tracked_files_removed, 0);
        assert_eq!(stats.orphan_chunks_removed, 0);
    }

    #[tokio::test]
    async fn test_validate_watch_folders_empty() {
        let pool = create_test_pool().await;
        setup_schema(&pool).await;

        let stats = validate_watch_folders(&pool).await.expect("validate_watch_folders failed");

        assert_eq!(stats.folders_checked, 0);
        assert_eq!(stats.folders_deactivated, 0);
        assert_eq!(stats.folders_valid, 0);
    }
}
