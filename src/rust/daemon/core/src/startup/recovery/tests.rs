//! Tests for startup recovery.

use super::*;

#[test]
fn test_recovery_stats_default() {
    let stats = RecoveryStats::default();
    assert_eq!(stats.files_to_ingest, 0);
    assert_eq!(stats.files_to_delete, 0);
    assert_eq!(stats.files_to_update, 0);
    assert_eq!(stats.files_unchanged, 0);
    assert_eq!(stats.files_routed_to_library, 0);
    assert_eq!(stats.files_newly_excluded, 0);
    assert_eq!(stats.errors, 0);
}

#[test]
fn test_full_recovery_stats_total() {
    let mut stats = FullRecoveryStats::default();
    stats.per_folder.push((
        "w1".to_string(),
        RecoveryStats {
            progressive_scans_enqueued: 1,
            files_to_delete: 2,
            files_newly_excluded: 1,
            ..RecoveryStats::default()
        },
    ));
    stats.per_folder.push((
        "w2".to_string(),
        RecoveryStats {
            progressive_scans_enqueued: 1,
            files_to_delete: 3,
            files_newly_excluded: 0,
            ..RecoveryStats::default()
        },
    ));

    // total_queued = progressive_scans + deletes + newly_excluded
    assert_eq!(stats.total_queued(), 1 + 2 + 1 + 1 + 3 + 0);
}

#[test]
fn test_compute_relative_path_for_recovery() {
    let root = Path::new("/home/user/project");
    let abs = Path::new("/home/user/project/src/main.rs");
    let rel = abs
        .strip_prefix(root)
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert_eq!(rel, "src/main.rs");
}

use crate::tracked_files_schema::{
    self as tfs, CREATE_TRACKED_FILES_V37_INDEXES_SQL, CREATE_TRACKED_FILES_V37_SQL,
};
use crate::unified_queue_schema::{CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL};
use crate::watch_folders_schema;
use sqlx::sqlite::SqlitePoolOptions;
use std::time::Duration;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn setup_reconcile_tables(pool: &SqlitePool) {
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(CREATE_TRACKED_FILES_V37_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_TRACKED_FILES_V37_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }
    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }
}

/// Insert a watch_folder and a tracked_file with needs_reconcile=1
async fn insert_reconcile_fixture(pool: &SqlitePool, base_path: &str, rel_path: &str) -> i64 {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at)
         VALUES ('wf-rc', ?1, 'projects', 'tenant-rc', 1, 0, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .bind(base_path)
    .execute(pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, relative_path, file_mtime, file_hash, chunk_count, collection, needs_reconcile, reconcile_reason, created_at, updated_at)
         VALUES ('wf-rc', ?1, '2025-01-01T00:00:00Z', 'abc123', 3, 'projects', 1, 'ingest_tx_failed: test', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .bind(rel_path)
    .execute(pool).await.unwrap();

    sqlx::query_scalar::<_, i64>("SELECT last_insert_rowid()")
        .fetch_one(pool)
        .await
        .unwrap()
}

#[tokio::test]
async fn test_reconcile_flagged_files_requeues_existing() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;

    // Use a real temp dir so the file "exists"
    let tmp = tempfile::tempdir().unwrap();
    let base_path = tmp.path().to_string_lossy().to_string();
    let rel_path = "src/main.rs";
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join(rel_path), "fn main() {}").unwrap();

    let file_id = insert_reconcile_fixture(&pool, &base_path, rel_path).await;

    let queue_manager = QueueManager::new(pool.clone());
    let mut stats = FullRecoveryStats::default();

    reconcile_flagged_files(&pool, &queue_manager, &mut stats).await;

    assert_eq!(stats.reconciled, 1);
    assert_eq!(stats.reconcile_errors, 0);

    // F-020: needs_reconcile must NOT be cleared immediately after enqueue.
    // The flag is only cleared when the queue item completes (delete_unified_item).
    let flagged = tfs::get_files_needing_reconcile(&pool).await.unwrap();
    assert_eq!(
        flagged.len(),
        1,
        "needs_reconcile must remain set until the queue item completes (F-020)"
    );

    // Verify an item was enqueued
    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1, "Should have enqueued one update item");

    // Verify the enqueued item is an update operation
    let op: String =
        sqlx::query_scalar("SELECT op FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(op, "update");

    // Simulate successful processing: delete_unified_item clears needs_reconcile.
    let queue_id: String =
        sqlx::query_scalar("SELECT queue_id FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    queue_manager
        .delete_unified_item(&queue_id)
        .await
        .expect("delete_unified_item failed");

    let flagged_after = tfs::get_files_needing_reconcile(&pool).await.unwrap();
    assert!(
        flagged_after.is_empty(),
        "needs_reconcile should be cleared after queue item completes (F-020)"
    );

    drop(tmp);
    let _ = file_id; // used for insert verification
}

#[tokio::test]
async fn test_reconcile_flagged_files_deleted_file_queues_delete() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;

    // Use a path that does NOT exist on disk
    let base_path = "/tmp/nonexistent_recovery_test_dir";
    let rel_path = "gone.rs";

    insert_reconcile_fixture(&pool, base_path, rel_path).await;

    let queue_manager = QueueManager::new(pool.clone());
    let mut stats = FullRecoveryStats::default();

    reconcile_flagged_files(&pool, &queue_manager, &mut stats).await;

    assert_eq!(stats.reconciled, 1);
    assert_eq!(stats.reconcile_errors, 0);

    // Verify the enqueued item is a delete operation
    let op: String =
        sqlx::query_scalar("SELECT op FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(op, "delete");
}

#[tokio::test]
async fn test_reconcile_no_flagged_files_is_noop() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;

    let queue_manager = QueueManager::new(pool.clone());
    let mut stats = FullRecoveryStats::default();

    reconcile_flagged_files(&pool, &queue_manager, &mut stats).await;

    assert_eq!(stats.reconciled, 0);
    assert_eq!(stats.reconcile_errors, 0);
}

/// A file whose on-disk hash diverges from the stored hash (edited while the
/// daemon was down) must enqueue an Update when reconcile_modified is on. This
/// is the root-cause fix for stale chunks surviving across a daemon restart.
#[tokio::test]
async fn test_process_tracked_file_modified_enqueues_update() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;
    let queue_manager = QueueManager::new(pool.clone());

    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let abs = tmp.path().join("src/main.rs");
    std::fs::write(&abs, "fn main() { println!(\"v2\"); }").unwrap();

    let mut stats = RecoveryStats::default();
    let queued = super::process_tracked_file(
        &queue_manager,
        "tenant-rc",
        "projects",
        tmp.path(),
        &abs,
        "src/main.rs",
        "stale_old_hash", // differs from the on-disk content hash
        true,             // reconcile_modified
        &mut stats,
    )
    .await;

    assert_eq!(queued, 1);
    assert_eq!(stats.files_to_update, 1);
    assert_eq!(stats.files_unchanged, 0);

    let op: String =
        sqlx::query_scalar("SELECT op FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(op, "update");
}

/// A file whose on-disk hash matches the stored hash is counted as unchanged
/// and enqueues nothing — no spurious churn for untouched files.
#[tokio::test]
async fn test_process_tracked_file_unchanged_enqueues_nothing() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;
    let queue_manager = QueueManager::new(pool.clone());

    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let abs = tmp.path().join("src/main.rs");
    std::fs::write(&abs, "fn main() {}").unwrap();

    // Stored hash == the real on-disk hash → unchanged.
    let on_disk = wqm_common::hashing::compute_file_hash(&abs).unwrap();

    let mut stats = RecoveryStats::default();
    let queued = super::process_tracked_file(
        &queue_manager,
        "tenant-rc",
        "projects",
        tmp.path(),
        &abs,
        "src/main.rs",
        &on_disk,
        true,
        &mut stats,
    )
    .await;

    assert_eq!(queued, 0);
    assert_eq!(stats.files_unchanged, 1);
    assert_eq!(stats.files_to_update, 0);

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-rc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 0);
}

/// With reconcile_modified disabled, a modified file is left untouched — the
/// feature is strictly opt-in via StartupConfig.
#[tokio::test]
async fn test_process_tracked_file_modified_skipped_when_reconcile_off() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;
    let queue_manager = QueueManager::new(pool.clone());

    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let abs = tmp.path().join("src/main.rs");
    std::fs::write(&abs, "fn main() { /* edited */ }").unwrap();

    let mut stats = RecoveryStats::default();
    let queued = super::process_tracked_file(
        &queue_manager,
        "tenant-rc",
        "projects",
        tmp.path(),
        &abs,
        "src/main.rs",
        "stale_old_hash",
        false, // reconcile_modified OFF
        &mut stats,
    )
    .await;

    assert_eq!(queued, 0);
    assert_eq!(stats.files_to_update, 0);
    assert_eq!(stats.files_unchanged, 0);
}
