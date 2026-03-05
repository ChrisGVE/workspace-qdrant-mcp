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
    self as tfs, CREATE_TRACKED_FILES_INDEXES_SQL, CREATE_TRACKED_FILES_SQL,
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
    sqlx::query(CREATE_TRACKED_FILES_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_TRACKED_FILES_INDEXES_SQL {
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
        "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, chunk_count, collection, needs_reconcile, reconcile_reason, created_at, updated_at)
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

    // Verify flag was cleared
    let flagged = tfs::get_files_needing_reconcile(&pool).await.unwrap();
    assert!(
        flagged.is_empty(),
        "needs_reconcile should be cleared after reconciliation"
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
