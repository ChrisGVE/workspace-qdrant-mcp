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

use crate::schema_version::SchemaManager;
use crate::tracked_files_schema::{self as tfs};
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
    // Build the real v48 schema (watch_folders + tracked_files + unified_queue
    // and the rest) so reconcile queries run against production columns.
    SchemaManager::new(pool.clone())
        .run_migrations()
        .await
        .expect("v48 migration chain must apply");
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
        "INSERT INTO tracked_files
             (watch_folder_id, tenant_id, branch, file_identity_id, content_key,
              relative_path, file_mtime, file_hash, chunk_count, collection,
              needs_reconcile, reconcile_reason, created_at, updated_at)
         VALUES ('wf-rc', 'tenant-rc', 'main', 'fid-rc', 'ck-rc',
                 ?1, '2025-01-01T00:00:00Z', 'abc123', 3, 'projects',
                 1, 'ingest_tx_failed: test', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
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

/// When a tracked file's on-disk mtime still matches the value stored at
/// ingest, `process_tracked_file` must count it unchanged WITHOUT re-hashing —
/// proven here by passing a deliberately wrong stored hash: if the gate hashed,
/// the mismatch would enqueue an Update; instead nothing is enqueued.
#[tokio::test]
async fn test_process_tracked_file_mtime_gate_skips_hash() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;
    let queue_manager = QueueManager::new(pool.clone());

    let tmp = tempfile::tempdir().unwrap();
    let rel_path = "src/main.rs";
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let abs = tmp.path().join(rel_path);
    std::fs::write(&abs, "fn main() {}").unwrap();
    let on_disk_mtime = tfs::get_file_mtime(&abs).unwrap();

    let mut stats = RecoveryStats::default();
    let enqueued = process_tracked_file(
        &queue_manager,
        "tenant-rc",
        "projects",
        tmp.path(),
        &abs,
        rel_path,
        "stale-hash-that-would-mismatch", // would trigger an Update if hashed
        &on_disk_mtime,                   // but mtime matches → no hash, no enqueue
        true,                             // reconcile_modified
        &mut stats,
    )
    .await;

    assert_eq!(enqueued, 0, "matching mtime must not enqueue anything");
    assert_eq!(
        stats.files_unchanged, 1,
        "matching mtime counts as unchanged"
    );
    assert_eq!(stats.files_to_update, 0, "no update when mtime matches");
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "mtime gate must skip the content hash entirely");
}

/// When the on-disk mtime differs from the stored value, the gate falls through
/// to the content hash; a genuine content change enqueues an Update.
#[tokio::test]
async fn test_process_tracked_file_mtime_changed_rehashes_and_updates() {
    let pool = create_test_pool().await;
    setup_reconcile_tables(&pool).await;
    let queue_manager = QueueManager::new(pool.clone());

    let tmp = tempfile::tempdir().unwrap();
    let rel_path = "src/main.rs";
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let abs = tmp.path().join(rel_path);
    std::fs::write(&abs, "fn main() { /* changed */ }").unwrap();

    let mut stats = RecoveryStats::default();
    let enqueued = process_tracked_file(
        &queue_manager,
        "tenant-rc",
        "projects",
        tmp.path(),
        &abs,
        rel_path,
        "old-content-hash",     // differs from the real on-disk hash
        "1999-01-01T00:00:00Z", // mtime differs → fall through to hashing
        true,
        &mut stats,
    )
    .await;

    assert_eq!(enqueued, 1, "changed content enqueues one item");
    assert_eq!(stats.files_to_update, 1, "content change counts as update");
    let op: String = sqlx::query_scalar("SELECT op FROM unified_queue")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(op, "update");
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
