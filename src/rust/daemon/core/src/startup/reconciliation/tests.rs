//! Tests for startup reconciliation.

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
    manager
        .run_migrations()
        .await
        .expect("Failed to run schema migrations");
}

// -----------------------------------------------------------------------
// clean_stale_state tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_clean_stale_state_resets_in_progress() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    // Insert a watch folder for tenant t1 (so items aren't purged as orphan tenant items)
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/test-reset-in-progress', 'projects', 't1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert two queue items with status='in_progress'
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, worker_id, lease_until) \
         VALUES ('q1', 'file', 'add', 't1', 'projects', 'in_progress', \
         'key1', 'worker-old', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, worker_id, lease_until) \
         VALUES ('q2', 'file', 'add', 't1', 'projects', 'in_progress', \
         'key2', 'worker-old', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert one pending item that should NOT be affected
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q3', 'file', 'add', 't1', 'projects', 'pending', 'key3')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = clean_stale_state(&pool)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(stats.items_reset, 2, "Should reset 2 in_progress items");

    // Verify the items are now pending with cleared lease fields
    let row = sqlx::query(
        "SELECT status, worker_id, lease_until FROM unified_queue WHERE queue_id = 'q1'",
    )
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
         VALUES ('q_old_done', 'file', 'add', 't1', 'projects', 'done', \
         'key_old_done', '2020-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert an old failed item
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_old_fail', 'file', 'add', 't1', 'projects', 'failed', \
         'key_old_fail', '2020-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a recent done item that should NOT be removed
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_recent_done', 'file', 'add', 't1', 'projects', 'done', 'key_recent')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = clean_stale_state(&pool)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(
        stats.items_cleaned, 2,
        "Should clean 2 old done/failed items"
    );

    // Verify old items are gone
    let old_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE queue_id IN ('q_old_done', 'q_old_fail')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(old_count, 0, "Old items should be deleted");

    // Verify recent done item still exists
    let recent_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_recent_done'")
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

    let stats = clean_stale_state(&pool)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(
        stats.tracked_files_removed, 1,
        "Should remove 1 stale tracked file"
    );

    // Verify existing file is still tracked
    let remaining: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files WHERE file_path = 'exists.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(remaining, 1);

    // Verify gone file is removed
    let gone: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files WHERE file_path = 'gone.rs'")
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

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE file_path = 'file.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();

    // Insert a valid chunk referencing the file
    sqlx::query(
        "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at) \
         VALUES (?1, 'point-valid', 0, 'chash1', '2025-01-01T00:00:00Z')",
    )
    .bind(file_id)
    .execute(&pool)
    .await
    .unwrap();

    // Now delete the tracked file directly (bypassing CASCADE by disabling foreign keys)
    // to simulate an orphan chunk scenario.
    sqlx::query("PRAGMA foreign_keys = OFF")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
        .bind(file_id)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(&pool)
        .await
        .unwrap();

    // Verify the chunk is still there (orphaned)
    let chunk_count_before: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        chunk_count_before, 1,
        "Orphan chunk should exist before cleanup"
    );

    let stats = clean_stale_state(&pool)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(
        stats.orphan_chunks_removed, 1,
        "Should remove 1 orphan chunk"
    );

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
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 1);
    assert_eq!(stats.folders_deactivated, 1);
    assert_eq!(stats.folders_valid, 0);

    // Verify the folder is now inactive
    let is_active: bool =
        sqlx::query_scalar("SELECT is_active FROM watch_folders WHERE watch_id = 'w_gone'")
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
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_path)
    .execute(&pool)
    .await
    .unwrap();

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 1);
    assert_eq!(stats.folders_deactivated, 0);
    assert_eq!(stats.folders_valid, 1);

    // Verify the folder is still active
    let is_active: bool =
        sqlx::query_scalar("SELECT is_active FROM watch_folders WHERE watch_id = 'w_valid'")
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
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
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
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 2);
    assert_eq!(stats.folders_deactivated, 1);
    assert_eq!(stats.folders_valid, 1);
}

#[tokio::test]
async fn test_clean_stale_state_empty_tables() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    // Run on completely empty tables
    let stats = clean_stale_state(&pool)
        .await
        .expect("clean_stale_state failed");

    assert!(!stats.has_changes(), "No changes expected on empty tables");
    assert_eq!(stats.items_reset, 0);
    assert_eq!(stats.items_cleaned, 0);
    assert_eq!(stats.orphan_tenant_items_removed, 0);
    assert_eq!(stats.tracked_files_removed, 0);
    assert_eq!(stats.orphan_chunks_removed, 0);
}

#[tokio::test]
async fn test_clean_stale_state_purges_orphan_tenant_items() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    // Insert a watch folder for tenant t1 (valid tenant)
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/test-orphan-tenant-512', 'projects', 't1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a failed queue item for t1 (has watch_folder → should NOT be removed)
    // Use recent updated_at so step 2 (purge old done/failed) doesn't delete it
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_valid', 'file', 'add', 't1', 'projects', 'failed', \
         'key_valid')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert failed queue items for t_gone (NO watch_folder → should be removed)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_orphan1', 'file', 'add', 't_gone', 'projects', 'failed', \
         'key_orphan1', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_orphan2', 'file', 'add', 't_gone', 'projects', 'pending', \
         'key_orphan2', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert an in_progress item for t_gone (should NOT be touched — only failed/pending)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_inprog', 'file', 'add', 't_gone', 'projects', 'in_progress', \
         'key_inprog', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a doc item for t_gone (should NOT be touched — only file items)
    // Use recent updated_at so step 2 doesn't delete it
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_doc', 'doc', 'add', 't_gone', 'projects', 'failed', \
         'key_doc')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = clean_stale_state(&pool)
        .await
        .expect("clean_stale_state failed");

    assert_eq!(
        stats.orphan_tenant_items_removed, 2,
        "Should remove 2 orphan tenant items (failed + pending file items for t_gone)"
    );

    // Verify orphan items are gone
    let orphan_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE queue_id IN ('q_orphan1', 'q_orphan2')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(orphan_count, 0, "Orphan tenant items should be deleted");

    // Verify valid tenant item still exists
    let valid_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_valid'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(valid_count, 1, "Valid tenant item should remain");

    // Verify in_progress item for t_gone was reset (by step 1) but not deleted by step 3
    // Step 1 resets it to pending, then step 3 would delete it since t_gone has no watch_folder
    // Actually: step 1 runs first → resets to pending; step 3 runs after → deletes pending+failed for missing tenants
    // So q_inprog gets reset to pending by step 1, then deleted by step 3
    let inprog_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_inprog'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        inprog_count, 0,
        "in_progress item for orphan tenant should be reset then purged"
    );

    // Verify doc item still exists (only file items are purged)
    let doc_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_doc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(doc_count, 1, "Doc items should not be purged by orphan tenant cleanup");
}

#[tokio::test]
async fn test_validate_watch_folders_empty() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 0);
    assert_eq!(stats.folders_deactivated, 0);
    assert_eq!(stats.folders_valid, 0);
}
