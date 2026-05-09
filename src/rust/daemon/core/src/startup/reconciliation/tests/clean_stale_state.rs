//! Tests for `clean_stale_state`.

use sqlx::Row;

use super::super::clean_stale_state;
use super::{create_test_pool, setup_schema};

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

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let watch_path = tmp_dir.path().to_str().unwrap();

    let real_file = tmp_dir.path().join("exists.rs");
    std::fs::write(&real_file, "fn main() {}").unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', ?1, 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .bind(watch_path)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('w1', 'exists.rs', '2025-01-01T00:00:00Z', 'hash1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('w1', 'gone.rs', '2025-01-01T00:00:00Z', 'hash2', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
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

    let remaining: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files WHERE file_path = 'exists.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(remaining, 1);

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

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('w1', '/tmp/nonexistent-test-dir-512', 'projects', 't1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('w1', 'file.rs', '2025-01-01T00:00:00Z', 'hash1', \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE file_path = 'file.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();

    sqlx::query(
        "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at) \
         VALUES (?1, 'point-valid', 0, 'chash1', '2025-01-01T00:00:00Z')",
    )
    .bind(file_id)
    .execute(&pool)
    .await
    .unwrap();

    // Delete tracked file without CASCADE to simulate orphan chunk.
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

#[tokio::test]
async fn test_clean_stale_state_empty_tables() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

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
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_valid', 'file', 'add', 't1', 'projects', 'failed', 'key_valid')",
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

    // Insert an in_progress item for t_gone (step 1 resets to pending, step 3 purges it)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key, updated_at) \
         VALUES ('q_inprog', 'file', 'add', 't_gone', 'projects', 'in_progress', \
         'key_inprog', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a doc item for t_gone (should NOT be touched — only file items purged)
    sqlx::query(
        "INSERT INTO unified_queue (queue_id, item_type, op, tenant_id, collection, status, \
         idempotency_key) \
         VALUES ('q_doc', 'doc', 'add', 't_gone', 'projects', 'failed', 'key_doc')",
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

    let orphan_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE queue_id IN ('q_orphan1', 'q_orphan2')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(orphan_count, 0, "Orphan tenant items should be deleted");

    let valid_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_valid'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(valid_count, 1, "Valid tenant item should remain");

    // Step 1 resets q_inprog to pending; step 3 then purges it (orphan tenant).
    let inprog_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_inprog'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        inprog_count, 0,
        "in_progress item for orphan tenant should be reset then purged"
    );

    let doc_count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q_doc'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(
        doc_count, 1,
        "Doc items should not be purged by orphan tenant cleanup"
    );
}
