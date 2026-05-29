//! Delete, stats, cleanup, and queue depth tests.

use super::*;
use crate::tracked_files_schema::CREATE_TRACKED_FILES_V37_SQL;

#[tokio::test]
async fn test_unified_queue_delete_item() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_delete.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue and dequeue
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"content":"test"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    let items = manager
        .dequeue_unified(10, "worker-1", None, None, None, None, None, None)
        .await
        .unwrap();
    assert_eq!(items.len(), 1);

    // Delete item after successful processing (per spec)
    let deleted = manager.delete_unified_item(&queue_id).await.unwrap();
    assert!(deleted);

    // Verify item is completely gone
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.done_items, 0);
    assert_eq!(stats.in_progress_items, 0);
    assert_eq!(stats.pending_items, 0);
    assert_eq!(stats.failed_items, 0);
}

#[tokio::test]
async fn test_unified_queue_stats() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_stats.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue items of different types
    manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file1.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"content":"test content"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Delete,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file2.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    let stats = manager.get_unified_queue_stats().await.unwrap();

    assert_eq!(stats.total_items, 3);
    assert_eq!(stats.pending_items, 3);
    assert_eq!(stats.by_item_type.get("file"), Some(&2));
    assert_eq!(stats.by_item_type.get("text"), Some(&1));
    assert_eq!(stats.by_operation.get("add"), Some(&2));
    assert_eq!(stats.by_operation.get("delete"), Some(&1));
}

#[tokio::test]
async fn test_unified_queue_cleanup() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_cleanup.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Enqueue and dequeue an item
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"content":"test"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    manager
        .dequeue_unified(10, "worker-1", None, None, None, None, None, None)
        .await
        .unwrap();

    // Set status to 'done' directly via SQL to test cleanup of done items
    sqlx::query("UPDATE unified_queue SET status = 'done', lease_until = NULL, worker_id = NULL WHERE queue_id = ?1")
        .bind(&queue_id)
        .execute(&pool)
        .await
        .unwrap();

    // With 24 hours retention, recently completed items should NOT be cleaned up
    let cleaned = manager
        .cleanup_completed_unified_items(Some(24))
        .await
        .unwrap();
    assert_eq!(cleaned, 0); // Item is too recent

    // Verify item still exists
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.done_items, 1);
}

#[tokio::test]
async fn test_unified_queue_depth() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_depth.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue items
    for i in 0..5 {
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                &format!(r#"{{"file_path":"/test/file{}.rs"}}"#, i),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Check depth
    let depth = manager.get_unified_queue_depth(None, None).await.unwrap();
    assert_eq!(depth, 5);

    // Check depth filtered by type
    let depth_file = manager
        .get_unified_queue_depth(Some(ItemType::File), None)
        .await
        .unwrap();
    assert_eq!(depth_file, 5);

    let depth_content = manager
        .get_unified_queue_depth(Some(ItemType::Text), None)
        .await
        .unwrap();
    assert_eq!(depth_content, 0);
}

#[tokio::test]
async fn test_oldest_pending_age_empty_queue() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_oldest_pending_empty.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let age = manager.get_oldest_pending_age_seconds().await.unwrap();
    assert_eq!(age, 0, "empty queue must report age 0");
}

#[tokio::test]
async fn test_oldest_pending_age_single_item() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_oldest_pending_single.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Enqueue one item with a created_at 10 seconds in the past.
    manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "tenant",
            "collection",
            r#"{"content":"x"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Backdate the pending item by 10 seconds.
    sqlx::query(
        "UPDATE unified_queue SET created_at = \
         strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-10 seconds') \
         WHERE status = 'pending'",
    )
    .execute(&pool)
    .await
    .unwrap();

    let age = manager.get_oldest_pending_age_seconds().await.unwrap();
    assert!(
        (9..=12).contains(&age),
        "expected age ~10s, got {} (tolerance for clock jitter)",
        age
    );
}

#[tokio::test]
async fn test_oldest_pending_age_multiple_items_returns_oldest() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_oldest_pending_multi.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Enqueue three items.
    for i in 0..3 {
        manager
            .enqueue_unified(
                ItemType::Text,
                UnifiedOp::Add,
                "tenant",
                "collection",
                &format!(r#"{{"content":"item-{}"}}"#, i),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Backdate each to a distinct offset: 60s, 20s, 5s ago.
    let queue_ids: Vec<(String,)> = sqlx::query_as(
        "SELECT queue_id FROM unified_queue WHERE status = 'pending' ORDER BY created_at ASC",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(queue_ids.len(), 3);
    let offsets = ["-60 seconds", "-20 seconds", "-5 seconds"];
    for ((id,), offset) in queue_ids.iter().zip(offsets.iter()) {
        sqlx::query(
            "UPDATE unified_queue \
             SET created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?1) \
             WHERE queue_id = ?2",
        )
        .bind(offset)
        .bind(id)
        .execute(&pool)
        .await
        .unwrap();
    }

    let age = manager.get_oldest_pending_age_seconds().await.unwrap();
    assert!(
        (58..=63).contains(&age),
        "expected oldest age ~60s, got {}",
        age
    );
}

#[tokio::test]
async fn test_oldest_pending_age_ignores_non_pending() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_oldest_pending_status.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Enqueue and flip to 'done' — must not be considered by the age query.
    let (qid, _) = manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "tenant",
            "collection",
            r#"{"content":"done-item"}"#,
            None,
            None,
        )
        .await
        .unwrap();
    sqlx::query(
        "UPDATE unified_queue \
         SET status = 'done', \
             created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-120 seconds') \
         WHERE queue_id = ?1",
    )
    .bind(&qid)
    .execute(&pool)
    .await
    .unwrap();

    let age = manager.get_oldest_pending_age_seconds().await.unwrap();
    assert_eq!(age, 0, "non-pending items must not contribute to age");
}
/// Regression test: Delete-completion side-effects (F-036) must be scoped by
/// tenant_id. After the T7 relative-path migration, `unified_queue.file_path`
/// stores the relative path anchored to the owning watch_folder root. Two
/// tenants with identical relative paths under different roots must not have
/// their tracked_files rows cross-deleted when only one tenant's Delete
/// completes.
#[tokio::test]
async fn test_delete_unified_item_tracked_files_scoped_by_tenant() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_tenant_isolation.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    sqlx::query(CREATE_TRACKED_FILES_V37_SQL)
        .execute(&pool)
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Two watch-folders with distinct tenant_ids, each holding a tracked_files
    // row at the SAME relative path. Post-T7, the queue's file_path column
    // stores the relative path, so a collision now happens whenever two
    // tenants share an identical content layout — the deletion side-effects
    // must be scoped by `tenant_id` to keep them isolated.
    //
    //   tenant-alpha: watch_root=/data/alpha    rel=src/lib.rs
    //   tenant-beta:  watch_root=/data/beta     rel=src/lib.rs
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('wf-t1', '/data/alpha', 'projects', 'tenant-alpha',
                 '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('wf-t2', '/data/beta', 'projects', 'tenant-beta',
                 '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Both tenants' tracked_files rows use the same relative path.
    let rel_path = "src/lib.rs";

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, relative_path, branch, needs_reconcile, file_mtime, file_hash, \
          collection, created_at, updated_at) \
         VALUES ('wf-t1', ?1, 'main', 0, '2025-01-01T00:00:00Z', 'hash1', \
                 'projects', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(rel_path)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, relative_path, branch, needs_reconcile, file_mtime, file_hash, \
          collection, created_at, updated_at) \
         VALUES ('wf-t2', ?1, 'main', 0, '2025-01-01T00:00:00Z', 'hash1', \
                 'projects', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(rel_path)
    .execute(&pool)
    .await
    .unwrap();

    // Verify both rows exist before the test
    let count_before: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tracked_files")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count_before, 2, "setup: expected 2 tracked_files rows");

    // Insert a Delete queue item for tenant-alpha directly so we can control
    // tenant_id without going through the full enqueue validation path. The
    // queue's file_path column carries the relative path (post-T7 semantics).
    let queue_id = "qid-delete-alpha";
    sqlx::query(
        "INSERT INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          status, branch, payload_json, metadata, file_path, created_at, updated_at) \
         VALUES (?1, ?2, 'file', 'delete', 'tenant-alpha', 'projects', \
                 'in_progress', 'main', '{}', '{}', ?3, \
                 '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(queue_id)
    .bind("idem-key-alpha-delete")
    .bind(rel_path)
    .execute(&pool)
    .await
    .unwrap();

    // Complete the Delete: only tenant-alpha's tracked_files row should be removed.
    let deleted = manager.delete_unified_item(queue_id).await.unwrap();
    assert!(deleted, "queue item should be deleted");

    // tenant-alpha's row must be gone
    let alpha_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = 'tenant-alpha'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        alpha_count, 0,
        "tenant-alpha's tracked_files row must be removed after its Delete completes"
    );

    // tenant-beta's row must survive
    let beta_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = 'tenant-beta'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        beta_count, 1,
        "tenant-beta's tracked_files row must NOT be removed — it belongs to a different tenant"
    );
}
