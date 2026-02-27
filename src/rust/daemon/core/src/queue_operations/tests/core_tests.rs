//! Core queue operation tests: enqueue/dequeue, delete, fail/retry, stats, cleanup, depth.

use super::*;

#[tokio::test]
async fn test_unified_queue_enqueue_dequeue() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_queue.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue an item
    let (queue_id, is_new) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    assert!(is_new);
    assert!(!queue_id.is_empty());

    // Enqueue same item again (idempotent)
    let (queue_id2, is_new2) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(queue_id, queue_id2);
    assert!(!is_new2); // Should be duplicate

    // Dequeue
    let items = manager
        .dequeue_unified(10, "worker-1", Some(300), None, None, None)
        .await
        .unwrap();

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].queue_id, queue_id);
    assert_eq!(items[0].status, QueueStatus::InProgress);
    assert_eq!(items[0].worker_id, Some("worker-1".to_string()));
}

#[tokio::test]
async fn test_unified_queue_delete_item() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_delete.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
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
        .dequeue_unified(10, "worker-1", None, None, None, None)
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
async fn test_unified_queue_mark_failed_retry() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_failed.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    let test_pool = pool.clone(); // Keep reference for test-only backoff reset

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue
    manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    // First failure (transient) - should retry with backoff
    let will_retry = manager
        .mark_unified_failed(&queue_id, "Test error 1", false, 3)
        .await
        .unwrap();
    assert!(will_retry);

    // Check it's back to pending (with backoff lease_until)
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.pending_items, 1);

    // Item has backoff, so dequeue won't return it. Reset lease_until for test.
    sqlx::query("UPDATE unified_queue SET lease_until = NULL WHERE queue_id = ?1")
        .bind(&queue_id)
        .execute(&test_pool)
        .await
        .unwrap();

    // Dequeue again and fail until max retries
    for i in 2..=3 {
        manager
            .dequeue_unified(10, "worker-1", None, None, None, None)
            .await
            .unwrap();
        let will_retry = manager
            .mark_unified_failed(&queue_id, &format!("Test error {}", i), false, 3)
            .await
            .unwrap();

        if i < 3 {
            assert!(will_retry);
            // Clear backoff for next test iteration
            sqlx::query("UPDATE unified_queue SET lease_until = NULL WHERE queue_id = ?1")
                .bind(&queue_id)
                .execute(&test_pool)
                .await
                .unwrap();
        } else {
            assert!(!will_retry); // Max retries exceeded
        }
    }

    // Verify permanently failed
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.failed_items, 1);
    assert_eq!(stats.pending_items, 0);
}

#[tokio::test]
async fn test_unified_queue_mark_failed_permanent() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_permanent.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas
    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue
    manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    // Permanent failure - should NOT retry even though retries remain
    let will_retry = manager
        .mark_unified_failed(&queue_id, "File not found: /test/file.rs", true, 3)
        .await
        .unwrap();
    assert!(!will_retry, "Permanent errors should not retry");

    // Verify immediately failed (not pending)
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.failed_items, 1);
    assert_eq!(stats.pending_items, 0);
}

#[tokio::test]
async fn test_unified_queue_backoff_prevents_immediate_dequeue() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_backoff.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue
    manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    // Transient failure with backoff
    let will_retry = manager
        .mark_unified_failed(&queue_id, "Connection refused", false, 3)
        .await
        .unwrap();
    assert!(will_retry);

    // Try to dequeue immediately - should get nothing (item is in backoff)
    let items = manager
        .dequeue_unified(10, "worker-2", None, None, None, None)
        .await
        .unwrap();
    assert!(items.is_empty(), "Item should not be dequeued during backoff");

    // Verify item is still pending (not lost)
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.pending_items, 1);
    assert_eq!(stats.failed_items, 0);
}

#[tokio::test]
async fn test_unified_queue_recover_stale_leases() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_stale.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue
    manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue with very short lease (1 second)
    manager
        .dequeue_unified(10, "worker-1", Some(1), None, None, None)
        .await
        .unwrap();

    // Wait for lease to expire
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Recover stale leases
    let recovered = manager.recover_stale_unified_leases().await.unwrap();
    assert_eq!(recovered, 1);

    // Verify it's back to pending
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.pending_items, 1);
    assert_eq!(stats.in_progress_items, 0);
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
    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
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
        .dequeue_unified(10, "worker-1", None, None, None, None)
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
    let depth = manager
        .get_unified_queue_depth(None, None)
        .await
        .unwrap();
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

/// Test FIFO ordering: priority_descending=true -> created_at ASC (oldest first)
#[tokio::test]
async fn test_dequeue_fifo_ordering() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_fifo.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Create an inactive project watch_folder
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES ('w1', '/test', 'projects', 'tenant-a', 0,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    )
    .execute(&pool)
    .await
    .unwrap();

    // Enqueue 3 items with staggered timestamps (old -> new)
    for i in 1..=3 {
        let ts = format!("2026-01-0{}T00:00:00.000Z", i);
        sqlx::query(
            r#"INSERT INTO unified_queue
               (queue_id, item_type, op, tenant_id, collection, status,
                branch, idempotency_key, payload_json, created_at, updated_at)
               VALUES (?1, 'file', 'add', 'tenant-a', 'projects', 'pending',
                'main', ?2, ?3, ?4, ?4)"#,
        )
        .bind(format!("fifo-q{}", i))
        .bind(format!("key-fifo-{}", i))
        .bind(format!(r#"{{"file_path":"/test/file{}.rs"}}"#, i))
        .bind(&ts)
        .execute(&pool)
        .await
        .unwrap();
    }

    // DESC direction -> FIFO (oldest first)
    let items = manager
        .dequeue_unified(3, "test-worker", Some(300), None, None, Some(true))
        .await
        .unwrap();

    assert_eq!(items.len(), 3);
    assert_eq!(items[0].queue_id, "fifo-q1"); // oldest
    assert_eq!(items[1].queue_id, "fifo-q2");
    assert_eq!(items[2].queue_id, "fifo-q3"); // newest
}

/// Test LIFO ordering: priority_descending=false -> created_at DESC (newest first)
#[tokio::test]
async fn test_dequeue_lifo_ordering() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_lifo.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(
        &pool,
        include_str!("../../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Create an inactive project watch_folder
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES ('w1', '/test', 'projects', 'tenant-a', 0,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    )
    .execute(&pool)
    .await
    .unwrap();

    // Enqueue 3 items with staggered timestamps (old -> new)
    for i in 1..=3 {
        let ts = format!("2026-01-0{}T00:00:00.000Z", i);
        sqlx::query(
            r#"INSERT INTO unified_queue
               (queue_id, item_type, op, tenant_id, collection, status,
                branch, idempotency_key, payload_json, created_at, updated_at)
               VALUES (?1, 'file', 'add', 'tenant-a', 'projects', 'pending',
                'main', ?2, ?3, ?4, ?4)"#,
        )
        .bind(format!("lifo-q{}", i))
        .bind(format!("key-lifo-{}", i))
        .bind(format!(r#"{{"file_path":"/test/lifo{}.rs"}}"#, i))
        .bind(&ts)
        .execute(&pool)
        .await
        .unwrap();
    }

    // ASC direction -> LIFO (newest first)
    let items = manager
        .dequeue_unified(3, "test-worker", Some(300), None, None, Some(false))
        .await
        .unwrap();

    assert_eq!(items.len(), 3);
    assert_eq!(items[0].queue_id, "lifo-q3"); // newest
    assert_eq!(items[1].queue_id, "lifo-q2");
    assert_eq!(items[2].queue_id, "lifo-q1"); // oldest
}
