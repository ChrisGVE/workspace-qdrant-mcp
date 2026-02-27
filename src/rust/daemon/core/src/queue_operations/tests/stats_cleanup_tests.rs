//! Delete, stats, cleanup, and queue depth tests.

use super::*;

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
