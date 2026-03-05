//! Failure handling, retry logic, backoff, and stale lease recovery tests.

use super::*;

#[tokio::test]
async fn test_unified_queue_mark_failed_retry() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_failed.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    let test_pool = pool.clone(); // Keep reference for test-only backoff reset

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
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
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
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

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
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
    assert!(
        items.is_empty(),
        "Item should not be dequeued during backoff"
    );

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
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
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
