//! Concurrent idempotency tests (Task 45).

use super::*;

#[tokio::test]
async fn test_concurrent_enqueue_idempotency() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_concurrent_idempotency.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = Arc::new(QueueManager::new(pool));
    manager.init_unified_queue().await.unwrap();

    // Spawn 10 concurrent enqueue operations for the same item
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let mgr = Arc::clone(&manager);
            tokio::spawn(async move {
                mgr.enqueue_unified(
                    ItemType::File,
                    UnifiedOp::Add,
                    "test-tenant",
                    "test-collection",
                    r#"{"file_path":"/test/concurrent_file.rs"}"#,
                    Some("main"),
                    Some(&format!(r#"{{"worker":{}}}"#, i)),
                )
                .await
            })
        })
        .collect();

    // Wait for all to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // All should succeed (no errors from UNIQUE constraint violation)
    assert!(results.iter().all(|r| r.is_ok()));

    // All should return the same queue_id
    let queue_ids: Vec<_> = results.into_iter().map(|r| r.unwrap().0).collect();

    let first_id = &queue_ids[0];
    assert!(
        queue_ids.iter().all(|id| id == first_id),
        "All concurrent enqueues should return the same queue_id"
    );

    // Only one row should exist in the database
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(
        stats.total_items, 1,
        "Only one item should exist despite concurrent enqueues"
    );
}

#[tokio::test]
async fn test_concurrent_enqueue_different_items() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_concurrent_different.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = Arc::new(QueueManager::new(pool));
    manager.init_unified_queue().await.unwrap();

    // Spawn 10 concurrent enqueue operations for different items
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let mgr = Arc::clone(&manager);
            tokio::spawn(async move {
                mgr.enqueue_unified(
                    ItemType::File,
                    UnifiedOp::Add,
                    "test-tenant",
                    "test-collection",
                    &format!(r#"{{"file_path":"/test/file_{}.rs"}}"#, i),
                    Some("main"),
                    None,
                )
                .await
            })
        })
        .collect();

    // Wait for all to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // All should succeed
    assert!(results.iter().all(|r| r.is_ok()));

    // All should be new items
    let new_flags: Vec<_> = results.into_iter().map(|r| r.unwrap().1).collect();
    assert!(
        new_flags.iter().all(|&is_new| is_new),
        "All different items should be marked as new"
    );

    // All 10 items should exist
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.total_items, 10, "All 10 different items should exist");
}

#[tokio::test]
async fn test_concurrent_enqueue_mixed_operations() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_concurrent_mixed.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = Arc::new(QueueManager::new(pool));
    manager.init_unified_queue().await.unwrap();

    // Enqueue the same content with different operations (each should be unique)
    // Note: Uses ItemType::Text (not File) to avoid per-file UNIQUE constraint (Task 22)
    let ops = vec![
        (UnifiedOp::Add, "ingest"),
        (UnifiedOp::Update, "update"),
        (UnifiedOp::Delete, "delete"),
    ];

    let handles: Vec<_> = ops
        .into_iter()
        .map(|(op, _name)| {
            let mgr = Arc::clone(&manager);
            tokio::spawn(async move {
                mgr.enqueue_unified(
                    ItemType::Text,
                    op,
                    "test-tenant",
                    "test-collection",
                    r#"{"content":"test content","source_type":"test"}"#,
                    Some("main"),
                    None,
                )
                .await
            })
        })
        .collect();

    // Wait for all to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // All should succeed
    assert!(results.iter().all(|r| r.is_ok()));

    // All should be new (different operations = different idempotency keys)
    let new_flags: Vec<_> = results.into_iter().map(|r| r.unwrap().1).collect();
    assert!(
        new_flags.iter().all(|&is_new| is_new),
        "Different operations should create different items"
    );

    // All 3 items should exist
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(
        stats.total_items, 3,
        "3 items with different operations should exist"
    );
}

#[tokio::test]
async fn test_idempotency_across_workers() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_idempotency_workers.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = Arc::new(QueueManager::new(pool));
    manager.init_unified_queue().await.unwrap();

    // First enqueue
    let (queue_id1, is_new1) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/worker_test.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    assert!(is_new1);

    // Dequeue with worker-1
    let items = manager
        .dequeue_unified(10, "worker-1", Some(300), None, None, None)
        .await
        .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].worker_id, Some("worker-1".to_string()));

    // Try to enqueue same item again (while worker-1 is processing)
    let (queue_id2, is_new2) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/worker_test.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Should return same queue_id, not new
    assert_eq!(queue_id1, queue_id2);
    assert!(!is_new2);

    // Item should still be in_progress with worker-1's lease
    let items_after = manager
        .dequeue_unified(10, "worker-2", Some(300), None, None, None)
        .await
        .unwrap();
    assert_eq!(
        items_after.len(),
        0,
        "No items should be available - lease still held by worker-1"
    );
}
