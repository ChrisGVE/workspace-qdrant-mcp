//! Enqueue/dequeue and ordering tests for the unified queue.

use super::*;

#[tokio::test]
async fn test_unified_queue_enqueue_dequeue() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_queue.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
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

/// Test FIFO ordering: priority_descending=true -> created_at ASC (oldest first)
#[tokio::test]
async fn test_dequeue_fifo_ordering() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_fifo.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
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

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
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
