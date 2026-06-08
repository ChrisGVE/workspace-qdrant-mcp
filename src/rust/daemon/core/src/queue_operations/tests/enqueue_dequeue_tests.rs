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
        .dequeue_unified(10, "worker-1", Some(300), None, None, None, None, None)
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
        .dequeue_unified(
            3,
            "test-worker",
            Some(300),
            None,
            None,
            Some(true),
            None,
            None,
        )
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
        .dequeue_unified(
            3,
            "test-worker",
            Some(300),
            None,
            None,
            Some(false),
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(items.len(), 3);
    assert_eq!(items[0].queue_id, "lifo-q3"); // newest
    assert_eq!(items[1].queue_id, "lifo-q2");
    assert_eq!(items[2].queue_id, "lifo-q1"); // oldest
}

/// issue-64 task 4: enqueue/dequeue update Prometheus counters.
#[tokio::test]
async fn test_enqueue_dequeue_updates_prometheus_counters() {
    use crate::monitoring::metrics_core::METRICS;

    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("metrics_wire.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let enqueued_before = METRICS
        .unified_queue_enqueues_total
        .with_label_values(&["daemon"])
        .get();
    let dequeued_before = METRICS
        .unified_queue_dequeues_total
        .with_label_values(&["file"])
        .get();

    let (_qid, is_new) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-metrics",
            "test-collection",
            r#"{"file_path":"/tmp/metrics-file.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();
    assert!(is_new);

    let items = manager
        .dequeue_unified(
            10,
            "worker-metrics",
            Some(300),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(items.len(), 1);

    let enqueued_after = METRICS
        .unified_queue_enqueues_total
        .with_label_values(&["daemon"])
        .get();
    let dequeued_after = METRICS
        .unified_queue_dequeues_total
        .with_label_values(&["file"])
        .get();

    // Global METRICS is shared across parallel tests, so only assert the
    // counter moved forward by at least the operations we performed here.
    assert!(enqueued_after > enqueued_before);
    assert!(dequeued_after > dequeued_before);
}

/// issue-64 task 4: depth query returns grouped (item_type, status) counts.
#[tokio::test]
async fn test_depth_by_type_status_excludes_done() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("depth_group.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    for i in 0..3 {
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "tenant-depth",
                "test-collection",
                &format!(r#"{{"file_path":"/tmp/depth-{i}.rs"}}"#),
                Some("main"),
                None,
            )
            .await
            .unwrap();
    }

    let rows = manager
        .get_unified_queue_depth_by_type_status()
        .await
        .unwrap();
    let file_pending = rows
        .iter()
        .find(|(t, s, _)| t == "file" && s == "pending")
        .map(|(_, _, c)| *c)
        .unwrap_or(0);
    assert_eq!(file_pending, 3);
}

/// Regression for issue #59: a single `enqueue_unified_batch` call must
/// commit hundreds of rows in one SQLite transaction (one commit) rather
/// than the N transactions that per-row `enqueue_unified` calls would
/// generate.
///
/// `PRAGMA data_version` changes only when a *different* connection
/// commits — a connection's own writes never bump its view. Both reads
/// therefore pin one dedicated probe connection (held across the batch so
/// the write checks out a different pooled connection); the probe then
/// observes exactly one increment per external commit. Reading through the
/// pool instead would be racy: if the `after` read landed on the very
/// connection that performed the batch, the delta would be 0.
#[tokio::test]
async fn test_enqueue_unified_batch_is_single_transaction() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("batch.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Dedicated probe connection (pool max_connections = 10, so holding it
    // cannot starve the batch write below).
    let mut probe = pool.acquire().await.unwrap();

    let before: i64 = sqlx::query_scalar("PRAGMA data_version")
        .fetch_one(&mut *probe)
        .await
        .unwrap();

    let payloads: Vec<String> = (0..10)
        .map(|i| format!(r#"{{"file_path":"/tmp/batch-{i}.rs"}}"#))
        .collect();

    let inserted = manager
        .enqueue_unified_batch(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-batch",
            "projects",
            &payloads,
            None,
        )
        .await
        .unwrap();

    let after: i64 = sqlx::query_scalar("PRAGMA data_version")
        .fetch_one(&mut *probe)
        .await
        .unwrap();

    assert_eq!(inserted, 10);
    assert_eq!(
        after - before,
        1,
        "batch of 10 should commit exactly once (data_version delta == 1)"
    );

    let total: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-batch'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(total, 10);
}

/// Batch enqueue must skip duplicates via the existing idempotency-key /
/// file_path uniqueness guard just like the single-item path does.
#[tokio::test]
async fn test_enqueue_unified_batch_deduplicates() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("batch_dedup.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let payloads = vec![
        r#"{"file_path":"/tmp/dup.rs"}"#.to_string(),
        r#"{"file_path":"/tmp/dup.rs"}"#.to_string(),
        r#"{"file_path":"/tmp/fresh.rs"}"#.to_string(),
    ];

    let inserted = manager
        .enqueue_unified_batch(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-dedup",
            "projects",
            &payloads,
            None,
        )
        .await
        .unwrap();
    assert_eq!(inserted, 2, "duplicate file_path should be ignored");
}
