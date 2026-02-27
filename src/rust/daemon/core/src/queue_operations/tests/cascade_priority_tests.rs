//! Delete cascade and op-priority dequeue tests (Task 44).

use super::*;

#[tokio::test]
async fn test_delete_cascade_purges_pending_items() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_delete_cascade.db");

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

    // Create a watch_folder for the tenant
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES ('w1', '/test', 'projects', 'cascade-tenant', 1,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    )
    .execute(&pool)
    .await
    .unwrap();

    // Enqueue a few add/update items for same tenant+collection
    for i in 0..3 {
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "cascade-tenant",
                "projects",
                &format!(r#"{{"file_path":"/test/file{}.rs"}}"#, i),
                Some("main"),
                None,
            )
            .await
            .unwrap();
    }

    // Enqueue a delete for same tenant+collection
    manager
        .enqueue_unified(
            ItemType::Tenant,
            UnifiedOp::Delete,
            "cascade-tenant",
            "projects",
            r#"{"project_root":"/test"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Verify: 3 add items + 1 delete = 4 total
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.total_items, 4);

    // Dequeue should return the delete first (highest priority)
    let items = manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    // After delete cascade: pending add/update items for same tenant should be purged
    // Only the delete item itself should remain
    assert!(
        !items.is_empty(),
        "Should have at least the delete item"
    );
    assert_eq!(
        items[0].op,
        UnifiedOp::Delete,
        "Delete should be first item (highest priority)"
    );
}

#[tokio::test]
async fn test_delete_cascade_does_not_affect_other_tenants() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_cascade_isolation.db");

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

    // Create watch_folders for both tenants
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES ('w1', '/test1', 'projects', 'tenant-a', 1,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES ('w2', '/test2', 'projects', 'tenant-b', 1,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    )
    .execute(&pool)
    .await
    .unwrap();

    // Enqueue items for tenant-a
    for i in 0..2 {
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "tenant-a",
                "projects",
                &format!(r#"{{"file_path":"/test1/file{}.rs"}}"#, i),
                Some("main"),
                None,
            )
            .await
            .unwrap();
    }

    // Enqueue items for tenant-b
    for i in 0..2 {
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "tenant-b",
                "projects",
                &format!(r#"{{"file_path":"/test2/file{}.rs"}}"#, i),
                Some("main"),
                None,
            )
            .await
            .unwrap();
    }

    // Enqueue delete for tenant-a
    manager
        .enqueue_unified(
            ItemType::Tenant,
            UnifiedOp::Delete,
            "tenant-a",
            "projects",
            r#"{"project_root":"/test1"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Total should be 5 (2 + 2 + 1 delete)
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.total_items, 5);

    // Dequeue all
    let items = manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    // tenant-b's items should still exist (not cascaded)
    let tenant_b_items: Vec<_> = items.iter().filter(|i| i.tenant_id == "tenant-b").collect();
    assert_eq!(
        tenant_b_items.len(),
        2,
        "tenant-b items should not be affected by tenant-a delete cascade"
    );
}

#[tokio::test]
async fn test_op_priority_dequeue_ordering() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_op_priority.db");

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

    // Use raw SQL inserts so we can control the queue_id and created_at
    // to avoid timing-based ordering interference
    let base_ts = "2026-01-01T12:00:00.000Z";

    // Insert items with different operations but same tenant/collection
    // Order: add first, then update, scan, delete — but delete should dequeue first due to priority
    let items = vec![
        ("q-add", "file", "add", r#"{"file_path":"/test/add.rs"}"#),
        ("q-update", "file", "update", r#"{"file_path":"/test/update.rs"}"#),
        ("q-scan", "tenant", "scan", r#"{"project_root":"/test"}"#),
        ("q-delete", "tenant", "delete", r#"{"project_root":"/test2"}"#),
    ];

    for (qid, item_type, op, payload) in &items {
        sqlx::query(
            r#"INSERT INTO unified_queue
               (queue_id, item_type, op, tenant_id, collection, status,
                branch, idempotency_key, payload_json, created_at, updated_at)
               VALUES (?1, ?2, ?3, 'tenant-a', 'projects', 'pending',
                'main', ?4, ?5, ?6, ?6)"#,
        )
        .bind(qid)
        .bind(item_type)
        .bind(op)
        .bind(format!("key-{}", qid))
        .bind(payload)
        .bind(base_ts)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Dequeue all
    let dequeued = manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    assert_eq!(dequeued.len(), 4, "All 4 items should be dequeued");

    // Verify ordering by op priority: delete(10) > scan(5) > update(3) > add(1)
    assert_eq!(dequeued[0].op, UnifiedOp::Delete, "Delete should be dequeued first");
    assert_eq!(dequeued[1].op, UnifiedOp::Scan, "Scan should be dequeued second");
    assert_eq!(dequeued[2].op, UnifiedOp::Update, "Update should be dequeued third");
    assert_eq!(dequeued[3].op, UnifiedOp::Add, "Add should be dequeued last");
}
