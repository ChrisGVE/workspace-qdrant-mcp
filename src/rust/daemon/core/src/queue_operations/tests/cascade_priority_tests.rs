//! Delete cascade and op-priority dequeue tests (Task 44).

use super::*;

/// Create a pool with watch_folders + unified_queue schema, return (pool, QueueManager, TempDir).
async fn setup_cascade_pool() -> (SqlitePool, QueueManager, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("cascade_test.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    (pool, manager, temp_dir)
}

/// Insert a watch_folder row with fixed timestamps.
async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, path: &str, tenant_id: &str) {
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES (?1, ?2, 'projects', ?3, 1,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    )
    .bind(watch_id)
    .bind(path)
    .bind(tenant_id)
    .execute(pool)
    .await
    .unwrap();
}

/// Enqueue N file-add items for a tenant.
async fn enqueue_file_adds(manager: &QueueManager, tenant_id: &str, base_path: &str, count: usize) {
    for i in 0..count {
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                tenant_id,
                "projects",
                &format!(r#"{{"file_path":"{}/file{}.rs"}}"#, base_path, i),
                Some("main"),
                None,
            )
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn test_delete_cascade_purges_pending_items() {
    let (pool, manager, _temp_dir) = setup_cascade_pool().await;
    insert_watch_folder(&pool, "w1", "/test", "cascade-tenant").await;

    enqueue_file_adds(&manager, "cascade-tenant", "/test", 3).await;

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

    // Cascade purge happens during enqueue: 3 add items purged, only the delete remains
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(
        stats.total_items, 1,
        "Only delete should remain after cascade purge"
    );

    // Dequeue should return the delete
    let items = manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    assert_eq!(items.len(), 1, "Only the delete item should be dequeued");
    assert_eq!(
        items[0].op,
        UnifiedOp::Delete,
        "The remaining item should be the delete"
    );
}

#[tokio::test]
async fn test_delete_cascade_does_not_affect_other_tenants() {
    let (pool, manager, _temp_dir) = setup_cascade_pool().await;

    insert_watch_folder(&pool, "w1", "/test1", "tenant-a").await;
    insert_watch_folder(&pool, "w2", "/test2", "tenant-b").await;

    enqueue_file_adds(&manager, "tenant-a", "/test1", 2).await;
    enqueue_file_adds(&manager, "tenant-b", "/test2", 2).await;

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

    // Cascade purge removes tenant-a's 2 adds; remaining: 2 (tenant-b) + 1 (tenant-a delete) = 3
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(
        stats.total_items, 3,
        "tenant-a adds purged by cascade, tenant-b untouched"
    );

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
    let (pool, manager, _temp_dir) = setup_cascade_pool().await;
    insert_watch_folder(&pool, "w1", "/test", "tenant-a").await;

    // Use raw SQL inserts so we can control the queue_id and created_at
    // to avoid timing-based ordering interference
    let base_ts = "2026-01-01T12:00:00.000Z";

    // Insert items with different operations but same tenant/collection
    // Order: add first, then update, scan, delete — but delete should dequeue first due to priority
    let items = vec![
        ("q-add", "file", "add", r#"{"file_path":"/test/add.rs"}"#),
        (
            "q-update",
            "file",
            "update",
            r#"{"file_path":"/test/update.rs"}"#,
        ),
        ("q-scan", "tenant", "scan", r#"{"project_root":"/test"}"#),
        (
            "q-delete",
            "tenant",
            "delete",
            r#"{"project_root":"/test2"}"#,
        ),
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

    // Verify ordering by op priority: delete(10) > add(5) > update(3) > scan(1)
    assert_eq!(
        dequeued[0].op,
        UnifiedOp::Delete,
        "Delete should be dequeued first"
    );
    assert_eq!(
        dequeued[1].op,
        UnifiedOp::Add,
        "Add should be dequeued second"
    );
    assert_eq!(
        dequeued[2].op,
        UnifiedOp::Update,
        "Update should be dequeued third"
    );
    assert_eq!(
        dequeued[3].op,
        UnifiedOp::Scan,
        "Scan should be dequeued last"
    );
}
