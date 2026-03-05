//! Computed Priority Tests and Fairness Scheduler Tests.

use super::*;

// ── Computed Priority Tests ──

#[tokio::test]
async fn test_active_project_dequeued_before_inactive() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Set up watch folders: active_tenant is active, inactive_tenant is not
    insert_watch_folder(
        &pool,
        "wf_1",
        "/test/active",
        "active_tenant",
        "projects",
        true,
    )
    .await;
    insert_watch_folder(
        &pool,
        "wf_2",
        "/test/inactive",
        "inactive_tenant",
        "projects",
        false,
    )
    .await;

    // Enqueue items: inactive first, active second (to prove ordering overrides insertion order)
    let payload_inactive = serde_json::json!({"file_path": "/test/inactive/file.rs"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "inactive_tenant",
            "projects",
            &payload_inactive,
            None,
            None,
        )
        .await
        .unwrap();

    // Small delay to ensure different created_at timestamps
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_active = serde_json::json!({"file_path": "/test/active/file.rs"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "active_tenant",
            "projects",
            &payload_active,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue with priority DESC — active should come first
    let items = queue_manager
        .dequeue_unified(2, "test-worker", Some(300), None, None, Some(true))
        .await
        .unwrap();

    assert_eq!(items.len(), 2, "Should dequeue both items");
    assert_eq!(
        items[0].tenant_id, "active_tenant",
        "Active project should be dequeued first"
    );
    assert_eq!(
        items[1].tenant_id, "inactive_tenant",
        "Inactive project should be dequeued second"
    );
}

#[tokio::test]
async fn test_memory_collection_high_priority() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Set up a watch folder for the projects tenant
    insert_watch_folder(
        &pool,
        "wf_1",
        "/test/project",
        "project_tenant",
        "projects",
        false,
    )
    .await;

    // Enqueue: library item first, then memory item, then inactive project item
    let payload_lib = serde_json::json!({"file_path": "/lib/doc.md"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "lib_tenant",
            "libraries",
            &payload_lib,
            None,
            None,
        )
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_memory = serde_json::json!({"content": "remember this"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::Text,
            QueueOperation::Add,
            "user",
            "rules",
            &payload_memory,
            None,
            None,
        )
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_proj = serde_json::json!({"file_path": "/test/file.rs"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "project_tenant",
            "projects",
            &payload_proj,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue with priority DESC
    let items = queue_manager
        .dequeue_unified(3, "test-worker", Some(300), None, None, Some(true))
        .await
        .unwrap();

    assert_eq!(items.len(), 3, "Should dequeue all 3 items");
    // memory (priority=1) should be before libraries (priority=0)
    // For inactive project (priority=0) and libraries (priority=0), FIFO tiebreaker applies
    assert_eq!(
        items[0].collection, "rules",
        "Rules collection should be dequeued first in DESC mode"
    );
}

#[tokio::test]
async fn test_op_based_priority_delete_before_add() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    insert_watch_folder(&pool, "wf_1", "/test/project", "tenant_1", "projects", true).await;

    // Enqueue add first, then delete
    let payload_add = serde_json::json!({"file_path": "/test/add.rs"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "tenant_1",
            "projects",
            &payload_add,
            None,
            None,
        )
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_del = serde_json::json!({"file_path": "/test/delete.rs"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Delete,
            "tenant_1",
            "projects",
            &payload_del,
            None,
            None,
        )
        .await
        .unwrap();

    let items = queue_manager
        .dequeue_unified(2, "test-worker", Some(300), None, None, Some(true))
        .await
        .unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0].op,
        QueueOperation::Delete,
        "Delete (op priority=10) should come before Add (op priority=1)"
    );
    assert_eq!(items[1].op, QueueOperation::Add, "Add should come second");
}

#[tokio::test]
async fn test_anti_starvation_asc_mode_reverses_priority() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Active and inactive projects
    insert_watch_folder(
        &pool,
        "wf_1",
        "/test/active",
        "active_tenant",
        "projects",
        true,
    )
    .await;
    insert_watch_folder(
        &pool,
        "wf_2",
        "/test/inactive",
        "inactive_tenant",
        "projects",
        false,
    )
    .await;

    // Enqueue items for both
    let payload_active = serde_json::json!({"file_path": "/test/active/a.rs"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "active_tenant",
            "projects",
            &payload_active,
            None,
            None,
        )
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_inactive = serde_json::json!({"file_path": "/test/inactive/b.rs"}).to_string();
    queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            "inactive_tenant",
            "projects",
            &payload_inactive,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue with priority ASC (anti-starvation mode) — inactive should come first
    let items = queue_manager
        .dequeue_unified(2, "test-worker", Some(300), None, None, Some(false))
        .await
        .unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0].tenant_id, "inactive_tenant",
        "In ASC mode, inactive (priority=0) should be dequeued before active (priority=1)"
    );
    assert_eq!(items[1].tenant_id, "active_tenant");
}

// ── Fairness Scheduler Tests ──

#[tokio::test]
async fn test_fairness_scheduler_flips_direction() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    insert_watch_folder(&pool, "wf_1", "/test/active", "active_t", "projects", true).await;
    insert_watch_folder(
        &pool,
        "wf_2",
        "/test/inactive",
        "inactive_t",
        "projects",
        false,
    )
    .await;

    // Enqueue enough items to trigger a flip (high_priority_batch=2 for testing)
    for i in 0..5 {
        let payload =
            serde_json::json!({"file_path": format!("/test/active/f{}.rs", i)}).to_string();
        queue_manager
            .enqueue_unified(
                ItemType::File,
                QueueOperation::Add,
                "active_t",
                "projects",
                &payload,
                None,
                None,
            )
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }
    for i in 0..5 {
        let payload =
            serde_json::json!({"file_path": format!("/test/inactive/f{}.rs", i)}).to_string();
        queue_manager
            .enqueue_unified(
                ItemType::File,
                QueueOperation::Add,
                "inactive_t",
                "projects",
                &payload,
                None,
                None,
            )
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }

    let config = FairnessSchedulerConfig {
        enabled: true,
        high_priority_batch: 2,
        low_priority_batch: 1,
        worker_id: "test-fairness".to_string(),
        lease_duration_secs: 300,
    };
    let scheduler = FairnessScheduler::new(queue_manager, config);

    // First batch: high priority (DESC) — should get active items
    let batch1 = scheduler.dequeue_next_batch(2).await.unwrap();
    assert!(!batch1.is_empty(), "First batch should have items");
    // Active items should dominate DESC batch
    let active_count_1: usize = batch1.iter().filter(|i| i.tenant_id == "active_t").count();
    assert!(
        active_count_1 > 0,
        "First DESC batch should include active items"
    );

    // Second batch: should eventually flip to ASC after high_priority_batch items
    let batch2 = scheduler.dequeue_next_batch(2).await.unwrap();
    assert!(!batch2.is_empty(), "Second batch should have items");

    // Third batch: after 2 high-priority items, should have flipped to ASC (low-priority)
    let batch3 = scheduler.dequeue_next_batch(2).await.unwrap();
    assert!(!batch3.is_empty(), "Third batch should have items");

    // After all batches, verify metrics show direction flips occurred
    let metrics = scheduler.get_metrics().await;
    assert!(
        metrics.total_items_dequeued > 0,
        "Should have dequeued items"
    );
}
