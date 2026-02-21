//! Tests for queue operations.

use super::*;
use crate::queue_config::QueueConnectionConfig;
use crate::unified_queue_schema::{
    DestinationStatus, ItemType, QueueOperation as UnifiedOp, QueueStatus,
};
use sqlx::SqlitePool;
use std::sync::Arc;
use tempfile::tempdir;

async fn apply_sql_script(pool: &SqlitePool, script: &str) -> Result<(), sqlx::Error> {
    let mut conn = pool.acquire().await?;
    let mut statement = String::new();
    let mut in_trigger = false;

    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") {
            continue;
        }

        if trimmed.to_uppercase().starts_with("CREATE TRIGGER") {
            in_trigger = true;
        }

        statement.push_str(line);
        statement.push('\n');

        if in_trigger {
            if trimmed.eq_ignore_ascii_case("END;") || trimmed.eq_ignore_ascii_case("END") {
                in_trigger = false;
                let stmt = statement.trim();
                if !stmt.is_empty() {
                    sqlx::query(stmt).execute(&mut *conn).await?;
                }
                statement.clear();
            }
            continue;
        }

        if trimmed.ends_with(';') {
            let stmt = statement.trim();
            if !stmt.is_empty() {
                sqlx::query(stmt).execute(&mut *conn).await?;
            }
            statement.clear();
        }
    }

    let remainder = statement.trim();
    if !remainder.is_empty() {
        sqlx::query(remainder).execute(&mut *conn).await?;
    }

    Ok(())
}

// NOTE: Legacy queue tests removed per Task 21
// The following tests for ingestion_queue have been removed:
// - test_enqueue_dequeue
// - test_priority_validation
// - test_update_retry_from
// - test_mark_failed
// - test_missing_metadata_queue
//
// See unified queue tests below for spec-compliant queue testing.

// ========================================================================
// Unified Queue Tests (Task 37.21-37.29)
// ========================================================================

#[tokio::test]
async fn test_unified_queue_enqueue_dequeue() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_unified_queue.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(
        &pool,
        include_str!("../schema/watch_folders_schema.sql"),
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
        include_str!("../schema/watch_folders_schema.sql"),
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
        include_str!("../schema/watch_folders_schema.sql"),
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
        .mark_unified_failed(&queue_id, "Test error 1", false)
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
            .mark_unified_failed(&queue_id, &format!("Test error {}", i), false)
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
        include_str!("../schema/watch_folders_schema.sql"),
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
        .mark_unified_failed(&queue_id, "File not found: /test/file.rs", true)
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

    // Initialize schemas
    apply_sql_script(
        &pool,
        include_str!("../schema/watch_folders_schema.sql"),
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
        .mark_unified_failed(&queue_id, "Connection refused", false)
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
        include_str!("../schema/watch_folders_schema.sql"),
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
        include_str!("../schema/watch_folders_schema.sql"),
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
    let cleaned = manager.cleanup_completed_unified_items(Some(24)).await.unwrap();
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

// Concurrent Idempotency Tests (Task 45)
// ========================================================================

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
    let queue_ids: Vec<_> = results
        .into_iter()
        .map(|r| r.unwrap().0)
        .collect();

    let first_id = &queue_ids[0];
    assert!(queue_ids.iter().all(|id| id == first_id),
        "All concurrent enqueues should return the same queue_id");

    // Only one row should exist in the database
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.total_items, 1,
        "Only one item should exist despite concurrent enqueues");
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
    let new_flags: Vec<_> = results
        .into_iter()
        .map(|r| r.unwrap().1)
        .collect();
    assert!(new_flags.iter().all(|&is_new| is_new),
        "All different items should be marked as new");

    // All 10 items should exist
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.total_items, 10,
        "All 10 different items should exist");
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
    let new_flags: Vec<_> = results
        .into_iter()
        .map(|r| r.unwrap().1)
        .collect();
    assert!(new_flags.iter().all(|&is_new| is_new),
        "Different operations should create different items");

    // All 3 items should exist
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.total_items, 3,
        "3 items with different operations should exist");
}

#[tokio::test]
async fn test_idempotency_across_workers() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_idempotency_workers.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
    apply_sql_script(
        &pool,
        include_str!("../schema/watch_folders_schema.sql"),
    )
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
    assert_eq!(items_after.len(), 0, "No items should be available - lease still held by worker-1");
}

// Queue Validation Tests (Task 46)
// ========================================================================

#[tokio::test]
async fn test_validation_empty_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_tenant.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Empty tenant_id should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::EmptyTenantId));
}

#[tokio::test]
async fn test_validation_whitespace_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_ws_tenant.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Whitespace-only tenant_id should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "   ",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::EmptyTenantId));
}

#[tokio::test]
async fn test_validation_empty_collection() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_collection.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Empty collection should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::EmptyCollection));
}

#[tokio::test]
async fn test_validation_invalid_json_payload() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_json.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Invalid JSON should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            "not valid json",
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::InvalidPayloadJson(_)));
}

#[tokio::test]
async fn test_validation_file_missing_file_path() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_file_path.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // File item without file_path should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"other_field":"value"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "file" && field == "file_path"
    ));
}

#[tokio::test]
async fn test_validation_content_missing_content() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_content.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Content item without content field should fail
    let result = manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"source_type":"mcp"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "text" && field == "content"
    ));
}

#[tokio::test]
async fn test_validation_delete_document_missing_document_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_delete_doc.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // DeleteDocument without document_id should fail
    let result = manager
        .enqueue_unified(
            ItemType::Doc,
            UnifiedOp::Delete,
            "test-tenant",
            "test-collection",
            r#"{"point_ids":["abc"]}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "doc" && field == "document_id"
    ));
}

#[tokio::test]
async fn test_validation_file_rename_missing_old_path() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_rename.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // File rename without old_path should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Rename,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/new.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "file" && field == "old_path"
    ));
}

#[tokio::test]
async fn test_validation_valid_items_pass() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_valid.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Valid file item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());

    // Valid content item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"content":"test content","source_type":"mcp"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());

    // Valid file rename item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Rename,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/new.rs","old_path":"/test/old.rs"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());

    // Valid doc delete item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::Doc,
            UnifiedOp::Delete,
            "test-tenant",
            "test-collection",
            r#"{"document_id":"doc-123"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_validation_empty_string_in_required_field() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_empty_field.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // File with empty file_path should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":""}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "file" && field == "file_path"
    ));
}

/// Test FIFO ordering: priority_descending=true -> created_at ASC (oldest first)
#[tokio::test]
async fn test_dequeue_fifo_ordering() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_fifo.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql"))
        .await.unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Create an inactive project watch_folder
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES ('w1', '/test', 'projects', 'tenant-a', 0,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    ).execute(&pool).await.unwrap();

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
        .execute(&pool).await.unwrap();
    }

    // DESC direction -> FIFO (oldest first)
    let items = manager
        .dequeue_unified(3, "test-worker", Some(300), None, None, Some(true))
        .await.unwrap();

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

    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql"))
        .await.unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Create an inactive project watch_folder
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
           created_at, updated_at)
           VALUES ('w1', '/test', 'projects', 'tenant-a', 0,
           '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
    ).execute(&pool).await.unwrap();

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
        .execute(&pool).await.unwrap();
    }

    // ASC direction -> LIFO (newest first)
    let items = manager
        .dequeue_unified(3, "test-worker", Some(300), None, None, Some(false))
        .await.unwrap();

    assert_eq!(items.len(), 3);
    assert_eq!(items[0].queue_id, "lifo-q3"); // newest
    assert_eq!(items[1].queue_id, "lifo-q2");
    assert_eq!(items[2].queue_id, "lifo-q1"); // oldest
}

// ===== Library document payload validation =====

#[test]
fn test_validate_library_document_payload_valid() {
    let payload = serde_json::json!({
        "document_path": "/docs/report.pdf",
        "library_name": "internal-docs",
        "document_type": "page_based",
        "source_format": "pdf",
        "doc_id": "550e8400-e29b-41d4-a716-446655440000",
    });
    assert!(QueueManager::validate_library_document_payload(&payload).is_ok());
}

#[test]
fn test_validate_library_document_payload_stream_based() {
    let payload = serde_json::json!({
        "document_path": "/books/novel.epub",
        "library_name": "ebooks",
        "document_type": "stream_based",
        "source_format": "epub",
        "doc_id": "uuid-here",
    });
    assert!(QueueManager::validate_library_document_payload(&payload).is_ok());
}

#[test]
fn test_validate_library_document_payload_missing_field() {
    let payload = serde_json::json!({
        "document_path": "/docs/report.pdf",
        "library_name": "internal-docs",
        // missing document_type, source_format, doc_id
    });
    let result = QueueManager::validate_library_document_payload(&payload);
    assert!(result.is_err());
}

#[test]
fn test_validate_library_document_payload_invalid_document_type() {
    let payload = serde_json::json!({
        "document_path": "/docs/report.pdf",
        "library_name": "internal-docs",
        "document_type": "unknown_type",
        "source_format": "pdf",
        "doc_id": "uuid-here",
    });
    let result = QueueManager::validate_library_document_payload(&payload);
    assert!(result.is_err());
}

#[test]
fn test_validate_library_document_payload_empty_fields() {
    let payload = serde_json::json!({
        "document_path": "",
        "library_name": "docs",
        "document_type": "page_based",
        "source_format": "pdf",
        "doc_id": "uuid",
    });
    let result = QueueManager::validate_library_document_payload(&payload);
    assert!(result.is_err()); // document_path is empty
}

// ========================================================================
// Task 44: Delete cascade and op-priority dequeue tests
// ========================================================================

#[tokio::test]
async fn test_delete_cascade_purges_pending_items() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_delete_cascade.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(
        &pool,
        include_str!("../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue 5 add items for the same tenant
    for i in 0..5 {
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "tenant-to-delete",
                "projects",
                &format!(r#"{{"file_path":"/file_{}.rs"}}"#, i),
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Verify 5 pending items
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.pending_items, 5);

    // Enqueue a tenant delete -- this should cascade-purge the 5 add items
    let (delete_id, is_new) = manager
        .enqueue_unified(
            ItemType::Tenant,
            UnifiedOp::Delete,
            "tenant-to-delete",
            "projects",
            r#"{"tenant_id":"tenant-to-delete"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    assert!(is_new);

    // Only the delete item should remain
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.pending_items, 1, "Only the delete item should remain");

    // Dequeue and verify it's the delete
    let items = manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].queue_id, delete_id);
    assert_eq!(items[0].op, UnifiedOp::Delete);
}

#[tokio::test]
async fn test_delete_cascade_does_not_affect_other_tenants() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_delete_cascade_isolation.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(
        &pool,
        include_str!("../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue items for two different tenants
    manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-a",
            "projects",
            r#"{"file_path":"/a/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-b",
            "projects",
            r#"{"file_path":"/b/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Delete tenant-a -- should NOT affect tenant-b
    manager
        .enqueue_unified(
            ItemType::Tenant,
            UnifiedOp::Delete,
            "tenant-a",
            "projects",
            r#"{"tenant_id":"tenant-a"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Should have 2 items: tenant-a delete + tenant-b file add
    let stats = manager.get_unified_queue_stats().await.unwrap();
    assert_eq!(stats.pending_items, 2);
}

#[tokio::test]
async fn test_op_priority_dequeue_ordering() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_op_priority.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(
        &pool,
        include_str!("../schema/watch_folders_schema.sql"),
    )
    .await
    .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue items with different ops for DIFFERENT tenants
    // (using different tenants so cascade doesn't purge them)
    // Add op (lowest op priority)
    manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-add",
            "projects",
            r#"{"file_path":"/add/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Update op (medium op priority)
    manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Update,
            "tenant-update",
            "projects",
            r#"{"file_path":"/update/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Delete op (highest op priority)
    manager
        .enqueue_unified(
            ItemType::Tenant,
            UnifiedOp::Delete,
            "tenant-delete",
            "projects",
            r#"{"tenant_id":"tenant-delete"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Scan op (mid-high op priority)
    manager
        .enqueue_unified(
            ItemType::Tenant,
            UnifiedOp::Scan,
            "tenant-scan",
            "projects",
            r#"{"tenant_id":"tenant-scan"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Dequeue all -- delete should come first, then scan, update, add
    let items = manager
        .dequeue_unified(10, "worker-1", None, None, None, None)
        .await
        .unwrap();

    assert_eq!(items.len(), 4);
    assert_eq!(items[0].op, UnifiedOp::Delete, "Delete should be dequeued first");
    assert_eq!(items[1].op, UnifiedOp::Scan, "Scan should be dequeued second");
    assert_eq!(items[2].op, UnifiedOp::Update, "Update should be dequeued third");
    assert_eq!(items[3].op, UnifiedOp::Add, "Add should be dequeued last");
}

// ========================================================================
// Reference Counting & Decision Tests (Task 5)
// ========================================================================

#[tokio::test]
async fn test_has_other_references_single_watch() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_refcount.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    // Set up watch_folders and tracked_files schemas
    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();

    use crate::tracked_files_schema::CREATE_TRACKED_FILES_SQL;
    sqlx::query(CREATE_TRACKED_FILES_SQL).execute(&pool).await.unwrap();

    let manager = QueueManager::new(pool.clone());

    // Insert a watch folder and a tracked file with base_point
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/project1', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, base_point, relative_path, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', 'projects', 'bp_abc123', 'src/main.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    // Single watch folder -- no other references
    let has_refs = manager.has_other_references("bp_abc123", "w1").await.unwrap();
    assert!(!has_refs, "Single watch folder should have no other references");
}

#[tokio::test]
async fn test_has_other_references_two_watches() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_refcount2.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();

    use crate::tracked_files_schema::CREATE_TRACKED_FILES_SQL;
    sqlx::query(CREATE_TRACKED_FILES_SQL).execute(&pool).await.unwrap();

    let manager = QueueManager::new(pool.clone());

    // Insert two watch folders
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w1', '/tmp/clone1', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w2', '/tmp/clone2', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    // Both reference the same base_point (same file version in two clones)
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, base_point, relative_path, created_at, updated_at)
         VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', 'projects', 'bp_shared', 'src/main.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, base_point, relative_path, created_at, updated_at)
         VALUES ('w2', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', 'projects', 'bp_shared', 'src/main.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    // w1 should see w2 as another reference
    let has_refs = manager.has_other_references("bp_shared", "w1").await.unwrap();
    assert!(has_refs, "Should detect w2 as another reference");

    // w2 should see w1 as another reference
    let has_refs2 = manager.has_other_references("bp_shared", "w2").await.unwrap();
    assert!(has_refs2, "Should detect w1 as another reference");

    // Now delete w2's tracked file -- w1 should have no more references
    sqlx::query("DELETE FROM tracked_files WHERE watch_folder_id = 'w2'")
        .execute(&pool).await.unwrap();

    let has_refs3 = manager.has_other_references("bp_shared", "w1").await.unwrap();
    assert!(!has_refs3, "After removing w2's file, w1 should have no other references");
}

#[tokio::test]
async fn test_store_queue_decision() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_decision.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Enqueue an item
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File, UnifiedOp::Update, "t1", "projects",
            r#"{"file_path":"/test/file.rs"}"#, Some("main"), None,
        )
        .await
        .unwrap();

    // Store a decision
    let decision = wqm_common::queue_types::QueueDecision {
        delete_old: true,
        old_base_point: Some("bp_old_123".to_string()),
        new_base_point: "bp_new_456".to_string(),
        old_file_hash: Some("hash_old".to_string()),
        new_file_hash: "hash_new".to_string(),
    };

    manager.store_queue_decision(&queue_id, &decision).await.unwrap();

    // Verify stored correctly
    let stored_json: Option<String> = sqlx::query_scalar(
        "SELECT decision_json FROM unified_queue WHERE queue_id = ?1"
    )
    .bind(&queue_id)
    .fetch_one(&pool)
    .await
    .unwrap();

    assert!(stored_json.is_some(), "decision_json should be stored");
    let stored: wqm_common::queue_types::QueueDecision =
        serde_json::from_str(stored_json.as_ref().unwrap()).unwrap();
    assert!(stored.delete_old);
    assert_eq!(stored.old_base_point.as_deref(), Some("bp_old_123"));
    assert_eq!(stored.new_base_point, "bp_new_456");

    // Verify statuses set to pending
    let qdrant_status: String = sqlx::query_scalar(
        "SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1"
    )
    .bind(&queue_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(qdrant_status, "pending");
}

#[tokio::test]
async fn test_update_destination_status() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_dest_status.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File, UnifiedOp::Add, "t1", "projects",
            r#"{"file_path":"/test/file.rs"}"#, Some("main"), None,
        )
        .await
        .unwrap();

    // Update qdrant status to done
    manager.update_destination_status(&queue_id, "qdrant", DestinationStatus::Done)
        .await
        .unwrap();

    let qdrant_status: String = sqlx::query_scalar(
        "SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1"
    )
    .bind(&queue_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(qdrant_status, "done");

    // Update search status to failed
    manager.update_destination_status(&queue_id, "search", DestinationStatus::Failed)
        .await
        .unwrap();

    let search_status: String = sqlx::query_scalar(
        "SELECT search_status FROM unified_queue WHERE queue_id = ?1"
    )
    .bind(&queue_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(search_status, "failed");

    // Invalid destination should error
    let err = manager.update_destination_status(&queue_id, "invalid", DestinationStatus::Done).await;
    assert!(err.is_err(), "Invalid destination should return error");
}

#[tokio::test]
async fn test_check_and_finalize_both_done() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("finalize_both.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager.enqueue_unified(
        ItemType::File, UnifiedOp::Add, "fin-tenant", "projects",
        r#"{"file_path":"/test/fin.rs"}"#, Some("main"), None,
    ).await.unwrap();

    // Initially both pending -- should be InProgress
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::InProgress);

    // Mark qdrant done
    manager.update_destination_status(&queue_id, "qdrant", DestinationStatus::Done).await.unwrap();
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::InProgress);

    // Mark search done -- both done -> overall Done
    manager.update_destination_status(&queue_id, "search", DestinationStatus::Done).await.unwrap();
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Done);

    // Verify overall status was updated in DB
    let row: (String,) = sqlx::query_as("SELECT status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(row.0, "done");
}

#[tokio::test]
async fn test_check_and_finalize_partial_failure() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("finalize_fail.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager.enqueue_unified(
        ItemType::File, UnifiedOp::Add, "fin-tenant2", "projects",
        r#"{"file_path":"/test/fail.rs"}"#, Some("main"), None,
    ).await.unwrap();

    // Qdrant succeeds, search fails
    manager.update_destination_status(&queue_id, "qdrant", DestinationStatus::Done).await.unwrap();
    manager.update_destination_status(&queue_id, "search", DestinationStatus::Failed).await.unwrap();

    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Failed);

    // Verify overall status was updated in DB
    let row: (String,) = sqlx::query_as("SELECT status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(row.0, "failed");
}

#[tokio::test]
async fn test_check_and_finalize_nonexistent_item() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("finalize_missing.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let err = manager.check_and_finalize("nonexistent-id").await;
    assert!(err.is_err());
}

#[tokio::test]
async fn test_ensure_destinations_resolved_both_pending() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("resolve_pending.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager.enqueue_unified(
        ItemType::Tenant, UnifiedOp::Scan, "resolve-tenant", "projects",
        r#"{"project_root":"/test"}"#, None, None,
    ).await.unwrap();

    // Both statuses start as pending -- ensure_destinations_resolved should set both to done
    manager.ensure_destinations_resolved(&queue_id).await.unwrap();

    let qs: String = sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id).fetch_one(manager.pool()).await.unwrap();
    let ss: String = sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id).fetch_one(manager.pool()).await.unwrap();
    assert_eq!(qs, "done");
    assert_eq!(ss, "done");

    // check_and_finalize should now return Done
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Done);
}

#[tokio::test]
async fn test_ensure_destinations_resolved_preserves_explicit_status() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("resolve_preserve.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager.enqueue_unified(
        ItemType::File, UnifiedOp::Add, "resolve-tenant2", "projects",
        r#"{"file_path":"/test/file.rs"}"#, Some("main"), None,
    ).await.unwrap();

    // Explicitly set qdrant to failed, leave search as pending
    manager.update_destination_status(&queue_id, "qdrant", DestinationStatus::Failed).await.unwrap();

    // ensure_destinations_resolved should preserve failed qdrant, resolve pending search
    manager.ensure_destinations_resolved(&queue_id).await.unwrap();

    let qs: String = sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id).fetch_one(manager.pool()).await.unwrap();
    let ss: String = sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id).fetch_one(manager.pool()).await.unwrap();
    assert_eq!(qs, "failed", "Failed status must be preserved");
    assert_eq!(ss, "done", "Pending status must be resolved to done");

    // check_and_finalize should now return Failed (qdrant is failed)
    let status = manager.check_and_finalize(&queue_id).await.unwrap();
    assert_eq!(status, QueueStatus::Failed);
}

#[tokio::test]
async fn test_ensure_destinations_resolved_preserves_done() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("resolve_done.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../schema/watch_folders_schema.sql")).await.unwrap();
    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (queue_id, _) = manager.enqueue_unified(
        ItemType::File, UnifiedOp::Add, "resolve-tenant3", "projects",
        r#"{"file_path":"/test/done.rs"}"#, Some("main"), None,
    ).await.unwrap();

    // Explicitly set both to done
    manager.update_destination_status(&queue_id, "qdrant", DestinationStatus::Done).await.unwrap();
    manager.update_destination_status(&queue_id, "search", DestinationStatus::Done).await.unwrap();

    // ensure_destinations_resolved should be a no-op
    manager.ensure_destinations_resolved(&queue_id).await.unwrap();

    let qs: String = sqlx::query_scalar("SELECT qdrant_status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id).fetch_one(manager.pool()).await.unwrap();
    let ss: String = sqlx::query_scalar("SELECT search_status FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id).fetch_one(manager.pool()).await.unwrap();
    assert_eq!(qs, "done");
    assert_eq!(ss, "done");
}
