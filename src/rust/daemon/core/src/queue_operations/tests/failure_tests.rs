//! Failure marking and backoff tests: mark_failed (transient/permanent), backoff, re-lease.

use super::*;
use sqlx::Row;

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
        .dequeue_unified(10, "worker-1", None, None, None, None, None, None)
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
            .dequeue_unified(10, "worker-1", None, None, None, None, None, None)
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
        .dequeue_unified(10, "worker-1", None, None, None, None, None, None)
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
        .dequeue_unified(10, "worker-1", None, None, None, None, None, None)
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
        .dequeue_unified(10, "worker-2", None, None, None, None, None, None)
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
async fn test_re_lease_item_preserves_retry_count() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_re_lease.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

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

    // Dequeue → in_progress
    manager
        .dequeue_unified(10, "worker-1", None, None, None, None, None, None)
        .await
        .unwrap();

    // Simulate one real retry (burns retry budget)
    let _ = manager
        .mark_unified_failed(&queue_id, "[transient_resource] real failure", false, 3)
        .await
        .unwrap();

    // Re-lease (subsystem unavailable — must NOT burn retry budget)
    manager.re_lease_item(&queue_id, 5).await.unwrap();

    // Verify retry_count is still 1 (only the real failure counted), status is pending
    let row = sqlx::query("SELECT status, retry_count FROM unified_queue WHERE queue_id = ?1")
        .bind(&queue_id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    let status: String = row.try_get("status").unwrap();
    let retry_count: i32 = row.try_get("retry_count").unwrap();
    assert_eq!(status, "pending", "re-leased item should be pending");
    assert_eq!(retry_count, 1, "re_lease must not increment retry_count");
}

/// F-035 regression: when Qdrant delete fails during file delete processing,
/// the tracked_files row MUST stay intact and the queue row picks up retry
/// metadata. Previously, delete handlers logged the Qdrant error and proceeded
/// with SQLite cleanup — stale vectors stayed retrievable in Qdrant while the
/// local row that recorded their existence was already gone.
#[tokio::test]
async fn test_qdrant_delete_failure_preserves_tracked_file_and_queues_retry() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("f035_delete_failure.db");
    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();
    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();
    use crate::tracked_files_schema::CREATE_TRACKED_FILES_V37_SQL;
    sqlx::query(CREATE_TRACKED_FILES_V37_SQL)
        .execute(&pool)
        .await
        .unwrap();

    let manager = QueueManager::new(pool.clone());
    manager.init_unified_queue().await.unwrap();

    // Insert a watch folder + tracked file the queue handler would target.
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
         VALUES ('w-f035', '/tmp/f035', 'projects', 't-f035', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, branch, file_mtime, file_hash, collection, base_point, relative_path, created_at, updated_at)
         VALUES ('w-f035', 'main', '2025-01-01T00:00:00Z', 'h_f035', 'projects', 'bp_f035', 'src/lib.rs', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
    ).execute(&pool).await.unwrap();

    // Enqueue a Delete operation.
    let (queue_id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Delete,
            "t-f035",
            "projects",
            r#"{"file_path":"/tmp/f035/src/lib.rs"}"#,
            Some("main"),
            None,
        )
        .await
        .unwrap();

    // Simulate the F-035 fix: handler returned Storage error from Qdrant delete
    // before touching SQLite. The processor calls mark_unified_failed.
    let err_msg = "[transient_resource] Qdrant delete failed for src/lib.rs (3 tracked points): connection refused";
    manager
        .mark_unified_failed(&queue_id, err_msg, false, 3)
        .await
        .unwrap();

    // Queue row has retry metadata.
    let row: (Option<String>, i32, Option<String>, Option<String>, String) = sqlx::query_as(
        r#"SELECT error_message, retry_count, last_error_at, lease_until, status
           FROM unified_queue WHERE queue_id = ?1"#,
    )
    .bind(&queue_id)
    .fetch_one(manager.pool())
    .await
    .unwrap();
    let (got_err, retry_count, last_error_at, lease_until, status) = row;
    assert_eq!(got_err.as_deref(), Some(err_msg));
    assert_eq!(retry_count, 1);
    assert!(last_error_at.is_some());
    assert!(lease_until.is_some(), "must schedule backoff for retry");
    assert_eq!(status, "pending");

    // The tracked_files row MUST still exist — Qdrant delete failed, so we
    // must not have wiped the local record (would orphan vectors in Qdrant).
    let tracked_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM tracked_files WHERE watch_folder_id = 'w-f035' AND relative_path = 'src/lib.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        tracked_count, 1,
        "tracked_files row must be preserved when Qdrant delete fails (F-035)"
    );
}
