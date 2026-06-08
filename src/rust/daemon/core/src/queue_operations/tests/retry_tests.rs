//! Resurrection, stale-lease recovery, and resurrection-count tracking tests.

use super::*;
use serde_json;
use sqlx::Row;

#[tokio::test]
async fn test_resurrect_failed_transient_resets_items() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_resurrect.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Enqueue two items
    let (transient_id, _) = manager
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
    let (permanent_id, _) = manager
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

    // Force both to failed state with different error prefixes
    sqlx::query("UPDATE unified_queue SET status='failed', error_message=?1 WHERE queue_id=?2")
        .bind("[transient_resource] FastEmbed init failed")
        .bind(&transient_id)
        .execute(manager.pool())
        .await
        .unwrap();

    sqlx::query("UPDATE unified_queue SET status='failed', error_message=?1 WHERE queue_id=?2")
        .bind("[permanent_data] bad json payload")
        .bind(&permanent_id)
        .execute(manager.pool())
        .await
        .unwrap();

    // Run resurrection — only transient item should be reset
    let (resurrected, exhausted) = manager.resurrect_failed_transient(5).await.unwrap();
    assert_eq!(
        resurrected, 1,
        "only the transient item should be resurrected"
    );
    assert_eq!(exhausted, 0, "no items should be exhausted yet");

    let transient_row =
        sqlx::query("SELECT status, retry_count FROM unified_queue WHERE queue_id=?1")
            .bind(&transient_id)
            .fetch_one(manager.pool())
            .await
            .unwrap();
    let status: String = transient_row.try_get("status").unwrap();
    let retry_count: i32 = transient_row.try_get("retry_count").unwrap();
    assert_eq!(status, "pending", "transient item should be pending");
    assert_eq!(retry_count, 0, "retry_count should be reset to 0");

    let permanent_row = sqlx::query("SELECT status FROM unified_queue WHERE queue_id=?1")
        .bind(&permanent_id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    let status: String = permanent_row.try_get("status").unwrap();
    assert_eq!(status, "failed", "permanent item should remain failed");
}

#[tokio::test]
async fn test_resurrect_failed_transient_infrastructure() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_resurrect_infra.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-x",
            "projects",
            r#"{"file_path":"/x/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Transient infrastructure failure (Qdrant was down)
    sqlx::query("UPDATE unified_queue SET status='failed', error_message=?1 WHERE queue_id=?2")
        .bind("[transient_infrastructure] Qdrant connection refused")
        .bind(&id)
        .execute(manager.pool())
        .await
        .unwrap();

    let (resurrected, _exhausted) = manager.resurrect_failed_transient(5).await.unwrap();
    assert_eq!(
        resurrected, 1,
        "transient_infrastructure item should be resurrected"
    );

    let row = sqlx::query("SELECT status FROM unified_queue WHERE queue_id=?1")
        .bind(&id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    let status: String = row.try_get("status").unwrap();
    assert_eq!(status, "pending");
}

#[tokio::test]
async fn test_resurrect_exhaustion_after_max_resurrections() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_resurrect_exhaust.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-z",
            "projects",
            r#"{"file_path":"/z/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Set as failed with transient error and resurrection_count already at limit
    sqlx::query(
        "UPDATE unified_queue SET status='failed', error_message=?1, metadata=?2 WHERE queue_id=?3",
    )
    .bind("[transient_infrastructure] Qdrant timeout")
    .bind(r#"{"resurrection_count":3}"#)
    .bind(&id)
    .execute(manager.pool())
    .await
    .unwrap();

    // max_resurrections=3 — item at count 3 should be exhausted
    let (resurrected, exhausted) = manager.resurrect_failed_transient(3).await.unwrap();
    assert_eq!(resurrected, 0);
    assert_eq!(exhausted, 1);

    // Verify error message was updated to permanent_exhausted
    let row = sqlx::query("SELECT status, error_message FROM unified_queue WHERE queue_id=?1")
        .bind(&id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    let status: String = row.try_get("status").unwrap();
    let error_msg: String = row.try_get("error_message").unwrap();
    assert_eq!(status, "failed", "exhausted items stay in failed state");
    assert!(
        error_msg.starts_with("[permanent_exhausted]"),
        "error should be prefixed with [permanent_exhausted], got: {}",
        error_msg
    );
}

#[tokio::test]
async fn test_resurrect_increments_resurrection_count() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_resurrect_count.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    apply_sql_script(&pool, include_str!("../../schema/watch_folders_schema.sql"))
        .await
        .unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    let (id, _) = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant-w",
            "projects",
            r#"{"file_path":"/w/file.rs"}"#,
            None,
            None,
        )
        .await
        .unwrap();

    // Set as failed transient with no prior resurrections
    sqlx::query("UPDATE unified_queue SET status='failed', error_message=?1 WHERE queue_id=?2")
        .bind("[transient_infrastructure] Qdrant timeout")
        .bind(&id)
        .execute(manager.pool())
        .await
        .unwrap();

    // First resurrection
    let (resurrected, _) = manager.resurrect_failed_transient(5).await.unwrap();
    assert_eq!(resurrected, 1);

    // Check resurrection_count is now 1
    let row = sqlx::query("SELECT metadata FROM unified_queue WHERE queue_id=?1")
        .bind(&id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    let metadata_str: String = row.try_get("metadata").unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();
    assert_eq!(
        metadata["resurrection_count"], 1,
        "resurrection_count should be 1 after first resurrection"
    );

    // Verify item is pending
    let row = sqlx::query("SELECT status FROM unified_queue WHERE queue_id=?1")
        .bind(&id)
        .fetch_one(manager.pool())
        .await
        .unwrap();
    let status: String = row.try_get("status").unwrap();
    assert_eq!(status, "pending");
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
        .dequeue_unified(10, "worker-1", Some(1), None, None, None, None, None)
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
