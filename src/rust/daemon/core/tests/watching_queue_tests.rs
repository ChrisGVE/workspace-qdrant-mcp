//! Integration tests for file watching with SQLite queue
//!
//! Updated per Task 21 to use unified_queue instead of legacy ingestion_queue.

use sqlx::{Row, SqlitePool};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};
use workspace_qdrant_core::{
    unified_queue_schema::{
        FilePayload, ItemType, QueueOperation as UnifiedOp, CREATE_UNIFIED_QUEUE_INDEXES_SQL,
        CREATE_UNIFIED_QUEUE_SQL,
    },
    watch_folders_schema::{CREATE_WATCH_FOLDERS_INDEXES_SQL, CREATE_WATCH_FOLDERS_SQL},
    AllowedExtensions, FileWatcherQueue, QueueManager, WatchConfig, WatchManager, WatchType,
};

/// Helper to create an in-memory SQLite database using the canonical
/// production DDL constants, so this test schema cannot drift from the
/// schema the daemon actually creates.
async fn create_test_database() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory database");

    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");
    for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        sqlx::query(index_sql)
            .execute(&pool)
            .await
            .expect("Failed to create unified_queue index");
    }

    sqlx::query(CREATE_WATCH_FOLDERS_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create watch_folders table");
    for index_sql in CREATE_WATCH_FOLDERS_INDEXES_SQL {
        sqlx::query(index_sql)
            .execute(&pool)
            .await
            .expect("Failed to create watch_folders index");
    }

    pool
}

#[tokio::test]
async fn test_file_watcher_queue_creation() {
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let config = WatchConfig {
        id: "test-watch".to_string(),
        path: PathBuf::from("/tmp"),
        tenant_id: "test-tenant".to_string(),
        collection: "test-collection".to_string(),
        patterns: vec!["*.txt".to_string()],
        ignore_patterns: vec!["*.tmp".to_string()],
        recursive: true,
        debounce_ms: 1000,
        enabled: true,
        watch_type: WatchType::Project,
        library_name: None,
    };

    let allowed_extensions = Arc::new(AllowedExtensions::default());
    let watcher = FileWatcherQueue::new(config, queue_manager, allowed_extensions);
    assert!(watcher.is_ok());

    let watcher = watcher.unwrap();
    let stats = watcher.get_stats().await;

    assert_eq!(stats.events_received, 0);
    assert_eq!(stats.events_processed, 0);
    assert_eq!(stats.events_filtered, 0);
    assert_eq!(stats.queue_errors, 0);
}

#[tokio::test]
async fn test_watch_manager_creation() {
    let pool = create_test_database().await;
    let allowed_extensions = Arc::new(AllowedExtensions::default());
    let manager = WatchManager::new(pool, allowed_extensions);

    // Test that manager can be created and stats retrieved
    let stats = manager.get_all_stats().await;
    assert_eq!(stats.len(), 0);
}

#[tokio::test]
async fn test_watch_manager_with_configuration() {
    let pool = create_test_database().await;

    // Create a temporary directory for testing
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let watch_path = temp_dir.path().to_string_lossy().to_string();

    // Insert a watch configuration
    sqlx::query(
        r#"
        INSERT INTO watch_folders (
            watch_id, path, collection, tenant_id, enabled, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5,
                  strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                  strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        "#,
    )
    .bind("test-watch-1")
    .bind(&watch_path)
    .bind("projects")
    .bind("test-tenant")
    .bind(true)
    .execute(&pool)
    .await
    .expect("Failed to insert watch configuration");

    let allowed_extensions = Arc::new(AllowedExtensions::default());
    let manager = WatchManager::new(pool.clone(), allowed_extensions);

    // Start all watches
    let result = manager.start_all_watches().await;
    assert!(
        result.is_ok(),
        "Failed to start watches: {:?}",
        result.err()
    );

    // Give the watcher a moment to start
    sleep(Duration::from_millis(100)).await;

    // Check stats
    let stats = manager.get_all_stats().await;
    assert_eq!(stats.len(), 1);
    assert!(stats.contains_key("test-watch-1"));

    // Stop all watches
    let result = manager.stop_all_watches().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_unified_queue_enqueue_operation() {
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    // Create file payload
    let payload = FilePayload {
        file_path: wqm_common::paths::RelativePath::from_user_input(
            "/tmp/test.txt".trim_start_matches('/'),
        )
        .unwrap(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: Some(100),
        old_path: None,
    };
    let payload_json = serde_json::to_string(&payload).unwrap();

    // Test unified queue operations
    let result = queue_manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "projects",
            &payload_json,
            Some("main"),
            None,
        )
        .await;

    assert!(result.is_ok(), "Failed to enqueue file: {:?}", result.err());
    let (queue_id, is_new) = result.unwrap();
    assert!(is_new, "Expected new item to be created");
    assert!(!queue_id.is_empty(), "Queue ID should not be empty");

    // Verify the file was enqueued
    let row = sqlx::query("SELECT COUNT(*) as count FROM unified_queue")
        .fetch_one(&pool)
        .await
        .expect("Failed to query queue");

    let count: i64 = row.try_get("count").expect("Failed to get count");
    assert_eq!(count, 1);

    // Verify the queue item details
    let row = sqlx::query("SELECT * FROM unified_queue")
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch queue item");

    let item_type: String = row.try_get("item_type").expect("Failed to get item_type");
    let op: String = row.try_get("op").expect("Failed to get op");
    let tenant_id: String = row.try_get("tenant_id").expect("Failed to get tenant_id");

    assert_eq!(item_type, "file");
    assert_eq!(op, "add");
    // Priority column removed — computed dynamically at dequeue time via CASE/JOIN
    assert_eq!(tenant_id, "test-tenant");
}

#[tokio::test]
async fn test_unified_queue_multiple_operations() {
    // Priority is computed dynamically at dequeue time via CASE/JOIN — no priority column stored.
    // Verify that multiple operations can be enqueued with correct op types.
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let payload1 = FilePayload {
        file_path: wqm_common::paths::RelativePath::from_user_input(
            "/tmp/ingest.txt".trim_start_matches('/'),
        )
        .unwrap(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };
    queue_manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant",
            "projects",
            &serde_json::to_string(&payload1).unwrap(),
            Some("main"),
            None,
        )
        .await
        .expect("Failed to enqueue ingest");

    let payload2 = FilePayload {
        file_path: wqm_common::paths::RelativePath::from_user_input(
            "/tmp/update.txt".trim_start_matches('/'),
        )
        .unwrap(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };
    queue_manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Update,
            "tenant",
            "projects",
            &serde_json::to_string(&payload2).unwrap(),
            Some("main"),
            None,
        )
        .await
        .expect("Failed to enqueue update");

    let payload3 = FilePayload {
        file_path: wqm_common::paths::RelativePath::from_user_input(
            "/tmp/delete.txt".trim_start_matches('/'),
        )
        .unwrap(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };
    queue_manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Delete,
            "tenant",
            "projects",
            &serde_json::to_string(&payload3).unwrap(),
            Some("main"),
            None,
        )
        .await
        .expect("Failed to enqueue delete");

    // Verify all three items were enqueued with correct ops
    let rows = sqlx::query("SELECT file_path, op FROM unified_queue ORDER BY file_path")
        .fetch_all(&pool)
        .await
        .expect("Failed to fetch queue items");

    assert_eq!(rows.len(), 3);

    let ops: Vec<String> = rows
        .iter()
        .map(|r| r.try_get::<String, _>("op").unwrap())
        .collect();
    assert!(ops.contains(&"delete".to_string()));
    assert!(ops.contains(&"add".to_string()));
    assert!(ops.contains(&"update".to_string()));
}

#[tokio::test]
async fn test_unified_queue_idempotency() {
    // Test that duplicate enqueues are handled via idempotency
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let payload = FilePayload {
        file_path: wqm_common::paths::RelativePath::from_user_input(
            "/tmp/idempotent.txt".trim_start_matches('/'),
        )
        .unwrap(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };
    let payload_json = serde_json::to_string(&payload).unwrap();

    // First enqueue
    let (queue_id1, is_new1) = queue_manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant",
            "projects",
            &payload_json,
            Some("main"),
            None,
        )
        .await
        .expect("Failed to enqueue first time");

    assert!(is_new1, "First enqueue should be new");

    // Second enqueue with same payload (should be idempotent)
    let (queue_id2, is_new2) = queue_manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "tenant",
            "projects",
            &payload_json,
            Some("main"),
            None,
        )
        .await
        .expect("Failed to enqueue second time");

    assert!(!is_new2, "Second enqueue should not be new (idempotent)");
    assert_eq!(
        queue_id1, queue_id2,
        "Queue IDs should match for idempotent enqueue"
    );

    // Verify only one item in queue
    let row = sqlx::query("SELECT COUNT(*) as count FROM unified_queue")
        .fetch_one(&pool)
        .await
        .expect("Failed to query queue");

    let count: i64 = row.try_get("count").expect("Failed to get count");
    assert_eq!(count, 1, "Should only have one item due to idempotency");
}
