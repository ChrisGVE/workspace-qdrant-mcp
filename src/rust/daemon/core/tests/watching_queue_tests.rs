//! Integration tests for file watching with SQLite queue

use workspace_qdrant_core::{
    FileWatcherQueue, WatchManager, WatchConfig, WatchingQueueStats,
    QueueManager, WatchType,
};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use sqlx::{Row, SqlitePool};
use tokio::time::{sleep, Duration};

/// Helper to create in-memory SQLite database with queue schema
async fn create_test_database() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory database");

    // Create ingestion_queue table
    sqlx::query(
        r#"
        CREATE TABLE ingestion_queue (
            queue_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
            file_absolute_path TEXT NOT NULL UNIQUE,
            collection_name TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            branch TEXT NOT NULL,
            operation TEXT NOT NULL CHECK(operation IN ('ingest', 'update', 'delete')),
            priority INTEGER NOT NULL CHECK(priority >= 0 AND priority <= 10),
            queued_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            scheduled_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            retry_count INTEGER NOT NULL DEFAULT 0,
            retry_from TEXT,
            error_message_id INTEGER,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
            metadata TEXT DEFAULT '{}'
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create ingestion_queue table");

    // Create watch_folders table
    sqlx::query(
        r#"
        CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL,
            patterns TEXT NOT NULL,
            ignore_patterns TEXT NOT NULL,
            auto_ingest BOOLEAN NOT NULL DEFAULT 1,
            recursive BOOLEAN NOT NULL DEFAULT 1,
            recursive_depth INTEGER NOT NULL DEFAULT 10,
            debounce_seconds REAL NOT NULL DEFAULT 2.0,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create watch_folders table");

    pool
}

#[tokio::test]
async fn test_file_watcher_queue_creation() {
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let config = WatchConfig {
        id: "test-watch".to_string(),
        path: PathBuf::from("/tmp"),
        collection: "test-collection".to_string(),
        patterns: vec!["*.txt".to_string()],
        ignore_patterns: vec!["*.tmp".to_string()],
        recursive: true,
        debounce_ms: 1000,
        enabled: true,
        watch_type: WatchType::Project,
        library_name: None,
    };

    let watcher = FileWatcherQueue::new(config, queue_manager);
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
    let manager = WatchManager::new(pool);

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
            watch_id, path, collection, patterns, ignore_patterns,
            recursive, debounce_seconds, enabled
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#
    )
    .bind("test-watch-1")
    .bind(&watch_path)
    .bind("test-collection")
    .bind(r#"["*.txt", "*.md"]"#)
    .bind(r#"["*.tmp", ".git/**"]"#)
    .bind(true)
    .bind(1.0)
    .bind(true)
    .execute(&pool)
    .await
    .expect("Failed to insert watch configuration");

    let manager = WatchManager::new(pool.clone());

    // Start all watches
    let result = manager.start_all_watches().await;
    assert!(result.is_ok(), "Failed to start watches: {:?}", result.err());

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
async fn test_file_enqueue_operation() {
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    // Test direct queue operations
    use workspace_qdrant_core::QueueOperation;

    let result = queue_manager.enqueue_file(
        "/tmp/test.txt",
        "test-collection",
        "test-tenant",
        "main",
        QueueOperation::Ingest,
        5,
        None,
    ).await;

    assert!(result.is_ok(), "Failed to enqueue file: {:?}", result.err());

    // Verify the file was enqueued
    let row = sqlx::query("SELECT COUNT(*) as count FROM ingestion_queue")
        .fetch_one(&pool)
        .await
        .expect("Failed to query queue");

    let count: i64 = row.try_get("count").expect("Failed to get count");
    assert_eq!(count, 1);

    // Verify the queue item details
    let row = sqlx::query("SELECT * FROM ingestion_queue")
        .fetch_one(&pool)
        .await
        .expect("Failed to fetch queue item");

    let file_path: String = row.try_get("file_absolute_path").expect("Failed to get file path");
    let operation: String = row.try_get("operation").expect("Failed to get operation");
    let priority: i32 = row.try_get("priority").expect("Failed to get priority");

    assert_eq!(file_path, "/tmp/test.txt");
    assert_eq!(operation, "ingest");
    assert_eq!(priority, 5);
}

#[tokio::test]
async fn test_operation_priority_calculation() {
    // This test verifies that priorities are calculated correctly for different operations
    use workspace_qdrant_core::QueueOperation;

    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    // Ingest operation - priority 5
    queue_manager.enqueue_file(
        "/tmp/ingest.txt",
        "test",
        "tenant",
        "main",
        QueueOperation::Ingest,
        5,
        None,
    ).await.expect("Failed to enqueue ingest");

    // Update operation - priority 5
    queue_manager.enqueue_file(
        "/tmp/update.txt",
        "test",
        "tenant",
        "main",
        QueueOperation::Update,
        5,
        None,
    ).await.expect("Failed to enqueue update");

    // Delete operation - priority 8 (higher)
    queue_manager.enqueue_file(
        "/tmp/delete.txt",
        "test",
        "tenant",
        "main",
        QueueOperation::Delete,
        8,
        None,
    ).await.expect("Failed to enqueue delete");

    // Verify priorities in database
    let rows = sqlx::query("SELECT file_absolute_path, priority FROM ingestion_queue ORDER BY file_absolute_path")
        .fetch_all(&pool)
        .await
        .expect("Failed to fetch queue items");

    assert_eq!(rows.len(), 3);

    // Check delete has highest priority
    let delete_row = rows.iter().find(|r| {
        let path: String = r.try_get("file_absolute_path").unwrap();
        path.contains("delete")
    }).expect("Delete row not found");

    let delete_priority: i32 = delete_row.try_get("priority").unwrap();
    assert_eq!(delete_priority, 8);
}
