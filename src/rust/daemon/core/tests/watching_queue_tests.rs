//! Integration tests for file watching with SQLite queue
//!
//! Updated per Task 21 to use unified_queue instead of legacy ingestion_queue.

use workspace_qdrant_core::{
    FileWatcherQueue, WatchManager, WatchConfig, WatchingQueueStats,
    QueueManager, WatchType,
    unified_queue_schema::{ItemType, QueueOperation as UnifiedOp, FilePayload},
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

    // Create unified_queue table (spec-compliant)
    sqlx::query(
        r#"
        CREATE TABLE unified_queue (
            queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
            item_type TEXT NOT NULL CHECK (item_type IN (
                'content', 'file', 'folder', 'project', 'library',
                'delete_tenant', 'delete_document', 'rename'
            )),
            op TEXT NOT NULL CHECK (op IN ('ingest', 'update', 'delete', 'scan')),
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 0 AND priority <= 10),
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                'pending', 'in_progress', 'done', 'failed'
            )),
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            lease_until TEXT,
            worker_id TEXT,
            idempotency_key TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL DEFAULT '{}',
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            error_message TEXT,
            last_error_at TEXT,
            branch TEXT DEFAULT 'main',
            metadata TEXT DEFAULT '{}',
            file_path TEXT UNIQUE
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create unified_queue table");

    // Create watch_folders table (includes tenant_id and is_active for priority calculation JOIN)
    sqlx::query(
        r#"
        CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
            tenant_id TEXT NOT NULL,
            parent_watch_id TEXT,
            is_active INTEGER DEFAULT 0 CHECK (is_active IN (0, 1)),
            patterns TEXT NOT NULL,
            ignore_patterns TEXT NOT NULL,
            auto_ingest BOOLEAN NOT NULL DEFAULT 1,
            recursive BOOLEAN NOT NULL DEFAULT 1,
            recursive_depth INTEGER NOT NULL DEFAULT 10,
            debounce_seconds REAL NOT NULL DEFAULT 2.0,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_activity_at TEXT,
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
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
            watch_id, path, collection, tenant_id, patterns, ignore_patterns,
            recursive, debounce_seconds, enabled
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#
    )
    .bind("test-watch-1")
    .bind(&watch_path)
    .bind("projects")
    .bind("test-tenant")
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
async fn test_unified_queue_enqueue_operation() {
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    // Create file payload
    let payload = FilePayload {
        file_path: "/tmp/test.txt".to_string(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: Some(100),
    };
    let payload_json = serde_json::to_string(&payload).unwrap();

    // Test unified queue operations
    let result = queue_manager.enqueue_unified(
        ItemType::File,
        UnifiedOp::Ingest,
        "test-tenant",
        "projects",
        &payload_json,
        0,
        Some("main"),
        None,
    ).await;

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
    let priority: i32 = row.try_get("priority").expect("Failed to get priority");
    let tenant_id: String = row.try_get("tenant_id").expect("Failed to get tenant_id");

    assert_eq!(item_type, "file");
    assert_eq!(op, "ingest");
    assert_eq!(priority, 0);  // Priority is always 0 (computed dynamically at dequeue time)
    assert_eq!(tenant_id, "test-tenant");
}

#[tokio::test]
async fn test_unified_queue_priority_always_zero() {
    // Priority is computed dynamically at dequeue time via CASE/JOIN, not stored.
    // All enqueued items should have priority=0 regardless of operation type.
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let payload1 = FilePayload {
        file_path: "/tmp/ingest.txt".to_string(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
    };
    queue_manager.enqueue_unified(
        ItemType::File,
        UnifiedOp::Ingest,
        "tenant",
        "projects",
        &serde_json::to_string(&payload1).unwrap(),
        0,
        Some("main"),
        None,
    ).await.expect("Failed to enqueue ingest");

    let payload2 = FilePayload {
        file_path: "/tmp/update.txt".to_string(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
    };
    queue_manager.enqueue_unified(
        ItemType::File,
        UnifiedOp::Update,
        "tenant",
        "projects",
        &serde_json::to_string(&payload2).unwrap(),
        0,
        Some("main"),
        None,
    ).await.expect("Failed to enqueue update");

    let payload3 = FilePayload {
        file_path: "/tmp/delete.txt".to_string(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
    };
    queue_manager.enqueue_unified(
        ItemType::File,
        UnifiedOp::Delete,
        "tenant",
        "projects",
        &serde_json::to_string(&payload3).unwrap(),
        0,
        Some("main"),
        None,
    ).await.expect("Failed to enqueue delete");

    // Verify all priorities are 0 in database
    let rows = sqlx::query("SELECT file_path, priority FROM unified_queue ORDER BY file_path")
        .fetch_all(&pool)
        .await
        .expect("Failed to fetch queue items");

    assert_eq!(rows.len(), 3);

    for row in &rows {
        let priority: i32 = row.try_get("priority").unwrap();
        assert_eq!(priority, 0, "All stored priorities should be 0 (dynamic at dequeue time)");
    }
}

#[tokio::test]
async fn test_unified_queue_idempotency() {
    // Test that duplicate enqueues are handled via idempotency
    let pool = create_test_database().await;
    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let payload = FilePayload {
        file_path: "/tmp/idempotent.txt".to_string(),
        file_type: Some("text".to_string()),
        file_hash: None,
        size_bytes: None,
    };
    let payload_json = serde_json::to_string(&payload).unwrap();

    // First enqueue
    let (queue_id1, is_new1) = queue_manager.enqueue_unified(
        ItemType::File,
        UnifiedOp::Ingest,
        "tenant",
        "projects",
        &payload_json,
        0,
        Some("main"),
        None,
    ).await.expect("Failed to enqueue first time");

    assert!(is_new1, "First enqueue should be new");

    // Second enqueue with same payload (should be idempotent)
    let (queue_id2, is_new2) = queue_manager.enqueue_unified(
        ItemType::File,
        UnifiedOp::Ingest,
        "tenant",
        "projects",
        &payload_json,
        0,
        Some("main"),
        None,
    ).await.expect("Failed to enqueue second time");

    assert!(!is_new2, "Second enqueue should not be new (idempotent)");
    assert_eq!(queue_id1, queue_id2, "Queue IDs should match for idempotent enqueue");

    // Verify only one item in queue
    let row = sqlx::query("SELECT COUNT(*) as count FROM unified_queue")
        .fetch_one(&pool)
        .await
        .expect("Failed to query queue");

    let count: i64 = row.try_get("count").expect("Failed to get count");
    assert_eq!(count, 1, "Should only have one item due to idempotency");
}
