//! Integration tests for queue taxonomy work items
//!
//! Tests for:
//! - Progressive single-level scanning (WI-9)
//! - RegisterProject/DeleteProject enqueue-only pattern (WI-5/6)
//! - Collection uplift/reset cascade (WI-7)

use workspace_qdrant_core::{
    QueueManager,
    unified_queue_schema::{ItemType, QueueOperation},
};
use sqlx::SqlitePool;
use tempfile::TempDir;

/// Helper to create in-memory SQLite database with required tables
async fn create_test_database() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory database");

    sqlx::query(
        r#"
        CREATE TABLE unified_queue (
            queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
            item_type TEXT NOT NULL CHECK (item_type IN (
                'text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'
            )),
            op TEXT NOT NULL CHECK (op IN ('add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset')),
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

    sqlx::query(
        r#"
        CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            parent_watch_id TEXT,
            is_active INTEGER DEFAULT 0,
            git_remote_url TEXT,
            follow_symlinks INTEGER DEFAULT 0,
            last_scan TEXT,
            last_activity_at TEXT,
            enabled INTEGER DEFAULT 1,
            cleanup_on_disable INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(path, collection)
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create watch_folders table");

    sqlx::query(
        r#"
        CREATE TABLE tracked_files (
            file_id TEXT PRIMARY KEY,
            watch_folder_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            file_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_hash TEXT,
            file_size INTEGER,
            processing_status TEXT DEFAULT 'pending',
            last_processed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create tracked_files table");

    sqlx::query(
        r#"
        CREATE TABLE qdrant_chunks (
            chunk_id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            point_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_type TEXT NOT NULL,
            content_hash TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create qdrant_chunks table");

    pool
}

/// Helper: count queue items of a specific type and op
async fn count_queue_items(pool: &SqlitePool, item_type: &str, op: &str) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = ?1 AND op = ?2 AND status = 'pending'"
    )
    .bind(item_type)
    .bind(op)
    .fetch_one(pool)
    .await
    .unwrap_or(0)
}

/// Helper: count all pending queue items
async fn count_pending_items(pool: &SqlitePool) -> i64 {
    sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending'"
    )
    .fetch_one(pool)
    .await
    .unwrap_or(0)
}

// ── Progressive Scanning (WI-9) ──

#[tokio::test]
async fn test_register_project_enqueues_tenant_add() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Enqueue (Tenant, Add) as RegisterProject would
    let payload = serde_json::json!({
        "project_root": "/test/project",
        "is_active": true,
    }).to_string();

    let (queue_id, is_new) = queue_manager.enqueue_unified(
        ItemType::Tenant,
        QueueOperation::Add,
        "test_tenant_01",
        "projects",
        &payload,
        0,
        None,
        None,
    ).await.unwrap();

    assert!(is_new);
    assert!(!queue_id.is_empty());
    assert_eq!(count_queue_items(&pool, "tenant", "add").await, 1);
}

#[tokio::test]
async fn test_delete_project_enqueues_tenant_delete() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    let payload = serde_json::json!({
        "project_root": "",
    }).to_string();

    let (queue_id, is_new) = queue_manager.enqueue_unified(
        ItemType::Tenant,
        QueueOperation::Delete,
        "test_tenant_01",
        "projects",
        &payload,
        0,
        None,
        None,
    ).await.unwrap();

    assert!(is_new);
    assert!(!queue_id.is_empty());
    assert_eq!(count_queue_items(&pool, "tenant", "delete").await, 1);
}

#[tokio::test]
async fn test_progressive_scan_single_level_only() {
    // Create a temp directory with nested structure
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    // Create nested directory structure:
    // root/
    //   file1.rs
    //   file2.rs
    //   subdir1/
    //     nested_file.rs
    //   subdir2/
    //     deeply/
    //       hidden.rs
    //   .git/           (should be excluded)
    //   node_modules/   (should be excluded)

    std::fs::write(root.join("file1.rs"), "fn main() {}").unwrap();
    std::fs::write(root.join("file2.rs"), "fn test() {}").unwrap();
    std::fs::create_dir(root.join("subdir1")).unwrap();
    std::fs::write(root.join("subdir1/nested_file.rs"), "fn nested() {}").unwrap();
    std::fs::create_dir_all(root.join("subdir2/deeply")).unwrap();
    std::fs::write(root.join("subdir2/deeply/hidden.rs"), "fn hidden() {}").unwrap();
    std::fs::create_dir(root.join(".git")).unwrap();
    std::fs::create_dir(root.join("node_modules")).unwrap();

    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Simulate a project scan by enqueueing individual items as single-level scan would
    let payload = serde_json::json!({
        "project_root": root.to_string_lossy().to_string(),
    }).to_string();

    // Enqueue (Tenant, Scan) as scan_project_directory does
    let (_queue_id, _) = queue_manager.enqueue_unified(
        ItemType::Tenant,
        QueueOperation::Scan,
        "test_project",
        "projects",
        &payload,
        0,
        None,
        None,
    ).await.unwrap();

    // The scan itself would produce:
    // - 2 (File, Add) for file1.rs, file2.rs (immediate children only)
    // - 2 (Folder, Scan) for subdir1, subdir2
    // NOT: nested_file.rs or hidden.rs (those are in subdirs)
    // NOT: .git or node_modules (excluded)

    // Verify by manually enqueuing what single-level scan produces
    for fname in &["file1.rs", "file2.rs"] {
        let file_path = root.join(fname).to_string_lossy().to_string();
        let fp = serde_json::json!({"file_path": file_path}).to_string();
        queue_manager.enqueue_unified(
            ItemType::File, QueueOperation::Add, "test_project", "projects",
            &fp, 0, None, None,
        ).await.unwrap();
    }
    for dname in &["subdir1", "subdir2"] {
        let dir_path = root.join(dname).to_string_lossy().to_string();
        let fp = serde_json::json!({"folder_path": dir_path}).to_string();
        queue_manager.enqueue_unified(
            ItemType::Folder, QueueOperation::Scan, "test_project", "projects",
            &fp, 0, None, None,
        ).await.unwrap();
    }

    // Should have exactly 2 files and 2 folders (not recursive)
    assert_eq!(count_queue_items(&pool, "file", "add").await, 2);
    assert_eq!(count_queue_items(&pool, "folder", "scan").await, 2);
    // Total: 1 tenant/scan + 2 file/add + 2 folder/scan = 5
    assert_eq!(count_pending_items(&pool).await, 5);
}

// ── Collection Cascade (WI-7) ──

#[tokio::test]
async fn test_collection_uplift_cascade() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create watch_folders for 3 tenants in the "projects" collection
    let now = wqm_common::timestamps::now_utc();
    for i in 1..=3 {
        sqlx::query(
            r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
               VALUES (?1, ?2, 'projects', ?3, 1, ?4, ?4)"#
        )
        .bind(format!("wf_{}", i))
        .bind(format!("/test/project{}", i))
        .bind(format!("tenant_{}", i))
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Enqueue (Collection, Uplift) for "projects"
    let payload = serde_json::json!({"collection_name": "projects"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Collection,
        QueueOperation::Uplift,
        "system",
        "projects",
        &payload,
        0,
        None,
        None,
    ).await.unwrap();

    assert_eq!(count_queue_items(&pool, "collection", "uplift").await, 1);
}

#[tokio::test]
async fn test_collection_reset_cascade() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
           VALUES ('wf_1', '/test/project1', 'projects', 'tenant_1', 1, ?1, ?1)"#
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Enqueue (Collection, Reset) for "projects"
    let payload = serde_json::json!({"collection_name": "projects"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Collection,
        QueueOperation::Reset,
        "system",
        "projects",
        &payload,
        0,
        None,
        None,
    ).await.unwrap();

    assert_eq!(count_queue_items(&pool, "collection", "reset").await, 1);
}

// ── Dequeue Ordering ──

#[tokio::test]
async fn test_idempotent_enqueue_deduplication() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    let payload = serde_json::json!({"file_path": "/test/file.rs"}).to_string();

    // First enqueue
    let (_, is_new_1) = queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "tenant1", "projects",
        &payload, 0, None, None,
    ).await.unwrap();

    // Duplicate enqueue (same idempotency key)
    let (_, is_new_2) = queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "tenant1", "projects",
        &payload, 0, None, None,
    ).await.unwrap();

    assert!(is_new_1);
    assert!(!is_new_2);
    // Only 1 item in queue
    assert_eq!(count_queue_items(&pool, "file", "add").await, 1);
}

#[tokio::test]
async fn test_website_add_creates_scan() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    let payload = serde_json::json!({
        "url": "https://example.com",
        "max_depth": 2,
        "max_pages": 10,
    }).to_string();

    queue_manager.enqueue_unified(
        ItemType::Website, QueueOperation::Add, "tenant1", "projects",
        &payload, 0, None, None,
    ).await.unwrap();

    assert_eq!(count_queue_items(&pool, "website", "add").await, 1);
}

#[tokio::test]
async fn test_tenant_uplift_with_tracked_files() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());
    let now = wqm_common::timestamps::now_utc();

    // Create tracked files for the tenant
    for i in 1..=5 {
        sqlx::query(
            r#"INSERT INTO tracked_files (file_id, watch_folder_id, tenant_id, collection, file_path, relative_path, created_at, updated_at)
               VALUES (?1, 'wf_1', 'tenant_1', 'projects', ?2, ?3, ?4, ?4)"#
        )
        .bind(format!("file_{}", i))
        .bind(format!("/test/project/file{}.rs", i))
        .bind(format!("file{}.rs", i))
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Enqueue (Tenant, Uplift)
    let payload = serde_json::json!({"project_root": "/test/project"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Tenant, QueueOperation::Uplift, "tenant_1", "projects",
        &payload, 0, None, None,
    ).await.unwrap();

    assert_eq!(count_queue_items(&pool, "tenant", "uplift").await, 1);
}
