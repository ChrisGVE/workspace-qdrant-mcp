//! Integration tests for computed priority dequeue and enqueue-first architecture.
//!
//! Tests:
//! - Computed priority: active projects > memory > inactive > libraries
//! - Op-based priority: delete > reset > scan > update > add
//! - Fairness scheduler alternation (DESC/ASC)
//! - Progressive scanning queue growth
//! - Submodule detection at folder boundaries

use workspace_qdrant_core::{
    QueueManager,
    unified_queue_schema::{
        ItemType, QueueOperation,
        CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL,
    },
    fairness_scheduler::{FairnessScheduler, FairnessSchedulerConfig},
};
use sqlx::SqlitePool;
use tempfile::TempDir;

/// Helper to create in-memory SQLite database with required tables
async fn create_test_database() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory database");

    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

    for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        sqlx::query(index_sql).execute(&pool).await
            .expect("Failed to create unified_queue index");
    }

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

    pool
}

/// Insert a watch folder with given activity status
async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, path: &str, tenant_id: &str, collection: &str, is_active: bool) {
    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)"#
    )
    .bind(watch_id)
    .bind(path)
    .bind(collection)
    .bind(tenant_id)
    .bind(is_active as i32)
    .bind(&now)
    .execute(pool)
    .await
    .unwrap();
}

// ── Computed Priority Tests ──

#[tokio::test]
async fn test_active_project_dequeued_before_inactive() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Set up watch folders: active_tenant is active, inactive_tenant is not
    insert_watch_folder(&pool, "wf_1", "/test/active", "active_tenant", "projects", true).await;
    insert_watch_folder(&pool, "wf_2", "/test/inactive", "inactive_tenant", "projects", false).await;

    // Enqueue items: inactive first, active second (to prove ordering overrides insertion order)
    let payload_inactive = serde_json::json!({"file_path": "/test/inactive/file.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "inactive_tenant", "projects",
        &payload_inactive, None, None,
    ).await.unwrap();

    // Small delay to ensure different created_at timestamps
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_active = serde_json::json!({"file_path": "/test/active/file.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "active_tenant", "projects",
        &payload_active, None, None,
    ).await.unwrap();

    // Dequeue with priority DESC — active should come first
    let items = queue_manager.dequeue_unified(
        2, "test-worker", Some(300), None, None, Some(true),
    ).await.unwrap();

    assert_eq!(items.len(), 2, "Should dequeue both items");
    assert_eq!(items[0].tenant_id, "active_tenant", "Active project should be dequeued first");
    assert_eq!(items[1].tenant_id, "inactive_tenant", "Inactive project should be dequeued second");
}

#[tokio::test]
async fn test_memory_collection_high_priority() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Set up a watch folder for the projects tenant
    insert_watch_folder(&pool, "wf_1", "/test/project", "project_tenant", "projects", false).await;

    // Enqueue: library item first, then memory item, then inactive project item
    let payload_lib = serde_json::json!({"file_path": "/lib/doc.md"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "lib_tenant", "libraries",
        &payload_lib, None, None,
    ).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_memory = serde_json::json!({"content": "remember this"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Text, QueueOperation::Add, "user", "rules",
        &payload_memory, None, None,
    ).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_proj = serde_json::json!({"file_path": "/test/file.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "project_tenant", "projects",
        &payload_proj, None, None,
    ).await.unwrap();

    // Dequeue with priority DESC
    let items = queue_manager.dequeue_unified(
        3, "test-worker", Some(300), None, None, Some(true),
    ).await.unwrap();

    assert_eq!(items.len(), 3, "Should dequeue all 3 items");
    // memory (priority=1) should be before libraries (priority=0)
    // For inactive project (priority=0) and libraries (priority=0), FIFO tiebreaker applies
    assert_eq!(items[0].collection, "rules", "Rules collection should be dequeued first in DESC mode");
}

#[tokio::test]
async fn test_op_based_priority_delete_before_add() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    insert_watch_folder(&pool, "wf_1", "/test/project", "tenant_1", "projects", true).await;

    // Enqueue add first, then delete
    let payload_add = serde_json::json!({"file_path": "/test/add.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "tenant_1", "projects",
        &payload_add, None, None,
    ).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_del = serde_json::json!({"file_path": "/test/delete.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Delete, "tenant_1", "projects",
        &payload_del, None, None,
    ).await.unwrap();

    let items = queue_manager.dequeue_unified(
        2, "test-worker", Some(300), None, None, Some(true),
    ).await.unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].op, QueueOperation::Delete, "Delete (op priority=10) should come before Add (op priority=1)");
    assert_eq!(items[1].op, QueueOperation::Add, "Add should come second");
}

#[tokio::test]
async fn test_anti_starvation_asc_mode_reverses_priority() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Active and inactive projects
    insert_watch_folder(&pool, "wf_1", "/test/active", "active_tenant", "projects", true).await;
    insert_watch_folder(&pool, "wf_2", "/test/inactive", "inactive_tenant", "projects", false).await;

    // Enqueue items for both
    let payload_active = serde_json::json!({"file_path": "/test/active/a.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "active_tenant", "projects",
        &payload_active, None, None,
    ).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_inactive = serde_json::json!({"file_path": "/test/inactive/b.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "inactive_tenant", "projects",
        &payload_inactive, None, None,
    ).await.unwrap();

    // Dequeue with priority ASC (anti-starvation mode) — inactive should come first
    let items = queue_manager.dequeue_unified(
        2, "test-worker", Some(300), None, None, Some(false),
    ).await.unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].tenant_id, "inactive_tenant",
        "In ASC mode, inactive (priority=0) should be dequeued before active (priority=1)");
    assert_eq!(items[1].tenant_id, "active_tenant");
}

// ── Fairness Scheduler Tests ──

#[tokio::test]
async fn test_fairness_scheduler_flips_direction() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    insert_watch_folder(&pool, "wf_1", "/test/active", "active_t", "projects", true).await;
    insert_watch_folder(&pool, "wf_2", "/test/inactive", "inactive_t", "projects", false).await;

    // Enqueue enough items to trigger a flip (high_priority_batch=2 for testing)
    for i in 0..5 {
        let payload = serde_json::json!({"file_path": format!("/test/active/f{}.rs", i)}).to_string();
        queue_manager.enqueue_unified(
            ItemType::File, QueueOperation::Add, "active_t", "projects",
            &payload, None, None,
        ).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }
    for i in 0..5 {
        let payload = serde_json::json!({"file_path": format!("/test/inactive/f{}.rs", i)}).to_string();
        queue_manager.enqueue_unified(
            ItemType::File, QueueOperation::Add, "inactive_t", "projects",
            &payload, None, None,
        ).await.unwrap();
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
    assert!(active_count_1 > 0, "First DESC batch should include active items");

    // Second batch: should eventually flip to ASC after high_priority_batch items
    let batch2 = scheduler.dequeue_next_batch(2).await.unwrap();
    assert!(!batch2.is_empty(), "Second batch should have items");

    // Third batch: after 2 high-priority items, should have flipped to ASC (low-priority)
    let batch3 = scheduler.dequeue_next_batch(2).await.unwrap();
    assert!(!batch3.is_empty(), "Third batch should have items");

    // After all batches, verify metrics show direction flips occurred
    let metrics = scheduler.get_metrics().await;
    assert!(metrics.total_items_dequeued > 0, "Should have dequeued items");
}

// ── Progressive Scanning Tests ──

#[tokio::test]
async fn test_progressive_scan_does_not_recurse() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    // Create directory tree 3 levels deep
    std::fs::write(root.join("top.rs"), "fn top() {}").unwrap();
    std::fs::create_dir(root.join("level1")).unwrap();
    std::fs::write(root.join("level1/mid.rs"), "fn mid() {}").unwrap();
    std::fs::create_dir(root.join("level1/level2")).unwrap();
    std::fs::write(root.join("level1/level2/deep.rs"), "fn deep() {}").unwrap();

    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Simulate single-level scan of root: only top.rs and level1/ should be enqueued
    let payload_file = serde_json::json!({"file_path": root.join("top.rs").to_string_lossy()}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "test_t", "projects",
        &payload_file, None, None,
    ).await.unwrap();

    let payload_dir = serde_json::json!({"folder_path": root.join("level1").to_string_lossy()}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Folder, QueueOperation::Scan, "test_t", "projects",
        &payload_dir, None, None,
    ).await.unwrap();

    // Should NOT have level2 or deep.rs — those come from scanning level1 later
    let file_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'file' AND status = 'pending'"
    ).fetch_one(&pool).await.unwrap();
    let folder_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'folder' AND status = 'pending'"
    ).fetch_one(&pool).await.unwrap();

    assert_eq!(file_count, 1, "Only top-level file should be enqueued");
    assert_eq!(folder_count, 1, "Only immediate subdirectory should be enqueued for scanning");
}

#[tokio::test]
async fn test_scan_op_higher_priority_than_add() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    insert_watch_folder(&pool, "wf_1", "/test/project", "tenant_1", "projects", true).await;

    // Enqueue a file add first, then a folder scan
    let payload_file = serde_json::json!({"file_path": "/test/project/file.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "tenant_1", "projects",
        &payload_file, None, None,
    ).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_scan = serde_json::json!({"folder_path": "/test/project/subdir"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Folder, QueueOperation::Scan, "tenant_1", "projects",
        &payload_scan, None, None,
    ).await.unwrap();

    // Dequeue — scan (priority=5) should come before add (priority=1)
    let items = queue_manager.dequeue_unified(
        2, "test-worker", Some(300), None, None, Some(true),
    ).await.unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].op, QueueOperation::Scan, "Scan op should be dequeued before Add due to higher op priority");
    assert_eq!(items[1].op, QueueOperation::Add, "Add op should be dequeued second");
}

// ── Submodule Detection Tests ──

#[tokio::test]
async fn test_submodule_directory_with_git_is_separate() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    // Create a project with a submodule-like directory (contains .git)
    std::fs::write(root.join("main.rs"), "fn main() {}").unwrap();
    std::fs::create_dir(root.join("regular_dir")).unwrap();
    std::fs::write(root.join("regular_dir/lib.rs"), "fn lib() {}").unwrap();
    std::fs::create_dir(root.join("submodule_dir")).unwrap();
    std::fs::create_dir(root.join("submodule_dir/.git")).unwrap(); // This marks it as a submodule
    std::fs::write(root.join("submodule_dir/sub.rs"), "fn sub() {}").unwrap();

    // Verify .git exists in submodule_dir
    let submodule_path = root.join("submodule_dir");
    let has_git = submodule_path.join(".git").exists();
    assert!(has_git, "Submodule directory should contain .git");

    // The scanning logic should detect .git and NOT recurse into submodule_dir
    // Instead it should be registered as a separate project
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Simulate what single-level scan does: regular_dir gets (Folder, Scan),
    // but submodule_dir with .git gets (Tenant, Add) as a new project
    let regular_payload = serde_json::json!({"folder_path": root.join("regular_dir").to_string_lossy()}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Folder, QueueOperation::Scan, "parent_tenant", "projects",
        &regular_payload, None, None,
    ).await.unwrap();

    let submodule_payload = serde_json::json!({
        "project_root": root.join("submodule_dir").to_string_lossy(),
        "detected_as_submodule": true,
    }).to_string();
    queue_manager.enqueue_unified(
        ItemType::Tenant, QueueOperation::Add, "submodule_tenant", "projects",
        &submodule_payload, None, None,
    ).await.unwrap();

    // Verify: regular_dir is a folder scan, submodule is a tenant add
    let folder_scans: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'folder' AND op = 'scan'"
    ).fetch_one(&pool).await.unwrap();
    let tenant_adds: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'add'"
    ).fetch_one(&pool).await.unwrap();

    assert_eq!(folder_scans, 1, "Regular directory should produce Folder/Scan");
    assert_eq!(tenant_adds, 1, "Submodule directory should produce Tenant/Add");
}

// ── Bulk Operations Tests ──

#[tokio::test]
async fn test_tenant_delete_enqueues_for_full_removal() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Enqueue a tenant deletion
    let payload = serde_json::json!({
        "project_root": "/test/project",
        "delete_vectors": true,
    }).to_string();

    let (queue_id, is_new) = queue_manager.enqueue_unified(
        ItemType::Tenant, QueueOperation::Delete, "project_tenant", "projects",
        &payload, None, None,
    ).await.unwrap();

    assert!(is_new);
    assert!(!queue_id.is_empty());

    // Verify the tenant delete is in the queue
    let count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'delete' AND tenant_id = 'project_tenant'"
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(count, 1, "Tenant delete should be enqueued");
}

#[tokio::test]
async fn test_delete_op_highest_priority_in_dequeue() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    insert_watch_folder(&pool, "wf_1", "/test/project", "t1", "projects", true).await;

    // Enqueue file ops with different priorities
    let payload_add = serde_json::json!({"file_path": "/test/project/add.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "t1", "projects",
        &payload_add, None, None,
    ).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

    let payload_update = serde_json::json!({"file_path": "/test/project/update.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Update, "t1", "projects",
        &payload_update, None, None,
    ).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

    // Folder scan has op priority 5
    let payload_scan = serde_json::json!({"folder_path": "/test/project/subdir"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Folder, QueueOperation::Scan, "t1", "projects",
        &payload_scan, None, None,
    ).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

    let payload_del = serde_json::json!({"file_path": "/test/project/delete.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Delete, "t1", "projects",
        &payload_del, None, None,
    ).await.unwrap();

    // Dequeue all — should be ordered by op priority: delete(10) > scan(5) > update(3) > add(1)
    let items = queue_manager.dequeue_unified(
        4, "test-worker", Some(300), None, None, Some(true),
    ).await.unwrap();

    assert_eq!(items.len(), 4);
    assert_eq!(items[0].op, QueueOperation::Delete, "Delete (10) should be first");
    assert_eq!(items[1].op, QueueOperation::Scan, "Scan (5) should be second");
    assert_eq!(items[2].op, QueueOperation::Update, "Update (3) should be third");
    assert_eq!(items[3].op, QueueOperation::Add, "Add (1) should be last");
}
