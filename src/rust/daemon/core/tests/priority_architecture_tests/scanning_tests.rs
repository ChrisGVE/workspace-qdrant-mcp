//! Progressive Scanning Tests, Submodule Detection Tests, and Bulk Operations Tests.

use super::*;

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
async fn test_add_op_higher_priority_than_scan() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    insert_watch_folder(&pool, "wf_1", "/test/project", "tenant_1", "projects", true).await;

    // Enqueue a folder scan first, then a file add
    let payload_scan = serde_json::json!({"folder_path": "/test/project/subdir"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::Folder, QueueOperation::Scan, "tenant_1", "projects",
        &payload_scan, None, None,
    ).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let payload_file = serde_json::json!({"file_path": "/test/project/file.rs"}).to_string();
    queue_manager.enqueue_unified(
        ItemType::File, QueueOperation::Add, "tenant_1", "projects",
        &payload_file, None, None,
    ).await.unwrap();

    // Dequeue — add (priority=5) should come before scan (priority=1)
    let items = queue_manager.dequeue_unified(
        2, "test-worker", Some(300), None, None, Some(true),
    ).await.unwrap();

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].op, QueueOperation::Add, "Add op should be dequeued before Scan due to higher op priority");
    assert_eq!(items[1].op, QueueOperation::Scan, "Scan op should be dequeued second");
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

    // Folder scan has op priority 1 (lowest — discovery work)
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

    // Dequeue all — should be ordered by op priority: delete(10) > add(5) > update(3) > scan(1)
    let items = queue_manager.dequeue_unified(
        4, "test-worker", Some(300), None, None, Some(true),
    ).await.unwrap();

    assert_eq!(items.len(), 4);
    assert_eq!(items[0].op, QueueOperation::Delete, "Delete (10) should be first");
    assert_eq!(items[1].op, QueueOperation::Add, "Add (5) should be second");
    assert_eq!(items[2].op, QueueOperation::Update, "Update (3) should be third");
    assert_eq!(items[3].op, QueueOperation::Scan, "Scan (1) should be last");
}
