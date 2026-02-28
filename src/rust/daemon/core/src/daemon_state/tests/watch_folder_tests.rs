//! Watch folder CRUD, library config, collection filter, enabled toggle,
//! last scan update, and get-by-tenant-id tests.

use super::*;

#[tokio::test]
async fn test_watch_folder_crud() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("watch_folder_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create a watch folder record
    let record = WatchFolderRecord {
        watch_id: "test-watch-001".to_string(),
        path: "/projects/my-project".to_string(),
        collection: "projects".to_string(),
        tenant_id: "my-project-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/user/repo.git".to_string()),
        remote_hash: Some("abc123def456".to_string()),
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
        library_mode: None,
        follow_symlinks: false,
        enabled: true,
        cleanup_on_disable: false,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    };

    // Store
    manager.store_watch_folder(&record).await.unwrap();

    // Retrieve
    let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.path, "/projects/my-project");
    assert_eq!(retrieved.tenant_id, "my-project-tenant");
    assert!(!retrieved.is_active);
    assert!(retrieved.enabled);

    // Activate project
    let updated = manager.activate_project_group("test-watch-001").await.unwrap();
    assert_eq!(updated, 1);

    let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap().unwrap();
    assert!(retrieved.is_active);

    // List active projects
    let active = manager.list_active_projects().await.unwrap();
    assert_eq!(active.len(), 1);

    // Deactivate
    manager.deactivate_project_group("test-watch-001").await.unwrap();
    let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap().unwrap();
    assert!(!retrieved.is_active);

    // Delete
    let deleted = manager.remove_watch_folder("test-watch-001").await.unwrap();
    assert!(deleted);

    let retrieved = manager.get_watch_folder("test-watch-001").await.unwrap();
    assert!(retrieved.is_none());
}

#[tokio::test]
async fn test_watch_folder_with_submodule() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("watch_submodule_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create parent project
    let parent = WatchFolderRecord {
        watch_id: "parent-001".to_string(),
        path: "/projects/parent".to_string(),
        collection: "projects".to_string(),
        tenant_id: "parent-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/user/parent.git".to_string()),
        remote_hash: Some("parent12hash".to_string()),
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
        library_mode: None,
        follow_symlinks: false,
        enabled: true,
        cleanup_on_disable: false,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    };
    manager.store_watch_folder(&parent).await.unwrap();

    // Create submodule
    let submodule = WatchFolderRecord {
        watch_id: "submodule-001".to_string(),
        path: "/projects/parent/libs/sub".to_string(),
        collection: "projects".to_string(),
        tenant_id: "submodule-tenant".to_string(),
        parent_watch_id: Some("parent-001".to_string()),
        submodule_path: Some("libs/sub".to_string()),
        git_remote_url: Some("https://github.com/user/sub.git".to_string()),
        remote_hash: Some("sub123hash".to_string()),
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
        library_mode: None,
        follow_symlinks: false,
        enabled: true,
        cleanup_on_disable: false,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    };
    manager.store_watch_folder(&submodule).await.unwrap();

    // Activate parent should activate submodule too
    let updated = manager.activate_project_group("parent-001").await.unwrap();
    assert_eq!(updated, 2); // Both parent and submodule

    let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
    let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
    assert!(parent_record.is_active);
    assert!(submodule_record.is_active);

    // Deactivate submodule only affects the submodule (recursive goes DOWN, not UP)
    manager.deactivate_project_group("submodule-001").await.unwrap();
    let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
    let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
    assert!(parent_record.is_active); // Parent stays active
    assert!(!submodule_record.is_active);

    // Deactivate from parent deactivates entire group
    manager.deactivate_project_group("parent-001").await.unwrap();
    let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
    assert!(!parent_record.is_active);
}

#[tokio::test]
async fn test_watch_folder_library_config() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("watch_library_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create a library watch folder
    let record = WatchFolderRecord {
        watch_id: "lib-001".to_string(),
        path: "/libraries/my-docs".to_string(),
        collection: "libraries".to_string(),
        tenant_id: "my-docs".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
        library_mode: Some("sync".to_string()),
        follow_symlinks: true,
        enabled: true,
        cleanup_on_disable: true,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    };

    manager.store_watch_folder(&record).await.unwrap();

    // Retrieve and verify library-specific fields
    let retrieved = manager.get_watch_folder("lib-001").await.unwrap().unwrap();
    assert_eq!(retrieved.collection, "libraries");
    assert_eq!(retrieved.library_mode, Some("sync".to_string()));
    assert!(retrieved.follow_symlinks);
    assert!(retrieved.cleanup_on_disable);
}

#[tokio::test]
async fn test_watch_folder_collection_filter() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("watch_filter_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create project watches
    for i in 1..=3 {
        let record = WatchFolderRecord {
            watch_id: format!("project-{}", i),
            path: format!("/projects/proj{}", i),
            collection: "projects".to_string(),
            tenant_id: format!("proj{}-tenant", i),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            last_commit_hash: None,
            is_git_tracked: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&record).await.unwrap();
    }

    // Create library watches
    for i in 1..=2 {
        let record = WatchFolderRecord {
            watch_id: format!("library-{}", i),
            path: format!("/libraries/lib{}", i),
            collection: "libraries".to_string(),
            tenant_id: format!("lib{}", i),
            parent_watch_id: None,
            submodule_path: None,
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: false,
            last_activity_at: None,
            is_paused: false,
            pause_start_time: None,
            is_archived: false,
            last_commit_hash: None,
            is_git_tracked: false,
            library_mode: None,
            follow_symlinks: false,
            enabled: true,
            cleanup_on_disable: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_scan: None,
        };
        manager.store_watch_folder(&record).await.unwrap();
    }

    // Test filter by collection
    let projects = manager.list_watch_folders(Some("projects"), false).await.unwrap();
    assert_eq!(projects.len(), 3);

    let libraries = manager.list_watch_folders(Some("libraries"), false).await.unwrap();
    assert_eq!(libraries.len(), 2);

    // Test no filter
    let all = manager.list_watch_folders(None, false).await.unwrap();
    assert_eq!(all.len(), 5);
}

#[tokio::test]
async fn test_watch_folder_enabled_toggle() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("watch_enabled_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create enabled watch folder
    let record = WatchFolderRecord {
        watch_id: "toggle-test".to_string(),
        path: "/projects/toggle".to_string(),
        collection: "projects".to_string(),
        tenant_id: "toggle-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
        library_mode: None,
        follow_symlinks: false,
        enabled: true,
        cleanup_on_disable: false,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    };
    manager.store_watch_folder(&record).await.unwrap();

    // Verify initially enabled
    let retrieved = manager.get_watch_folder("toggle-test").await.unwrap().unwrap();
    assert!(retrieved.enabled);

    // Disable
    let updated = manager.set_watch_folder_enabled("toggle-test", false).await.unwrap();
    assert!(updated);
    let retrieved = manager.get_watch_folder("toggle-test").await.unwrap().unwrap();
    assert!(!retrieved.enabled);

    // List enabled only should not include disabled watch
    let enabled_only = manager.list_watch_folders(Some("projects"), true).await.unwrap();
    assert_eq!(enabled_only.len(), 0);

    // Re-enable
    let updated = manager.set_watch_folder_enabled("toggle-test", true).await.unwrap();
    assert!(updated);
    let retrieved = manager.get_watch_folder("toggle-test").await.unwrap().unwrap();
    assert!(retrieved.enabled);

    // List enabled only should now include watch
    let enabled_only = manager.list_watch_folders(Some("projects"), true).await.unwrap();
    assert_eq!(enabled_only.len(), 1);
}

#[tokio::test]
async fn test_watch_folder_last_scan_update() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("watch_scan_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create watch folder
    let record = WatchFolderRecord {
        watch_id: "scan-test".to_string(),
        path: "/projects/scan".to_string(),
        collection: "projects".to_string(),
        tenant_id: "scan-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
        library_mode: None,
        follow_symlinks: false,
        enabled: true,
        cleanup_on_disable: false,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    };
    manager.store_watch_folder(&record).await.unwrap();

    // Verify no last_scan initially
    let retrieved = manager.get_watch_folder("scan-test").await.unwrap().unwrap();
    assert!(retrieved.last_scan.is_none());

    // Update last_scan
    let updated = manager.update_watch_folder_last_scan("scan-test").await.unwrap();
    assert!(updated);

    // Verify last_scan is now set
    let retrieved = manager.get_watch_folder("scan-test").await.unwrap().unwrap();
    assert!(retrieved.last_scan.is_some());
}

#[tokio::test]
async fn test_get_watch_folder_by_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_tenant_lookup.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create a project watch folder
    let record = WatchFolderRecord {
        watch_id: "watch-001".to_string(),
        path: "/projects/myproject".to_string(),
        collection: "projects".to_string(),
        tenant_id: "abc123def456".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/user/myproject.git".to_string()),
        remote_hash: Some("abc123hash".to_string()),
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
        library_mode: None,
        follow_symlinks: false,
        enabled: true,
        cleanup_on_disable: false,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    };
    manager.store_watch_folder(&record).await.unwrap();

    // Look up by tenant_id
    let found = manager.get_watch_folder_by_tenant_id("abc123def456", "projects")
        .await.unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().watch_id, "watch-001");

    // Look up non-existent tenant_id
    let not_found = manager.get_watch_folder_by_tenant_id("nonexistent", "projects")
        .await.unwrap();
    assert!(not_found.is_none());

    // Look up wrong collection
    let wrong_collection = manager.get_watch_folder_by_tenant_id("abc123def456", "libraries")
        .await.unwrap();
    assert!(wrong_collection.is_none());
}
