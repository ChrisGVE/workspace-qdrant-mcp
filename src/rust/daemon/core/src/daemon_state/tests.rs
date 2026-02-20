//! Tests for daemon state management.

use super::*;
use chrono::Utc;
use tempfile::tempdir;

#[tokio::test]
async fn test_daemon_state_creation() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("daemon_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    assert!(db_path.exists());
}

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

#[tokio::test]
async fn test_activate_project_by_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_tenant_activate.db");

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
    manager.store_watch_folder(&parent).await.unwrap();

    // Create submodule
    let submodule = WatchFolderRecord {
        watch_id: "submodule-001".to_string(),
        path: "/projects/parent/sub".to_string(),
        collection: "projects".to_string(),
        tenant_id: "sub-tenant".to_string(),
        parent_watch_id: Some("parent-001".to_string()),
        submodule_path: Some("sub".to_string()),
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
    manager.store_watch_folder(&submodule).await.unwrap();

    // Activate by tenant_id
    let (affected, watch_id) = manager.activate_project_by_tenant_id("parent-tenant")
        .await.unwrap();

    assert_eq!(affected, 2); // Parent and submodule
    assert_eq!(watch_id, Some("parent-001".to_string()));

    // Verify both are active
    let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
    let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
    assert!(parent_record.is_active);
    assert!(submodule_record.is_active);
    assert!(parent_record.last_activity_at.is_some());
    assert!(submodule_record.last_activity_at.is_some());
}

#[tokio::test]
async fn test_deactivate_project_by_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_tenant_deactivate.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create parent project (active)
    let parent = WatchFolderRecord {
        watch_id: "parent-001".to_string(),
        path: "/projects/parent".to_string(),
        collection: "projects".to_string(),
        tenant_id: "parent-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: Some(Utc::now()),
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

    // Create submodule (active)
    let submodule = WatchFolderRecord {
        watch_id: "submodule-001".to_string(),
        path: "/projects/parent/sub".to_string(),
        collection: "projects".to_string(),
        tenant_id: "sub-tenant".to_string(),
        parent_watch_id: Some("parent-001".to_string()),
        submodule_path: Some("sub".to_string()),
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: Some(Utc::now()),
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

    // Deactivate by tenant_id
    let (affected, watch_id) = manager.deactivate_project_by_tenant_id("parent-tenant")
        .await.unwrap();

    assert_eq!(affected, 2); // Parent and submodule
    assert_eq!(watch_id, Some("parent-001".to_string()));

    // Verify both are inactive
    let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
    let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
    assert!(!parent_record.is_active);
    assert!(!submodule_record.is_active);
}

#[tokio::test]
async fn test_heartbeat_project_group() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_heartbeat.db");

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
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: None, // No activity yet
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
        path: "/projects/parent/sub".to_string(),
        collection: "projects".to_string(),
        tenant_id: "sub-tenant".to_string(),
        parent_watch_id: Some("parent-001".to_string()),
        submodule_path: Some("sub".to_string()),
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: None, // No activity yet
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

    // Send heartbeat
    let affected = manager.heartbeat_project_group("parent-001").await.unwrap();
    assert_eq!(affected, 2); // Parent and submodule

    // Verify both have activity timestamps
    let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
    let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
    assert!(parent_record.last_activity_at.is_some());
    assert!(submodule_record.last_activity_at.is_some());
}

#[tokio::test]
async fn test_heartbeat_project_by_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_heartbeat_tenant.db");

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
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
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
        path: "/projects/parent/sub".to_string(),
        collection: "projects".to_string(),
        tenant_id: "sub-tenant".to_string(),
        parent_watch_id: Some("parent-001".to_string()),
        submodule_path: Some("sub".to_string()),
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
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

    // Heartbeat by tenant_id
    let (affected, watch_id) = manager.heartbeat_project_by_tenant_id("parent-tenant")
        .await.unwrap();

    assert_eq!(affected, 2); // Parent and submodule
    assert_eq!(watch_id, Some("parent-001".to_string()));

    // Verify both have activity timestamps
    let parent_record = manager.get_watch_folder("parent-001").await.unwrap().unwrap();
    let submodule_record = manager.get_watch_folder("submodule-001").await.unwrap().unwrap();
    assert!(parent_record.last_activity_at.is_some());
    assert!(submodule_record.last_activity_at.is_some());
}

#[tokio::test]
async fn test_recursive_activity_inheritance_3_levels() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_3level_recursive.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create 3-level hierarchy: root -> mid -> leaf
    let root = WatchFolderRecord {
        watch_id: "root-001".to_string(),
        path: "/projects/root".to_string(),
        collection: "projects".to_string(),
        tenant_id: "root-tenant".to_string(),
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
    manager.store_watch_folder(&root).await.unwrap();

    let mid = WatchFolderRecord {
        watch_id: "mid-001".to_string(),
        path: "/projects/root/libs/mid".to_string(),
        collection: "projects".to_string(),
        tenant_id: "mid-tenant".to_string(),
        parent_watch_id: Some("root-001".to_string()),
        submodule_path: Some("libs/mid".to_string()),
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
    manager.store_watch_folder(&mid).await.unwrap();

    let leaf = WatchFolderRecord {
        watch_id: "leaf-001".to_string(),
        path: "/projects/root/libs/mid/deps/leaf".to_string(),
        collection: "projects".to_string(),
        tenant_id: "leaf-tenant".to_string(),
        parent_watch_id: Some("mid-001".to_string()),
        submodule_path: Some("deps/leaf".to_string()),
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
    manager.store_watch_folder(&leaf).await.unwrap();

    // Activate from root should activate all 3 levels
    let affected = manager.activate_project_group("root-001").await.unwrap();
    assert_eq!(affected, 3);

    let root_r = manager.get_watch_folder("root-001").await.unwrap().unwrap();
    let mid_r = manager.get_watch_folder("mid-001").await.unwrap().unwrap();
    let leaf_r = manager.get_watch_folder("leaf-001").await.unwrap().unwrap();
    assert!(root_r.is_active);
    assert!(mid_r.is_active);
    assert!(leaf_r.is_active);
    assert!(root_r.last_activity_at.is_some());
    assert!(mid_r.last_activity_at.is_some());
    assert!(leaf_r.last_activity_at.is_some());

    // Heartbeat from root should touch all 3 levels
    let hb = manager.heartbeat_project_group("root-001").await.unwrap();
    assert_eq!(hb, 3);

    // Deactivate from root should deactivate all 3 levels
    let deact = manager.deactivate_project_group("root-001").await.unwrap();
    assert_eq!(deact, 3);

    let root_r = manager.get_watch_folder("root-001").await.unwrap().unwrap();
    let mid_r = manager.get_watch_folder("mid-001").await.unwrap().unwrap();
    let leaf_r = manager.get_watch_folder("leaf-001").await.unwrap().unwrap();
    assert!(!root_r.is_active);
    assert!(!mid_r.is_active);
    assert!(!leaf_r.is_active);

    // Activate from mid should only activate mid and leaf (not root)
    let affected = manager.activate_project_group("mid-001").await.unwrap();
    assert_eq!(affected, 2);

    let root_r = manager.get_watch_folder("root-001").await.unwrap().unwrap();
    let mid_r = manager.get_watch_folder("mid-001").await.unwrap().unwrap();
    let leaf_r = manager.get_watch_folder("leaf-001").await.unwrap().unwrap();
    assert!(!root_r.is_active); // Root stays inactive
    assert!(mid_r.is_active);
    assert!(leaf_r.is_active);
}

#[tokio::test]
async fn test_activate_nonexistent_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_nonexistent.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Activate non-existent tenant should return 0 affected
    let (affected, watch_id) = manager.activate_project_by_tenant_id("nonexistent")
        .await.unwrap();

    assert_eq!(affected, 0);
    assert!(watch_id.is_none());
}

#[tokio::test]
async fn test_with_pool_constructor() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_with_pool.db");

    // Create manager with new() to get the pool set up
    let manager1 = DaemonStateManager::new(&db_path).await.unwrap();
    manager1.initialize().await.unwrap();

    // Create a watch folder
    let record = WatchFolderRecord {
        watch_id: "test-001".to_string(),
        path: "/projects/test".to_string(),
        collection: "projects".to_string(),
        tenant_id: "test-tenant".to_string(),
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
    manager1.store_watch_folder(&record).await.unwrap();

    // Create another manager using with_pool - simulate sharing pool
    // Note: This simulates the gRPC service scenario
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    let connect_options = SqliteConnectOptions::new()
        .filename(&db_path)
        .create_if_missing(false);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(connect_options)
        .await
        .unwrap();

    let manager2 = DaemonStateManager::with_pool(pool);

    // Should be able to read the data
    let found = manager2.get_watch_folder("test-001").await.unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().tenant_id, "test-tenant");
}

// ========================================================================
// Disambiguation Tests (Task 3)
// ========================================================================

#[tokio::test]
async fn test_find_clones_by_remote_hash() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_find_clones.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create two projects with the same remote_hash (same repo, different paths)
    let remote_hash = "abc123hashxyz";

    let record1 = WatchFolderRecord {
        watch_id: "clone-001".to_string(),
        path: "/home/user/work/project".to_string(),
        collection: "projects".to_string(),
        tenant_id: "project-tenant-1".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/user/repo.git".to_string()),
        remote_hash: Some(remote_hash.to_string()),
        disambiguation_path: Some("work/project".to_string()),
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

    let record2 = WatchFolderRecord {
        watch_id: "clone-002".to_string(),
        path: "/home/user/personal/project".to_string(),
        collection: "projects".to_string(),
        tenant_id: "project-tenant-2".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/user/repo.git".to_string()),
        remote_hash: Some(remote_hash.to_string()),
        disambiguation_path: Some("personal/project".to_string()),
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

    manager.store_watch_folder(&record1).await.unwrap();
    manager.store_watch_folder(&record2).await.unwrap();

    // Find clones by remote_hash
    let clones = manager.find_clones_by_remote_hash(remote_hash).await.unwrap();
    assert_eq!(clones.len(), 2);

    // Verify different tenant_ids
    let tenant_ids: Vec<_> = clones.iter().map(|c| &c.tenant_id).collect();
    assert!(tenant_ids.contains(&&"project-tenant-1".to_string()));
    assert!(tenant_ids.contains(&&"project-tenant-2".to_string()));

    // Search for non-existent hash
    let no_clones = manager.find_clones_by_remote_hash("nonexistent").await.unwrap();
    assert!(no_clones.is_empty());
}

#[tokio::test]
async fn test_register_project_with_disambiguation_first_clone() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_disambig_first.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    use crate::project_disambiguation::ProjectIdCalculator;
    let calculator = ProjectIdCalculator::new();

    // Register first clone (no disambiguation needed)
    let remote_hash = calculator.calculate_remote_hash("https://github.com/user/repo.git");
    let tenant_id = calculator.calculate(
        std::path::Path::new("/home/user/work/project"),
        Some("https://github.com/user/repo.git"),
        None,
    );

    let record = WatchFolderRecord {
        watch_id: "first-clone".to_string(),
        path: "/home/user/work/project".to_string(),
        collection: "projects".to_string(),
        tenant_id: tenant_id.clone(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/user/repo.git".to_string()),
        remote_hash: Some(remote_hash.clone()),
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

    let (result, aliases) = manager.register_project_with_disambiguation(record).await.unwrap();

    // First clone should have no aliases
    assert!(aliases.is_empty());

    // First clone should not need disambiguation
    assert!(result.disambiguation_path.is_none() || result.disambiguation_path.as_deref() == Some(""));
}

#[tokio::test]
async fn test_register_project_with_disambiguation_second_clone() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_disambig_second.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    use crate::project_disambiguation::ProjectIdCalculator;
    let calculator = ProjectIdCalculator::new();

    let git_remote = "https://github.com/user/repo.git";
    let remote_hash = calculator.calculate_remote_hash(git_remote);

    // Register first clone
    let first_record = WatchFolderRecord {
        watch_id: "first-clone".to_string(),
        path: "/home/user/work/project".to_string(),
        collection: "projects".to_string(),
        tenant_id: calculator.calculate(
            std::path::Path::new("/home/user/work/project"),
            Some(git_remote),
            None,
        ),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some(git_remote.to_string()),
        remote_hash: Some(remote_hash.clone()),
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

    let (first_result, _) = manager.register_project_with_disambiguation(first_record).await.unwrap();
    let original_tenant_id = first_result.tenant_id.clone();

    // Register second clone (should trigger disambiguation for both)
    let second_record = WatchFolderRecord {
        watch_id: "second-clone".to_string(),
        path: "/home/user/personal/project".to_string(),
        collection: "projects".to_string(),
        tenant_id: calculator.calculate(
            std::path::Path::new("/home/user/personal/project"),
            Some(git_remote),
            None,
        ),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some(git_remote.to_string()),
        remote_hash: Some(remote_hash.clone()),
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

    let (second_result, aliases) = manager.register_project_with_disambiguation(second_record).await.unwrap();

    // Second clone should have disambiguation path
    assert!(second_result.disambiguation_path.is_some());
    let second_disambig = second_result.disambiguation_path.as_ref().unwrap();
    assert!(second_disambig.contains("personal"), "Expected 'personal' in disambiguation path: {}", second_disambig);

    // Verify first clone was updated with disambiguation
    let updated_first = manager.get_watch_folder("first-clone").await.unwrap().unwrap();
    assert!(updated_first.disambiguation_path.is_some());
    let first_disambig = updated_first.disambiguation_path.as_ref().unwrap();
    assert!(first_disambig.contains("work"), "Expected 'work' in disambiguation path: {}", first_disambig);

    // Verify tenant_ids are now different
    assert_ne!(updated_first.tenant_id, second_result.tenant_id,
        "Both clones should have different tenant_ids");

    // Verify alias was created for the first clone
    assert!(!aliases.is_empty(), "Should have created alias for first clone");
    let (old_id, new_id) = &aliases[0];
    assert_eq!(old_id, &original_tenant_id, "Alias should map from original tenant_id");
    assert_eq!(new_id, &updated_first.tenant_id, "Alias should map to new tenant_id");
}

#[tokio::test]
async fn test_is_path_registered() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_path_registered.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let record = WatchFolderRecord {
        watch_id: "test-project".to_string(),
        path: "/home/user/myproject".to_string(),
        collection: "projects".to_string(),
        tenant_id: "test-tenant".to_string(),
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

    // Path should not be registered initially
    assert!(!manager.is_path_registered("/home/user/myproject").await.unwrap());

    // Register the project
    manager.store_watch_folder(&record).await.unwrap();

    // Path should now be registered
    assert!(manager.is_path_registered("/home/user/myproject").await.unwrap());

    // Different path should not be registered
    assert!(!manager.is_path_registered("/home/user/other").await.unwrap());
}

#[tokio::test]
async fn test_any_watchers_paused() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("pause_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // No watchers at all - should return false
    assert!(!manager.any_watchers_paused().await.unwrap());

    // Add an enabled, non-paused watcher
    let record = WatchFolderRecord {
        watch_id: "pause-test-001".to_string(),
        path: "/projects/pause-test".to_string(),
        collection: "projects".to_string(),
        tenant_id: "pause-test-tenant".to_string(),
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

    // Not paused yet
    assert!(!manager.any_watchers_paused().await.unwrap());

    // Pause all watchers
    let paused = manager.pause_all_watchers().await.unwrap();
    assert_eq!(paused, 1);
    assert!(manager.any_watchers_paused().await.unwrap());

    // Resume
    let resumed = manager.resume_all_watchers().await.unwrap();
    assert_eq!(resumed, 1);
    assert!(!manager.any_watchers_paused().await.unwrap());
}

#[tokio::test]
async fn test_poll_pause_state() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("poll_pause_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let flag = std::sync::atomic::AtomicBool::new(false);

    // No change when no watchers
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);
    assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));

    // Add enabled watcher
    let record = WatchFolderRecord {
        watch_id: "poll-test-001".to_string(),
        path: "/projects/poll-test".to_string(),
        collection: "projects".to_string(),
        tenant_id: "poll-test-tenant".to_string(),
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

    // Still no change (not paused)
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);

    // Pause via DB
    manager.pause_all_watchers().await.unwrap();

    // Flag should change from false -> true
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(changed);
    assert!(flag.load(std::sync::atomic::Ordering::SeqCst));

    // Poll again - no change (already true)
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);

    // Resume via DB
    manager.resume_all_watchers().await.unwrap();

    // Flag should change from true -> false
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(changed);
    assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test]
async fn test_deactivate_inactive_projects() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_inactivity.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create an active project with old last_activity_at
    let old_project = WatchFolderRecord {
        watch_id: "old-project".to_string(),
        path: "/projects/old".to_string(),
        collection: "projects".to_string(),
        tenant_id: "old-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: Some(Utc::now() - chrono::Duration::hours(13)),
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
    manager.store_watch_folder(&old_project).await.unwrap();

    // Create an active project with recent activity
    let recent_project = WatchFolderRecord {
        watch_id: "recent-project".to_string(),
        path: "/projects/recent".to_string(),
        collection: "projects".to_string(),
        tenant_id: "recent-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: Some(Utc::now() - chrono::Duration::minutes(30)),
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
    manager.store_watch_folder(&recent_project).await.unwrap();

    // 12-hour timeout (43200 seconds)
    let deactivated = manager.deactivate_inactive_projects(43200).await.unwrap();
    assert_eq!(deactivated, 1, "Only the 13h-old project should be deactivated");

    // Verify old project is deactivated
    let old = manager.get_watch_folder("old-project").await.unwrap().unwrap();
    assert!(!old.is_active);

    // Verify recent project is still active
    let recent = manager.get_watch_folder("recent-project").await.unwrap().unwrap();
    assert!(recent.is_active);
}

#[tokio::test]
async fn test_deactivate_inactive_skips_null_activity() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_inactivity_null.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create active project with NULL last_activity_at (never had a session)
    let null_project = WatchFolderRecord {
        watch_id: "null-project".to_string(),
        path: "/projects/null".to_string(),
        collection: "projects".to_string(),
        tenant_id: "null-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
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
    manager.store_watch_folder(&null_project).await.unwrap();

    // Should not deactivate projects with NULL last_activity_at
    let deactivated = manager.deactivate_inactive_projects(43200).await.unwrap();
    assert_eq!(deactivated, 0);

    let record = manager.get_watch_folder("null-project").await.unwrap().unwrap();
    assert!(record.is_active);
}

#[tokio::test]
async fn test_archive_unarchive_watch_folder() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("archive_test.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let record = WatchFolderRecord {
        watch_id: "archive-test-001".to_string(),
        path: "/projects/archive-test".to_string(),
        collection: "projects".to_string(),
        tenant_id: "archive-test-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
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

    // Archive
    let archived = manager.archive_watch_folder("archive-test-001").await.unwrap();
    assert!(archived);
    let r = manager.get_watch_folder("archive-test-001").await.unwrap().unwrap();
    assert!(r.is_archived);

    // Idempotent: archiving again returns false
    let archived_again = manager.archive_watch_folder("archive-test-001").await.unwrap();
    assert!(!archived_again);

    // Unarchive
    let unarchived = manager.unarchive_watch_folder("archive-test-001").await.unwrap();
    assert!(unarchived);
    let r = manager.get_watch_folder("archive-test-001").await.unwrap().unwrap();
    assert!(!r.is_archived);

    // Idempotent: unarchiving again returns false
    let unarchived_again = manager.unarchive_watch_folder("archive-test-001").await.unwrap();
    assert!(!unarchived_again);
}

#[tokio::test]
async fn test_submodule_archive_safety_shared_submodule() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("submodule_safety_test.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Project A with submodule S1
    let project_a = WatchFolderRecord {
        watch_id: "project-a".to_string(),
        path: "/projects/a".to_string(),
        collection: "projects".to_string(),
        tenant_id: "a-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/org/project-a.git".to_string()),
        remote_hash: Some("hash-a".to_string()),
        disambiguation_path: None,
        is_active: true,
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
    manager.store_watch_folder(&project_a).await.unwrap();

    // Submodule S1 under project A
    let sub_a = WatchFolderRecord {
        watch_id: "sub-s1-under-a".to_string(),
        path: "/projects/a/submodules/s1".to_string(),
        collection: "projects".to_string(),
        tenant_id: "s1-tenant".to_string(),
        parent_watch_id: Some("project-a".to_string()),
        submodule_path: Some("submodules/s1".to_string()),
        git_remote_url: Some("https://github.com/org/shared-lib.git".to_string()),
        remote_hash: Some("hash-shared".to_string()),
        disambiguation_path: None,
        is_active: true,
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
    manager.store_watch_folder(&sub_a).await.unwrap();

    // Project B with same submodule S1
    let project_b = WatchFolderRecord {
        watch_id: "project-b".to_string(),
        path: "/projects/b".to_string(),
        collection: "projects".to_string(),
        tenant_id: "b-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: Some("https://github.com/org/project-b.git".to_string()),
        remote_hash: Some("hash-b".to_string()),
        disambiguation_path: None,
        is_active: true,
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
    manager.store_watch_folder(&project_b).await.unwrap();

    // Same submodule S1 under project B
    let sub_b = WatchFolderRecord {
        watch_id: "sub-s1-under-b".to_string(),
        path: "/projects/b/submodules/s1".to_string(),
        collection: "projects".to_string(),
        tenant_id: "s1-tenant-b".to_string(),
        parent_watch_id: Some("project-b".to_string()),
        submodule_path: Some("submodules/s1".to_string()),
        git_remote_url: Some("https://github.com/org/shared-lib.git".to_string()),
        remote_hash: Some("hash-shared".to_string()),
        disambiguation_path: None,
        is_active: true,
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
    manager.store_watch_folder(&sub_b).await.unwrap();

    // Archive project A -- submodule S1 should be SKIPPED (project B still active)
    let (archived, skipped) = manager.archive_project_with_submodules("project-a").await.unwrap();
    assert_eq!(archived.len(), 0);
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0], "sub-s1-under-a");

    // Verify parent A is archived but submodule under A stays active
    let a = manager.get_watch_folder("project-a").await.unwrap().unwrap();
    assert!(a.is_archived);
    let sa = manager.get_watch_folder("sub-s1-under-a").await.unwrap().unwrap();
    assert!(!sa.is_archived);
    // parent_watch_id preserved
    assert_eq!(sa.parent_watch_id.as_deref(), Some("project-a"));

    // Archive project B -- now submodule S1 should also be archived
    let (archived, skipped) = manager.archive_project_with_submodules("project-b").await.unwrap();
    assert_eq!(archived.len(), 1);
    assert_eq!(archived[0], "sub-s1-under-b");
    assert_eq!(skipped.len(), 0);

    let sb = manager.get_watch_folder("sub-s1-under-b").await.unwrap().unwrap();
    assert!(sb.is_archived);
    assert_eq!(sb.parent_watch_id.as_deref(), Some("project-b"));
}

#[tokio::test]
async fn test_get_submodules_for_project() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("submodules_list_test.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let parent = WatchFolderRecord {
        watch_id: "parent-001".to_string(),
        path: "/projects/parent".to_string(),
        collection: "projects".to_string(),
        tenant_id: "parent-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
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

    // Add 2 submodules
    for i in 1..=2 {
        let sub = WatchFolderRecord {
            watch_id: format!("sub-{}", i),
            path: format!("/projects/parent/sub{}", i),
            collection: "projects".to_string(),
            tenant_id: format!("sub-{}-tenant", i),
            parent_watch_id: Some("parent-001".to_string()),
            submodule_path: Some(format!("sub{}", i)),
            git_remote_url: None,
            remote_hash: None,
            disambiguation_path: None,
            is_active: true,
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
        manager.store_watch_folder(&sub).await.unwrap();
    }

    let subs = manager.get_submodules_for_project("parent-001").await.unwrap();
    assert_eq!(subs.len(), 2);

    // No submodules for unknown parent
    let empty = manager.get_submodules_for_project("nonexistent").await.unwrap();
    assert!(empty.is_empty());
}

#[tokio::test]
async fn test_operational_state_set_get_roundtrip() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_test.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Set a global value
    set_operational_state(pool, "last_log_prune", "daemon", "2026-02-17T00:00:00Z", None)
        .await.unwrap();

    // Get it back
    let val = get_operational_state(pool, "last_log_prune", "daemon", None)
        .await.unwrap();
    assert_eq!(val, Some("2026-02-17T00:00:00Z".to_string()));
}

#[tokio::test]
async fn test_operational_state_upsert() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_upsert.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Set initial value
    set_operational_state(pool, "key1", "daemon", "value1", None)
        .await.unwrap();

    // Overwrite with upsert
    set_operational_state(pool, "key1", "daemon", "value2", None)
        .await.unwrap();

    let val = get_operational_state(pool, "key1", "daemon", None)
        .await.unwrap();
    assert_eq!(val, Some("value2".to_string()));
}

#[tokio::test]
async fn test_operational_state_component_isolation() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_isolation.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Same key, different components
    set_operational_state(pool, "version", "daemon", "1.0", None).await.unwrap();
    set_operational_state(pool, "version", "server", "2.0", None).await.unwrap();
    set_operational_state(pool, "version", "cli", "3.0", None).await.unwrap();

    assert_eq!(
        get_operational_state(pool, "version", "daemon", None).await.unwrap(),
        Some("1.0".to_string())
    );
    assert_eq!(
        get_operational_state(pool, "version", "server", None).await.unwrap(),
        Some("2.0".to_string())
    );
    assert_eq!(
        get_operational_state(pool, "version", "cli", None).await.unwrap(),
        Some("3.0".to_string())
    );
}

#[tokio::test]
async fn test_operational_state_project_scoped() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_project.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    // Global entry
    set_operational_state(pool, "status", "daemon", "running", None).await.unwrap();
    // Project-scoped entry
    set_operational_state(pool, "status", "daemon", "indexing", Some("proj-1")).await.unwrap();

    // Global lookup should NOT return project-scoped value
    let global = get_operational_state(pool, "status", "daemon", None).await.unwrap();
    assert_eq!(global, Some("running".to_string()));

    // Project lookup should return project-scoped value
    let proj = get_operational_state(pool, "status", "daemon", Some("proj-1")).await.unwrap();
    assert_eq!(proj, Some("indexing".to_string()));

    // Unknown project returns None
    let unknown = get_operational_state(pool, "status", "daemon", Some("proj-999")).await.unwrap();
    assert_eq!(unknown, None);
}

#[tokio::test]
async fn test_operational_state_missing_key() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("op_state_missing.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    let pool = manager.pool();

    let val = get_operational_state(pool, "nonexistent", "daemon", None).await.unwrap();
    assert_eq!(val, None);
}
