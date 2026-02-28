//! Activate/deactivate by tenant, heartbeat, recursive activity inheritance,
//! and nonexistent tenant tests.

use super::*;

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
