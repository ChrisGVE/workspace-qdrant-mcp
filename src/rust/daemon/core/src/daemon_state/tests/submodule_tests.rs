//! Submodule archive safety and submodule listing tests.

use super::*;

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
