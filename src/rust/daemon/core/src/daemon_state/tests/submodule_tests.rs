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
        git_remote_url: Some("https://github.com/org/project-a.git".to_string()),
        remote_hash: Some("hash-a".to_string()),
        is_active: true,
        ..make_test_watch_folder("project-a", "/projects/a", "a-tenant")
    };
    manager.store_watch_folder(&project_a).await.unwrap();

    let sub_a = WatchFolderRecord {
        parent_watch_id: Some("project-a".to_string()),
        submodule_path: Some("submodules/s1".to_string()),
        git_remote_url: Some("https://github.com/org/shared-lib.git".to_string()),
        remote_hash: Some("hash-shared".to_string()),
        is_active: true,
        ..make_test_watch_folder("sub-s1-under-a", "/projects/a/submodules/s1", "s1-tenant")
    };
    manager.store_watch_folder(&sub_a).await.unwrap();

    // Project B with same submodule S1
    let project_b = WatchFolderRecord {
        git_remote_url: Some("https://github.com/org/project-b.git".to_string()),
        remote_hash: Some("hash-b".to_string()),
        is_active: true,
        ..make_test_watch_folder("project-b", "/projects/b", "b-tenant")
    };
    manager.store_watch_folder(&project_b).await.unwrap();

    let sub_b = WatchFolderRecord {
        parent_watch_id: Some("project-b".to_string()),
        submodule_path: Some("submodules/s1".to_string()),
        git_remote_url: Some("https://github.com/org/shared-lib.git".to_string()),
        remote_hash: Some("hash-shared".to_string()),
        is_active: true,
        ..make_test_watch_folder("sub-s1-under-b", "/projects/b/submodules/s1", "s1-tenant-b")
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
        is_active: true,
        ..make_test_watch_folder("parent-001", "/projects/parent", "parent-tenant")
    };
    manager.store_watch_folder(&parent).await.unwrap();

    // Add 2 submodules
    for i in 1..=2 {
        let sub = WatchFolderRecord {
            parent_watch_id: Some("parent-001".to_string()),
            submodule_path: Some(format!("sub{}", i)),
            is_active: true,
            ..make_test_watch_folder(
                &format!("sub-{}", i),
                &format!("/projects/parent/sub{}", i),
                &format!("sub-{}-tenant", i),
            )
        };
        manager.store_watch_folder(&sub).await.unwrap();
    }

    let subs = manager.get_submodules_for_project("parent-001").await.unwrap();
    assert_eq!(subs.len(), 2);

    // No submodules for unknown parent
    let empty = manager.get_submodules_for_project("nonexistent").await.unwrap();
    assert!(empty.is_empty());
}
