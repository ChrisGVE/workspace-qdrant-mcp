//! Activate/deactivate by tenant, heartbeat, recursive activity inheritance,
//! and nonexistent tenant tests.

use super::*;

/// Set up a manager with a parent + submodule pair.
async fn setup_parent_with_submodule(
    db_name: &str,
    parent_active: bool,
) -> (tempfile::TempDir, DaemonStateManager) {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join(db_name);
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let parent = WatchFolderRecord {
        is_active: parent_active,
        last_activity_at: if parent_active {
            Some(Utc::now())
        } else {
            None
        },
        ..make_test_watch_folder("parent-001", "/projects/parent", "parent-tenant")
    };
    manager.store_watch_folder(&parent).await.unwrap();

    let submodule = WatchFolderRecord {
        parent_watch_id: Some("parent-001".to_string()),
        submodule_path: Some("sub".to_string()),
        is_active: parent_active,
        last_activity_at: if parent_active {
            Some(Utc::now())
        } else {
            None
        },
        ..make_test_watch_folder("submodule-001", "/projects/parent/sub", "sub-tenant")
    };
    manager.store_watch_folder(&submodule).await.unwrap();

    (temp_dir, manager)
}

#[tokio::test]
async fn test_activate_project_by_tenant_id() {
    let (_tmp, manager) = setup_parent_with_submodule("test_tenant_activate.db", false).await;

    let (affected, watch_id) = manager
        .activate_project_by_tenant_id("parent-tenant")
        .await
        .unwrap();

    assert_eq!(affected, 2);
    assert_eq!(watch_id, Some("parent-001".to_string()));

    let parent_record = manager
        .get_watch_folder("parent-001")
        .await
        .unwrap()
        .unwrap();
    let submodule_record = manager
        .get_watch_folder("submodule-001")
        .await
        .unwrap()
        .unwrap();
    assert!(parent_record.is_active);
    assert!(submodule_record.is_active);
    assert!(parent_record.last_activity_at.is_some());
    assert!(submodule_record.last_activity_at.is_some());
}

#[tokio::test]
async fn test_deactivate_project_by_tenant_id() {
    let (_tmp, manager) = setup_parent_with_submodule("test_tenant_deactivate.db", true).await;

    let (affected, watch_id) = manager
        .deactivate_project_by_tenant_id("parent-tenant")
        .await
        .unwrap();

    assert_eq!(affected, 2);
    assert_eq!(watch_id, Some("parent-001".to_string()));

    let parent_record = manager
        .get_watch_folder("parent-001")
        .await
        .unwrap()
        .unwrap();
    let submodule_record = manager
        .get_watch_folder("submodule-001")
        .await
        .unwrap()
        .unwrap();
    assert!(!parent_record.is_active);
    assert!(!submodule_record.is_active);
}

#[tokio::test]
async fn test_heartbeat_project_group() {
    let (_tmp, manager) = setup_parent_with_submodule("test_heartbeat.db", true).await;

    let affected = manager.heartbeat_project_group("parent-001").await.unwrap();
    assert_eq!(affected, 2);

    let parent_record = manager
        .get_watch_folder("parent-001")
        .await
        .unwrap()
        .unwrap();
    let submodule_record = manager
        .get_watch_folder("submodule-001")
        .await
        .unwrap()
        .unwrap();
    assert!(parent_record.last_activity_at.is_some());
    assert!(submodule_record.last_activity_at.is_some());
}

#[tokio::test]
async fn test_heartbeat_project_by_tenant_id() {
    let (_tmp, manager) = setup_parent_with_submodule("test_heartbeat_tenant.db", true).await;

    let (affected, watch_id) = manager
        .heartbeat_project_by_tenant_id("parent-tenant")
        .await
        .unwrap();

    assert_eq!(affected, 2);
    assert_eq!(watch_id, Some("parent-001".to_string()));

    let parent_record = manager
        .get_watch_folder("parent-001")
        .await
        .unwrap()
        .unwrap();
    let submodule_record = manager
        .get_watch_folder("submodule-001")
        .await
        .unwrap()
        .unwrap();
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
    let root = make_test_watch_folder("root-001", "/projects/root", "root-tenant");
    manager.store_watch_folder(&root).await.unwrap();

    let mid = WatchFolderRecord {
        parent_watch_id: Some("root-001".to_string()),
        submodule_path: Some("libs/mid".to_string()),
        ..make_test_watch_folder("mid-001", "/projects/root/libs/mid", "mid-tenant")
    };
    manager.store_watch_folder(&mid).await.unwrap();

    let leaf = WatchFolderRecord {
        parent_watch_id: Some("mid-001".to_string()),
        submodule_path: Some("deps/leaf".to_string()),
        ..make_test_watch_folder(
            "leaf-001",
            "/projects/root/libs/mid/deps/leaf",
            "leaf-tenant",
        )
    };
    manager.store_watch_folder(&leaf).await.unwrap();

    // Activate from root should activate all 3 levels
    let affected = manager.activate_project_group("root-001").await.unwrap();
    assert_eq!(affected, 3);

    for id in &["root-001", "mid-001", "leaf-001"] {
        let r = manager.get_watch_folder(id).await.unwrap().unwrap();
        assert!(r.is_active);
        assert!(r.last_activity_at.is_some());
    }

    // Heartbeat from root should touch all 3 levels
    assert_eq!(
        manager.heartbeat_project_group("root-001").await.unwrap(),
        3
    );

    // Deactivate from root should deactivate all 3 levels
    assert_eq!(
        manager.deactivate_project_group("root-001").await.unwrap(),
        3
    );

    for id in &["root-001", "mid-001", "leaf-001"] {
        let r = manager.get_watch_folder(id).await.unwrap().unwrap();
        assert!(!r.is_active);
    }

    // Activate from mid should only activate mid and leaf (not root)
    let affected = manager.activate_project_group("mid-001").await.unwrap();
    assert_eq!(affected, 2);

    let root_r = manager.get_watch_folder("root-001").await.unwrap().unwrap();
    let mid_r = manager.get_watch_folder("mid-001").await.unwrap().unwrap();
    let leaf_r = manager.get_watch_folder("leaf-001").await.unwrap().unwrap();
    assert!(!root_r.is_active);
    assert!(mid_r.is_active);
    assert!(leaf_r.is_active);
}

#[tokio::test]
async fn test_activate_nonexistent_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_nonexistent.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let (affected, watch_id) = manager
        .activate_project_by_tenant_id("nonexistent")
        .await
        .unwrap();

    assert_eq!(affected, 0);
    assert!(watch_id.is_none());
}
