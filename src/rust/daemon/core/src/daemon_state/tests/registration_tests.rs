//! Pool constructor, clone disambiguation, and path registration tests.

use super::*;

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
