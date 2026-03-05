//! Pool constructor, clone disambiguation, and path registration tests.

use super::*;

#[tokio::test]
async fn test_with_pool_constructor() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_with_pool.db");

    let manager1 = DaemonStateManager::new(&db_path).await.unwrap();
    manager1.initialize().await.unwrap();

    let record = make_test_watch_folder("test-001", "/projects/test", "test-tenant");
    manager1.store_watch_folder(&record).await.unwrap();

    // Create another manager using with_pool - simulate sharing pool
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

    let remote_hash = "abc123hashxyz";

    let record1 = WatchFolderRecord {
        git_remote_url: Some("https://github.com/user/repo.git".to_string()),
        remote_hash: Some(remote_hash.to_string()),
        disambiguation_path: Some("work/project".to_string()),
        ..make_test_watch_folder("clone-001", "/home/user/work/project", "project-tenant-1")
    };

    let record2 = WatchFolderRecord {
        git_remote_url: Some("https://github.com/user/repo.git".to_string()),
        remote_hash: Some(remote_hash.to_string()),
        disambiguation_path: Some("personal/project".to_string()),
        ..make_test_watch_folder(
            "clone-002",
            "/home/user/personal/project",
            "project-tenant-2",
        )
    };

    manager.store_watch_folder(&record1).await.unwrap();
    manager.store_watch_folder(&record2).await.unwrap();

    let clones = manager
        .find_clones_by_remote_hash(remote_hash)
        .await
        .unwrap();
    assert_eq!(clones.len(), 2);

    let tenant_ids: Vec<_> = clones.iter().map(|c| &c.tenant_id).collect();
    assert!(tenant_ids.contains(&&"project-tenant-1".to_string()));
    assert!(tenant_ids.contains(&&"project-tenant-2".to_string()));

    let no_clones = manager
        .find_clones_by_remote_hash("nonexistent")
        .await
        .unwrap();
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

    let remote_hash = calculator.calculate_remote_hash("https://github.com/user/repo.git");
    let tenant_id = calculator.calculate(
        std::path::Path::new("/home/user/work/project"),
        Some("https://github.com/user/repo.git"),
        None,
    );

    let record = WatchFolderRecord {
        tenant_id: tenant_id.clone(),
        git_remote_url: Some("https://github.com/user/repo.git".to_string()),
        remote_hash: Some(remote_hash.clone()),
        ..make_test_watch_folder("first-clone", "/home/user/work/project", "")
    };

    let (result, aliases) = manager
        .register_project_with_disambiguation(record)
        .await
        .unwrap();

    assert!(aliases.is_empty());
    assert!(
        result.disambiguation_path.is_none() || result.disambiguation_path.as_deref() == Some("")
    );
}

/// Build a WatchFolderRecord for disambiguation tests with git remote info.
fn make_disambig_record(
    watch_id: &str,
    path: &str,
    git_remote: &str,
    remote_hash: &str,
    calculator: &crate::project_disambiguation::ProjectIdCalculator,
) -> WatchFolderRecord {
    WatchFolderRecord {
        tenant_id: calculator.calculate(std::path::Path::new(path), Some(git_remote), None),
        git_remote_url: Some(git_remote.to_string()),
        remote_hash: Some(remote_hash.to_string()),
        ..make_test_watch_folder(watch_id, path, "")
    }
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
    let first_record = make_disambig_record(
        "first-clone",
        "/home/user/work/project",
        git_remote,
        &remote_hash,
        &calculator,
    );
    let (first_result, _) = manager
        .register_project_with_disambiguation(first_record)
        .await
        .unwrap();
    let original_tenant_id = first_result.tenant_id.clone();

    // Register second clone (should trigger disambiguation for both)
    let second_record = make_disambig_record(
        "second-clone",
        "/home/user/personal/project",
        git_remote,
        &remote_hash,
        &calculator,
    );
    let (second_result, aliases) = manager
        .register_project_with_disambiguation(second_record)
        .await
        .unwrap();

    assert!(second_result.disambiguation_path.is_some());
    let second_disambig = second_result.disambiguation_path.as_ref().unwrap();
    assert!(
        second_disambig.contains("personal"),
        "Expected 'personal' in disambiguation path: {}",
        second_disambig
    );

    let updated_first = manager
        .get_watch_folder("first-clone")
        .await
        .unwrap()
        .unwrap();
    assert!(updated_first.disambiguation_path.is_some());
    let first_disambig = updated_first.disambiguation_path.as_ref().unwrap();
    assert!(
        first_disambig.contains("work"),
        "Expected 'work' in disambiguation path: {}",
        first_disambig
    );

    assert_ne!(
        updated_first.tenant_id, second_result.tenant_id,
        "Both clones should have different tenant_ids"
    );

    assert!(
        !aliases.is_empty(),
        "Should have created alias for first clone"
    );
    let (old_id, new_id) = &aliases[0];
    assert_eq!(
        old_id, &original_tenant_id,
        "Alias should map from original tenant_id"
    );
    assert_eq!(
        new_id, &updated_first.tenant_id,
        "Alias should map to new tenant_id"
    );
}

#[tokio::test]
async fn test_is_path_registered() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_path_registered.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let record = make_test_watch_folder("test-project", "/home/user/myproject", "test-tenant");

    assert!(!manager
        .is_path_registered("/home/user/myproject")
        .await
        .unwrap());

    manager.store_watch_folder(&record).await.unwrap();

    assert!(manager
        .is_path_registered("/home/user/myproject")
        .await
        .unwrap());
    assert!(!manager
        .is_path_registered("/home/user/other")
        .await
        .unwrap());
}
