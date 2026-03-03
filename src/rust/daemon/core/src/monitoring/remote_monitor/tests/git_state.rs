use super::*;
use tempfile::TempDir;

use git2::Repository;

#[tokio::test]
async fn test_git_state_check_no_change_local() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create a local (non-git) temp directory
    let temp = TempDir::new().unwrap();

    // Insert watch_folder as local (is_git_tracked=0, no remote)
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            is_git_tracked, git_remote_url)
        VALUES ('proj-local', ?1, 'projects', 'local_abc123', 1, 0, NULL)
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .execute(&pool)
    .await
    .unwrap();

    let result = check_git_state_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.projects_checked, 1);
    assert_eq!(result.transitions_detected, 0);
}

#[tokio::test]
async fn test_git_state_check_local_to_local_git() {
    // Transition 1: local → local-git (git init, no remote)
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    let temp = TempDir::new().unwrap();
    // Create git repo WITHOUT remote
    Repository::init(temp.path()).expect("Failed to init git repo");

    // Insert as local project (stored: is_git_tracked=0)
    let calculator = ProjectIdCalculator::new();
    let local_tid = calculator.calculate(temp.path(), None, None);

    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            is_git_tracked, git_remote_url)
        VALUES ('proj-1', ?1, 'projects', ?2, 1, 0, NULL)
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .bind(&local_tid)
    .execute(&pool)
    .await
    .unwrap();

    let result = check_git_state_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.projects_checked, 1);
    assert_eq!(result.transitions_detected, 1);

    // Verify is_git_tracked updated to 1
    let is_git: i32 = sqlx::query_scalar(
        "SELECT is_git_tracked FROM watch_folders WHERE watch_id = 'proj-1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(is_git, 1);

    // tenant_id should NOT change (still local_ prefix, same path)
    let tid: String = sqlx::query_scalar(
        "SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(tid, local_tid, "Transition 1 should not change tenant_id");
}

#[tokio::test]
async fn test_git_state_check_local_git_to_remote_git() {
    // Transition 3: local-git → remote-git (remote add)
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    let temp = TempDir::new().unwrap();
    // Create git repo WITH remote
    create_git_repo_with_remote(
        temp.path(),
        "https://github.com/user/repo.git",
    );

    // Insert as local-git (stored: is_git_tracked=1, no remote)
    let calculator = ProjectIdCalculator::new();
    let local_tid = calculator.calculate(temp.path(), None, None);

    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            is_git_tracked, git_remote_url)
        VALUES ('proj-2', ?1, 'projects', ?2, 1, 1, NULL)
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .bind(&local_tid)
    .execute(&pool)
    .await
    .unwrap();

    let result = check_git_state_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.projects_checked, 1);
    assert_eq!(result.transitions_detected, 1);

    // Verify remote was stored
    let remote: Option<String> = sqlx::query_scalar(
        "SELECT git_remote_url FROM watch_folders WHERE watch_id = 'proj-2'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(remote, Some("https://github.com/user/repo.git".to_string()));

    // tenant_id should have changed from local_ to remote-based
    let tid: String = sqlx::query_scalar(
        "SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-2'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_ne!(tid, local_tid, "Transition 3 should change tenant_id");
    assert!(!tid.starts_with("local_"), "Should be remote-based tenant_id");

    // Verify cascade rename was enqueued
    let rename_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'rename'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(rename_count >= 1, "Should have enqueued cascade rename");
}

#[tokio::test]
async fn test_git_state_check_git_to_local() {
    // Transition 4: git → local (rm -rf .git simulated by non-git temp dir)
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Plain temp dir (no .git) — simulates .git removal
    let temp = TempDir::new().unwrap();

    let calculator = ProjectIdCalculator::new();
    let remote_tid = calculator.calculate(
        temp.path(),
        Some("https://github.com/user/repo.git"),
        None,
    );

    // Insert as remote-git project
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            is_git_tracked, git_remote_url, remote_hash)
        VALUES ('proj-3', ?1, 'projects', ?2, 1, 1,
            'https://github.com/user/repo.git', 'somehash')
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .bind(&remote_tid)
    .execute(&pool)
    .await
    .unwrap();

    let result = check_git_state_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.projects_checked, 1);
    assert_eq!(result.transitions_detected, 1);

    // Verify is_git_tracked set to 0
    let is_git: i32 = sqlx::query_scalar(
        "SELECT is_git_tracked FROM watch_folders WHERE watch_id = 'proj-3'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(is_git, 0);

    // Verify git_remote_url cleared
    let remote: Option<String> = sqlx::query_scalar(
        "SELECT git_remote_url FROM watch_folders WHERE watch_id = 'proj-3'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(remote.is_none(), "Remote URL should be cleared");

    // tenant_id should have changed to local_
    let tid: String = sqlx::query_scalar(
        "SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-3'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(tid.starts_with("local_"), "Should be local-based tenant_id after .git removal");

    // Verify cascade rename was enqueued
    let rename_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'rename'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(rename_count >= 1);
}

#[tokio::test]
async fn test_git_state_check_skips_inactive() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    let temp = TempDir::new().unwrap();

    // Insert as inactive project
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            is_git_tracked)
        VALUES ('proj-inactive', ?1, 'projects', 'local_abc', 0, 0)
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .execute(&pool)
    .await
    .unwrap();

    let result = check_git_state_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.projects_checked, 0);
    assert_eq!(result.transitions_detected, 0);
}
