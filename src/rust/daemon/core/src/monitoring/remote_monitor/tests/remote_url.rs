use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_get_git_remote_url() {
    let temp = TempDir::new().unwrap();
    create_git_repo_with_remote(temp.path(), "https://github.com/user/repo.git");

    let url = get_git_remote_url(temp.path().to_str().unwrap()).unwrap();
    assert_eq!(url, "https://github.com/user/repo.git");
}

#[tokio::test]
async fn test_get_git_remote_url_not_git() {
    let temp = TempDir::new().unwrap();
    let result = get_git_remote_url(temp.path().to_str().unwrap());
    assert!(result.is_err());
}

#[tokio::test]
async fn test_update_watch_folders_remote() {
    let pool = create_test_database().await;

    // Insert parent project
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            git_remote_url, remote_hash, disambiguation_path)
        VALUES ('proj-1', '/tmp/repo', 'projects', 'old_tenant', 1,
            'https://github.com/old/repo.git', 'oldhash12345', NULL)
        "#,
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert submodule
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            parent_watch_id, git_remote_url, remote_hash)
        VALUES ('sub-1', '/tmp/repo/lib', 'projects', 'old_tenant', 1,
            'proj-1', 'https://github.com/lib/sub.git', 'subhash12345')
        "#,
    )
    .execute(&pool)
    .await
    .unwrap();

    // Task 14: Insert junction table row
    sqlx::query(
        "INSERT INTO watch_folder_submodules (parent_watch_id, child_watch_id, submodule_path, created_at) VALUES ('proj-1', 'sub-1', 'lib', datetime('now'))"
    )
    .execute(&pool)
    .await
    .unwrap();

    // Execute remote update
    update_watch_folders_remote(
        &pool,
        "proj-1",
        "https://github.com/new/repo.git",
        "newhash12345",
        "new_tenant",
    )
    .await
    .unwrap();

    // Verify parent updated
    let (tid, url, hash): (String, String, String) = sqlx::query_as(
        "SELECT tenant_id, git_remote_url, remote_hash FROM watch_folders WHERE watch_id = 'proj-1'"
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(tid, "new_tenant");
    assert_eq!(url, "https://github.com/new/repo.git");
    assert_eq!(hash, "newhash12345");

    // Verify submodule tenant_id updated (but git_remote_url unchanged)
    let (sub_tid, sub_url): (String, String) = sqlx::query_as(
        "SELECT tenant_id, git_remote_url FROM watch_folders WHERE watch_id = 'sub-1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(sub_tid, "new_tenant");
    assert_eq!(sub_url, "https://github.com/lib/sub.git"); // unchanged
}

#[tokio::test]
async fn test_check_remote_url_changes_no_change() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create git repo with known remote
    let temp = TempDir::new().unwrap();
    create_git_repo_with_remote(temp.path(), "https://github.com/user/repo.git");

    // Insert watch_folder with same remote (normalized)
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            git_remote_url, remote_hash)
        VALUES ('proj-1', ?1, 'projects', 'some_tenant', 1,
            'https://github.com/user/repo.git', 'somehash12345')
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .execute(&pool)
    .await
    .unwrap();

    let result = check_remote_url_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.projects_checked, 1);
    assert_eq!(result.changes_detected, 0);
    assert_eq!(result.errors, 0);
}

#[tokio::test]
async fn test_check_remote_url_changes_detects_change() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create git repo with NEW remote
    let temp = TempDir::new().unwrap();
    create_git_repo_with_remote(temp.path(), "https://github.com/new-org/repo.git");

    let calculator = ProjectIdCalculator::new();

    // Insert watch_folder with OLD remote
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            git_remote_url, remote_hash)
        VALUES ('proj-1', ?1, 'projects', 'old_tenant', 1,
            'https://github.com/old-org/repo.git', 'oldhash12345')
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .execute(&pool)
    .await
    .unwrap();

    let result = check_remote_url_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.projects_checked, 1);
    assert_eq!(result.changes_detected, 1);
    assert_eq!(result.errors, 0);

    // Verify watch_folder was updated
    let (tid, url): (String, String) = sqlx::query_as(
        "SELECT tenant_id, git_remote_url FROM watch_folders WHERE watch_id = 'proj-1'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(url, "https://github.com/new-org/repo.git");
    // Verify tenant_id changed
    assert_ne!(tid, "old_tenant");
    // Verify it matches expected calculation
    let expected_tid = calculator.calculate(
        temp.path(),
        Some("https://github.com/new-org/repo.git"),
        None,
    );
    assert_eq!(tid, expected_tid);

    // Verify cascade rename was enqueued
    let count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM unified_queue WHERE item_type = 'tenant' AND op = 'rename'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert!(
        count >= 1,
        "Expected at least 1 cascade rename queue item, got {}",
        count
    );
}

#[tokio::test]
async fn test_check_remote_url_changes_skips_archived() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create git repo with different remote
    let temp = TempDir::new().unwrap();
    create_git_repo_with_remote(temp.path(), "https://github.com/new-org/repo.git");

    // Insert archived watch_folder with old remote
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            is_archived, git_remote_url, remote_hash)
        VALUES ('proj-1', ?1, 'projects', 'old_tenant', 1,
            1, 'https://github.com/old-org/repo.git', 'oldhash')
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .execute(&pool)
    .await
    .unwrap();

    let result = check_remote_url_changes(&pool, &queue_manager)
        .await
        .unwrap();

    // Archived project should not be checked
    assert_eq!(result.projects_checked, 0);
    assert_eq!(result.changes_detected, 0);
}

#[tokio::test]
async fn test_check_remote_url_changes_preserves_disambiguation() {
    let pool = create_test_database().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create git repo with NEW remote
    let temp = TempDir::new().unwrap();
    create_git_repo_with_remote(temp.path(), "https://github.com/new-org/repo.git");

    let calculator = ProjectIdCalculator::new();

    // Insert watch_folder with OLD remote AND disambiguation_path
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
            git_remote_url, remote_hash, disambiguation_path)
        VALUES ('proj-1', ?1, 'projects', 'old_tenant', 1,
            'https://github.com/old-org/repo.git', 'oldhash12345', 'work/clone1')
        "#,
    )
    .bind(temp.path().to_str().unwrap())
    .execute(&pool)
    .await
    .unwrap();

    let result = check_remote_url_changes(&pool, &queue_manager)
        .await
        .unwrap();

    assert_eq!(result.changes_detected, 1);

    // Verify new tenant_id preserves disambiguation_path
    let tid: String =
        sqlx::query_scalar("SELECT tenant_id FROM watch_folders WHERE watch_id = 'proj-1'")
            .fetch_one(&pool)
            .await
            .unwrap();

    let expected_tid = calculator.calculate(
        temp.path(),
        Some("https://github.com/new-org/repo.git"),
        Some("work/clone1"),
    );
    assert_eq!(tid, expected_tid);
}
