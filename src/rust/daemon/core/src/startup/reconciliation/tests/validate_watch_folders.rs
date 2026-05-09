//! Tests for `validate_watch_folders`.

use super::super::validate_watch_folders;
use super::{create_test_pool, setup_schema};

#[tokio::test]
async fn test_validate_watch_folders_deactivates_invalid() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
         created_at, updated_at) \
         VALUES ('w_gone', '/tmp/nonexistent-path-task-512-test', 'projects', 't1', 1, 1, \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 1);
    assert_eq!(stats.folders_deactivated, 1);
    assert_eq!(stats.folders_valid, 0);

    let is_active: bool =
        sqlx::query_scalar("SELECT is_active FROM watch_folders WHERE watch_id = 'w_gone'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(!is_active, "Watch folder should be deactivated");
}

#[tokio::test]
async fn test_validate_watch_folders_keeps_valid() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let watch_path = tmp_dir.path().to_str().unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
         created_at, updated_at) \
         VALUES ('w_valid', ?1, 'projects', 't1', 1, 1, \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(watch_path)
    .execute(&pool)
    .await
    .unwrap();

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 1);
    assert_eq!(stats.folders_deactivated, 0);
    assert_eq!(stats.folders_valid, 1);

    let is_active: bool =
        sqlx::query_scalar("SELECT is_active FROM watch_folders WHERE watch_id = 'w_valid'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert!(is_active, "Watch folder should remain active");
}

#[tokio::test]
async fn test_validate_watch_folders_mixed() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    let tmp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let valid_path = tmp_dir.path().to_str().unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
         created_at, updated_at) \
         VALUES ('w_ok', ?1, 'projects', 't1', 1, 1, \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .bind(valid_path)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, enabled, \
         created_at, updated_at) \
         VALUES ('w_bad', '/tmp/definitely-does-not-exist-512', 'libraries', 'lib1', 0, 1, \
         '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(&pool)
    .await
    .unwrap();

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 2);
    assert_eq!(stats.folders_deactivated, 1);
    assert_eq!(stats.folders_valid, 1);
}

#[tokio::test]
async fn test_validate_watch_folders_empty() {
    let pool = create_test_pool().await;
    setup_schema(&pool).await;

    let stats = validate_watch_folders(&pool)
        .await
        .expect("validate_watch_folders failed");

    assert_eq!(stats.folders_checked, 0);
    assert_eq!(stats.folders_deactivated, 0);
    assert_eq!(stats.folders_valid, 0);
}
