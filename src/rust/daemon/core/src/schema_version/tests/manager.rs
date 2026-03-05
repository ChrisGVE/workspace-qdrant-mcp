use super::super::*;
use super::create_test_pool;

#[tokio::test]
async fn test_initialize_creates_table() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert!(
        exists,
        "schema_version table should exist after initialization"
    );
}

#[tokio::test]
async fn test_get_version_empty_db() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool);

    let version = manager
        .get_current_version()
        .await
        .expect("Failed to get version");
    assert_eq!(version, None, "Version should be None for fresh database");
}

#[tokio::test]
async fn test_record_and_get_version() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool);

    manager.initialize().await.expect("Failed to initialize");
    manager
        .record_migration(1)
        .await
        .expect("Failed to record migration");

    let version = manager
        .get_current_version()
        .await
        .expect("Failed to get version");
    assert_eq!(version, Some(1), "Version should be 1 after recording");
}

#[tokio::test]
async fn test_migration_history() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool);

    manager.initialize().await.expect("Failed to initialize");
    manager
        .record_migration(1)
        .await
        .expect("Failed to record v1");
    manager
        .record_migration(2)
        .await
        .expect("Failed to record v2");

    let history = manager
        .get_migration_history()
        .await
        .expect("Failed to get history");
    assert_eq!(history.len(), 2, "Should have 2 migration entries");
    assert_eq!(history[0].version, 1);
    assert_eq!(history[1].version, 2);
}

#[tokio::test]
async fn test_is_schema_initialized() {
    let pool = create_test_pool().await;

    assert!(
        !is_schema_initialized(&pool).await,
        "Should not be initialized yet"
    );

    let manager = SchemaManager::new(pool.clone());
    manager.initialize().await.expect("Failed to initialize");

    assert!(is_schema_initialized(&pool).await, "Should be initialized");
}

#[tokio::test]
async fn test_get_schema_version_helper() {
    let pool = create_test_pool().await;

    let version = get_schema_version(&pool).await;
    assert_eq!(version, None);

    let manager = SchemaManager::new(pool.clone());
    manager.initialize().await.expect("Failed to initialize");
    manager.record_migration(3).await.expect("Failed to record");

    let version = get_schema_version(&pool).await;
    assert_eq!(version, Some(3));
}

#[tokio::test]
async fn test_downgrade_not_supported() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool);

    manager.initialize().await.expect("Failed to initialize");
    manager
        .record_migration(CURRENT_SCHEMA_VERSION + 10)
        .await
        .expect("Failed to record future version");

    let result = manager.run_migrations().await;
    assert!(result.is_err());

    match result.unwrap_err() {
        SchemaError::DowngradeNotSupported {
            db_version,
            code_version,
        } => {
            assert_eq!(db_version, CURRENT_SCHEMA_VERSION + 10);
            assert_eq!(code_version, CURRENT_SCHEMA_VERSION);
        }
        other => panic!("Expected DowngradeNotSupported error, got: {:?}", other),
    }
}
