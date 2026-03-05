use super::super::*;
use super::create_test_pool;

#[tokio::test]
async fn test_run_migrations_from_scratch() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    let version = manager
        .get_current_version()
        .await
        .expect("Failed to get version");
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

    let watch_folders_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='watch_folders')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        watch_folders_exists,
        "watch_folders table should exist after migration"
    );

    let unified_queue_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='unified_queue')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        unified_queue_exists,
        "unified_queue table should exist after migration"
    );

    let tracked_files_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        tracked_files_exists,
        "tracked_files table should exist after migration v2"
    );

    let qdrant_chunks_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='qdrant_chunks')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        qdrant_chunks_exists,
        "qdrant_chunks table should exist after migration v2"
    );
}

#[tokio::test]
async fn test_run_migrations_idempotent() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool);

    manager
        .run_migrations()
        .await
        .expect("First migration failed");
    manager
        .run_migrations()
        .await
        .expect("Second migration should be idempotent");
}

#[tokio::test]
async fn test_incremental_migration_v1_to_current() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    manager.run_migration(1).await.expect("Failed to run v1");
    manager
        .record_migration(1)
        .await
        .expect("Failed to record v1");

    let tracked_exists_before: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !tracked_exists_before,
        "tracked_files should NOT exist before v2 migration"
    );

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations from v1");

    let tracked_exists_after: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        tracked_exists_after,
        "tracked_files should exist after v2 migration"
    );

    let has_reconcile: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'needs_reconcile'"
    )
    .fetch_one(&pool).await.unwrap();
    assert!(
        has_reconcile,
        "needs_reconcile column should exist after v3 migration"
    );

    let version = manager
        .get_current_version()
        .await
        .expect("Failed to get version");
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));
}

#[tokio::test]
async fn test_incremental_migration_v2_to_v3() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    manager.run_migration(1).await.expect("Failed to run v1");
    manager
        .record_migration(1)
        .await
        .expect("Failed to record v1");
    manager.run_migration(2).await.expect("Failed to run v2");
    manager
        .record_migration(2)
        .await
        .expect("Failed to record v2");

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations from v2");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

    let has_index: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_reconcile'"
    )
    .fetch_one(&pool).await.unwrap();
    assert!(has_index, "Reconcile index should exist after v3 migration");
}

#[tokio::test]
async fn test_incremental_migration_v3_to_v4() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    for v in 1..=3 {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
    }

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations from v3 to v4");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

    let has_is_paused: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_paused'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        has_is_paused,
        "is_paused column should exist after v4 migration"
    );

    let has_pause_start_time: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'pause_start_time'"
    )
    .fetch_one(&pool).await.unwrap();
    assert!(
        has_pause_start_time,
        "pause_start_time column should exist after v4 migration"
    );
}

#[tokio::test]
async fn test_incremental_migration_v4_to_current() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    for v in 1..=4 {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
    }

    let exists_before: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !exists_before,
        "metrics_history should NOT exist before v5 migration"
    );

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations from v4 to current");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION as i32));

    let exists_after: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        exists_after,
        "metrics_history table should exist after v5 migration"
    );

    let idx_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_metrics_%'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(idx_count, 3, "Should have 3 metrics_history indexes");

    let has_collection: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'collection'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        has_collection,
        "collection column should exist after v6 migration"
    );
}
