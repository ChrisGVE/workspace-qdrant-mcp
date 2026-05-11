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

// ============================================================================
// §7.11 — Migration v35 (active_provider column + reembed op)
// ============================================================================

#[tokio::test]
async fn test_v35_adds_active_provider_column() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    let has_active_provider: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
         WHERE name = 'active_provider'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        has_active_provider,
        "watch_folders.active_provider must exist after v35"
    );

    // Default value: an INSERT with no explicit value yields 'openai_compatible'.
    let now = "2026-01-01T00:00:00.000Z";
    sqlx::query(
        "INSERT INTO watch_folders \
            (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
    )
    .bind("wf-v35-default")
    .bind("/tmp/v35-default")
    .bind("projects")
    .bind("tenant-v35")
    .bind(now)
    .execute(&pool)
    .await
    .expect("Failed to insert watch_folders row");
    let provider: String = sqlx::query_scalar(
        "SELECT active_provider FROM watch_folders WHERE watch_id = 'wf-v35-default'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(provider, "openai_compatible");
}

#[tokio::test]
async fn test_v35_queue_op_accepts_reembed() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key) \
         VALUES ('collection', 'reembed', 't-v35', 'projects', 'idem-v35-1')",
    )
    .execute(&pool)
    .await
    .expect("Inserting op='reembed' should succeed after v35");
}

#[tokio::test]
async fn test_v35_existing_in_progress_rows_survive_migration() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("init");
    for v in 1..=34 {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
    }

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, status) \
         VALUES ('file', 'add', 't-v35', 'projects', 'idem-v35-survive', 'in_progress')",
    )
    .execute(&pool)
    .await
    .expect("Failed to insert pre-v35 in_progress row");

    manager.run_migration(35).await.expect("v35 should run");
    manager.record_migration(35).await.expect("record v35");

    let status: String = sqlx::query_scalar(
        "SELECT status FROM unified_queue WHERE idempotency_key = 'idem-v35-survive'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        status, "in_progress",
        "pre-v35 in_progress row must survive the table rebuild with status preserved"
    );
}

#[tokio::test]
async fn test_v35_ddl_has_no_extra_item_types() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    // The 8 canonical item_type values, in DDL order. No 'library', 'memory',
    // or 'scratchpad' should have leaked in.
    for kind in [
        "'text'",
        "'file'",
        "'url'",
        "'website'",
        "'doc'",
        "'folder'",
        "'tenant'",
        "'collection'",
    ] {
        assert!(
            queue_sql.contains(kind),
            "expected item_type {} in DDL, full DDL: {}",
            kind,
            queue_sql
        );
    }
    for forbidden in ["'library'", "'memory'", "'scratchpad'"] {
        assert!(
            !queue_sql.contains(forbidden),
            "v35 DDL must not contain {}, full DDL: {}",
            forbidden,
            queue_sql
        );
    }
}

// ============================================================================
// §7.12 — Migration v36 (composite partial UNIQUE on file_path)
// ============================================================================

/// v36 rebuilds `unified_queue` so `file_path` is a plain TEXT column
/// (no UNIQUE) and creates a composite partial UNIQUE index. Closes F-009.
#[tokio::test]
async fn test_v36_drops_column_unique_and_creates_composite_index() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !queue_sql.contains("file_path TEXT UNIQUE"),
        "v36 must drop the column UNIQUE on file_path; got: {}",
        queue_sql
    );

    let composite_index: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type='index' AND name='idx_unified_queue_file_path_composite')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        composite_index,
        "v36 must create idx_unified_queue_file_path_composite"
    );
}

/// v36 allows the same `file_path` for two tenants and rejects duplicate
/// inserts within the same `(tenant, branch, collection, item_type, op)`
/// scope. Closes F-009.
#[tokio::test]
async fn test_v36_composite_unique_enforced_after_migration() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-A', 'projects', 'idem-v36-1', 'main', '/data/x.md')",
    )
    .execute(&pool)
    .await
    .expect("first row must insert");

    // Same file_path, different tenant: allowed.
    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-B', 'projects', 'idem-v36-2', 'main', '/data/x.md')",
    )
    .execute(&pool)
    .await
    .expect("cross-tenant insert must succeed");

    // Same (tenant, branch, collection, item_type, op, file_path): rejected.
    let err = sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-A', 'projects', 'idem-v36-3', 'main', '/data/x.md')",
    )
    .execute(&pool)
    .await;
    assert!(
        err.is_err(),
        "duplicate within same scope must violate composite UNIQUE"
    );
}
