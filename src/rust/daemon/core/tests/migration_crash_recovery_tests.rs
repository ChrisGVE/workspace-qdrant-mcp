//! Integration tests for the v37 relative-path migration crash-recovery protocol.
//!
//! Each test simulates a crash at a specific phase boundary by stopping the
//! migration sequence mid-way, then verifying that the marker table allows the
//! daemon to detect the incomplete state and resume correctly.

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Executor, SqlitePool};
use std::time::Duration;

use workspace_qdrant_core::schema_version::v37::{
    finalize_relative_path_migration, get_relative_path_migration_status,
    is_relative_path_migration_in_progress, mark_initial_walk_complete,
    CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL,
};
use workspace_qdrant_core::{SchemaManager, CURRENT_SCHEMA_VERSION};

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn migrate_to(pool: &SqlitePool, target_version: i32) {
    let manager = SchemaManager::new(pool.clone());
    manager.initialize().await.expect("init");
    for v in 1..=target_version {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("migration v{v} failed: {e}"));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("record v{v} failed: {e}"));
    }
}

async fn seed_tracked_files(pool: &SqlitePool) {
    let now = "2026-01-01T00:00:00.000Z";
    sqlx::query(
        "INSERT INTO watch_folders \
             (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('wf-test', '/tmp/test', 'projects', 'tenant-cr', ?1, ?1)",
    )
    .bind(now)
    .execute(pool)
    .await
    .expect("insert watch_folder");

    sqlx::query(
        "INSERT INTO tracked_files \
             (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('wf-test', '/tmp/test/a.rs', ?1, 'hash_a', ?1, ?1)",
    )
    .bind(now)
    .execute(pool)
    .await
    .expect("insert tracked_file");
}

async fn count_rows(pool: &SqlitePool, table: &str) -> i64 {
    let exists: bool = sqlx::query_scalar(&format!(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}')"
    ))
    .fetch_one(pool)
    .await
    .unwrap();
    if !exists {
        return 0;
    }
    sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(pool)
        .await
        .unwrap()
}

/// Crash after Phase 1: marker row inserted, but tables not yet truncated.
#[tokio::test]
async fn crash_recovery_after_phase_1() {
    let pool = create_test_pool().await;
    migrate_to(&pool, 36).await;
    seed_tracked_files(&pool).await;

    // Phase 1 only: create marker table + insert sentinel row.
    pool.execute(CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL)
        .await
        .expect("create marker table");
    let now_unix: i64 = 1700000000;
    sqlx::query(
        "INSERT OR REPLACE INTO relative_path_migration_in_progress \
         (target_version, started_at, initial_walk_complete, initial_pending_count) \
         VALUES (37, ?1, 0, NULL)",
    )
    .bind(now_unix)
    .execute(&pool)
    .await
    .expect("insert marker");

    // "Restart": verify marker exists.
    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());

    let status = get_relative_path_migration_status(&pool)
        .await
        .unwrap()
        .expect("status must be Some");
    assert!(!status.initial_walk_complete);
    assert!(status.initial_pending_count.is_none());

    // tracked_files should still have the seeded row (phase 2 not done).
    assert_eq!(count_rows(&pool, "tracked_files").await, 1);

    // Re-running truncation (phase 2) is idempotent.
    pool.execute("DELETE FROM tracked_files")
        .await
        .expect("truncate tracked_files");
    assert_eq!(count_rows(&pool, "tracked_files").await, 0);

    // Complete remaining phases.
    mark_initial_walk_complete(&pool, 0).await.unwrap();
    finalize_relative_path_migration(&pool).await.unwrap();
    assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());
}

/// Crash after Phase 2: full V37Migration::up() ran but daemon crashed
/// before phase 2b/3/4.
#[tokio::test]
async fn crash_recovery_after_phase_2() {
    let pool = create_test_pool().await;
    migrate_to(&pool, 36).await;
    seed_tracked_files(&pool).await;

    let pre_count = count_rows(&pool, "tracked_files").await;
    assert!(pre_count > 0, "must have data before v37");

    let manager = SchemaManager::new(pool.clone());
    manager.run_migration(37).await.expect("v37 migration");
    manager.record_migration(37).await.expect("record v37");

    // "Restart": marker must exist.
    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());

    // Tables must already be truncated.
    assert_eq!(count_rows(&pool, "tracked_files").await, 0);

    // Re-running truncation is idempotent.
    pool.execute("DELETE FROM tracked_files")
        .await
        .expect("re-truncate is idempotent");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(37));
}

/// Crash during Phase 3: partial walk items in queue, then crash.
#[tokio::test]
async fn crash_recovery_during_phase_3() {
    let pool = create_test_pool().await;
    migrate_to(&pool, 37).await;

    // Simulate partial re-ingest.
    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-cr', 'projects', 'idem-cr-1', 'main', 'src/b.rs')",
    )
    .execute(&pool)
    .await
    .expect("insert queue item");

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-cr', 'projects', 'idem-cr-2', 'main', 'src/c.rs')",
    )
    .execute(&pool)
    .await
    .expect("insert queue item 2");

    // "Restart": marker must still be present.
    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());

    let status = get_relative_path_migration_status(&pool)
        .await
        .unwrap()
        .expect("status must be Some");
    assert!(!status.initial_walk_complete);

    // Queue items from the partial walk must survive.
    let queue_count = count_rows(&pool, "unified_queue").await;
    assert_eq!(queue_count, 2);

    // Complete the walk and finalize.
    mark_initial_walk_complete(&pool, queue_count)
        .await
        .unwrap();

    let updated = get_relative_path_migration_status(&pool)
        .await
        .unwrap()
        .expect("status after walk complete");
    assert!(updated.initial_walk_complete);
    assert_eq!(updated.initial_pending_count, Some(2));

    finalize_relative_path_migration(&pool).await.unwrap();
    assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());
}

/// Crash before Phase 4: walk complete but finalize not called.
#[tokio::test]
async fn crash_recovery_before_phase_4() {
    let pool = create_test_pool().await;
    migrate_to(&pool, 37).await;

    mark_initial_walk_complete(&pool, 42).await.unwrap();

    // "Restart": marker must exist with correct state.
    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());

    let status = get_relative_path_migration_status(&pool)
        .await
        .unwrap()
        .expect("status must be Some");
    assert!(status.initial_walk_complete);
    assert_eq!(status.initial_pending_count, Some(42));

    finalize_relative_path_migration(&pool).await.unwrap();
    assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());
}

/// Happy path: all phases in sequence, marker deleted at end.
#[tokio::test]
async fn crash_recovery_full_happy_path() {
    let pool = create_test_pool().await;
    migrate_to(&pool, 36).await;
    seed_tracked_files(&pool).await;

    let manager = SchemaManager::new(pool.clone());
    manager.run_migration(37).await.expect("v37");
    manager.record_migration(37).await.expect("record v37");

    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());
    assert_eq!(count_rows(&pool, "tracked_files").await, 0);

    mark_initial_walk_complete(&pool, 10).await.unwrap();

    let status = get_relative_path_migration_status(&pool)
        .await
        .unwrap()
        .expect("status");
    assert!(status.initial_walk_complete);
    assert_eq!(status.initial_pending_count, Some(10));

    finalize_relative_path_migration(&pool).await.unwrap();
    assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());
    assert!(get_relative_path_migration_status(&pool)
        .await
        .unwrap()
        .is_none());

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(37));
}

/// Calling finalize twice is a no-op.
#[tokio::test]
async fn crash_recovery_idempotent_finalize() {
    let pool = create_test_pool().await;
    migrate_to(&pool, 37).await;

    finalize_relative_path_migration(&pool).await.unwrap();
    assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());

    finalize_relative_path_migration(&pool).await.unwrap();
    assert!(!is_relative_path_migration_in_progress(&pool).await.unwrap());
}

/// Full migration chain from scratch reaches v37 with marker present.
#[tokio::test]
async fn crash_recovery_migrations_from_scratch_reach_v37() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.run_migrations().await.expect("full migration");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));
    assert_eq!(CURRENT_SCHEMA_VERSION, 44);

    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type='table' AND name='relative_path_migration_in_progress')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(table_exists);

    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());
}

/// Marker table has the expected column shape.
#[tokio::test]
async fn crash_recovery_marker_table_shape() {
    let pool = create_test_pool().await;
    migrate_to(&pool, 37).await;

    let columns: Vec<String> = sqlx::query_scalar(
        "SELECT name FROM pragma_table_info('relative_path_migration_in_progress') \
         ORDER BY cid",
    )
    .fetch_all(&pool)
    .await
    .unwrap();

    assert_eq!(
        columns,
        vec![
            "target_version".to_string(),
            "started_at".to_string(),
            "initial_walk_complete".to_string(),
            "initial_pending_count".to_string(),
        ]
    );
}
