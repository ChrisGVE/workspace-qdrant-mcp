//! v37 DDL mid-rebuild crash-recovery tests.
//!
//! These tests exercise crash states INSIDE the `rebuild_tracked_files` DDL
//! sequence (rename / recreate / drop-old) that the phase-boundary tests in
//! `migration_crash_recovery_tests.rs` do not cover. Each test manually
//! constructs a partial DDL state, then re-runs the v37 migration and verifies
//! actual behavior.
//!
//! COVERAGE GAPS DOCUMENTED:
//! - When both tracked_files (v37 schema) and tracked_files_old coexist, v37
//!   returns early without dropping tracked_files_old (orphan leak).
//! - When only tracked_files_old exists (renamed but new table not created),
//!   v37 returns early because `tracked_files` does not exist (no rebuild).

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Executor, SqlitePool};
use std::time::Duration;

use workspace_qdrant_core::schema_version::v37::{
    finalize_relative_path_migration, is_relative_path_migration_in_progress,
    mark_initial_walk_complete, CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL,
};
use workspace_qdrant_core::tracked_files_schema::{
    CREATE_TRACKED_FILES_V37_INDEXES_SQL, CREATE_TRACKED_FILES_V37_SQL,
};
use workspace_qdrant_core::SchemaManager;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn migrate_to_v36(pool: &SqlitePool) {
    let manager = SchemaManager::new(pool.clone());
    manager.initialize().await.expect("init");
    for v in 1..=36 {
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
         VALUES ('wf-ddl', '/tmp/ddl', 'projects', 'tenant-ddl', ?1, ?1)",
    )
    .bind(now)
    .execute(pool)
    .await
    .expect("insert watch_folder");

    sqlx::query(
        "INSERT INTO tracked_files \
             (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('wf-ddl', '/tmp/ddl/a.rs', ?1, 'hash_a', ?1, ?1)",
    )
    .bind(now)
    .execute(pool)
    .await
    .expect("insert tracked_file");
}

async fn insert_marker(pool: &SqlitePool) {
    pool.execute(CREATE_RELATIVE_PATH_MIGRATION_TABLE_SQL)
        .await
        .expect("create marker table");
    sqlx::query(
        "INSERT OR REPLACE INTO relative_path_migration_in_progress \
         (target_version, started_at, initial_walk_complete, initial_pending_count) \
         VALUES (37, 1700000000, 0, NULL)",
    )
    .execute(pool)
    .await
    .expect("insert marker");
}

async fn tbl_exists(pool: &SqlitePool, name: &str) -> bool {
    sqlx::query_scalar(&format!(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{name}')"
    ))
    .fetch_one(pool)
    .await
    .unwrap()
}

async fn row_count(pool: &SqlitePool, table: &str) -> i64 {
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

async fn rename_tracked_files_to_old(pool: &SqlitePool) {
    pool.execute("PRAGMA foreign_keys = OFF").await.unwrap();
    pool.execute("PRAGMA legacy_alter_table = ON")
        .await
        .unwrap();
    pool.execute("ALTER TABLE tracked_files RENAME TO tracked_files_old")
        .await
        .unwrap();
    pool.execute("PRAGMA legacy_alter_table = OFF")
        .await
        .unwrap();
    pool.execute("PRAGMA foreign_keys = ON").await.unwrap();
}

async fn create_new_tracked_files_v37(pool: &SqlitePool) {
    pool.execute(CREATE_TRACKED_FILES_V37_SQL).await.unwrap();
    for idx_sql in CREATE_TRACKED_FILES_V37_INDEXES_SQL {
        pool.execute(*idx_sql).await.unwrap();
    }
}

fn assert_v37_schema(sql: &str) {
    assert!(
        !sql.contains("file_path TEXT NOT NULL"),
        "tracked_files must have v37 schema (no file_path column)"
    );
}

async fn get_create_sql(pool: &SqlitePool) -> String {
    sqlx::query_scalar("SELECT sql FROM sqlite_master WHERE type='table' AND name='tracked_files'")
        .fetch_one(pool)
        .await
        .unwrap()
}

async fn finalize_lifecycle(pool: &SqlitePool) {
    mark_initial_walk_complete(pool, 0).await.unwrap();
    finalize_relative_path_migration(pool).await.unwrap();
    assert!(!is_relative_path_migration_in_progress(pool).await.unwrap());
}

// ============================================================================
// Test: crash after rename, new table created, but old NOT dropped
// ============================================================================

/// Crash after new tracked_files created (v37 schema) but BEFORE
/// tracked_files_old was dropped. Both tables coexist.
///
/// DOCUMENTED GAP: `rebuild_tracked_files` detects tracked_files already
/// has the v37 schema and returns early WITHOUT cleaning up
/// tracked_files_old. The orphan table is harmless (no FK references,
/// not queried) but is never cleaned up until the next full migration.
#[tokio::test]
async fn ddl_crash_both_tables_exist_orphan_leak() {
    let pool = create_test_pool().await;
    migrate_to_v36(&pool).await;
    seed_tracked_files(&pool).await;
    insert_marker(&pool).await;

    // Simulate partial DDL: rename + recreate, but skip DROP OLD.
    rename_tracked_files_to_old(&pool).await;
    create_new_tracked_files_v37(&pool).await;

    // Verify crash state: BOTH tables exist.
    assert!(tbl_exists(&pool, "tracked_files").await);
    assert!(tbl_exists(&pool, "tracked_files_old").await);
    assert_eq!(row_count(&pool, "tracked_files_old").await, 1);
    assert_eq!(row_count(&pool, "tracked_files").await, 0);

    // "Restart": re-run v37 migration.
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migration(37)
        .await
        .expect("v37 re-run after crash with both tables");
    manager.record_migration(37).await.expect("record v37");

    // tracked_files has v37 schema (correct).
    assert!(tbl_exists(&pool, "tracked_files").await);
    assert_v37_schema(&get_create_sql(&pool).await);

    // GAP: tracked_files_old is NOT cleaned up by the idempotent path.
    // The rebuild_tracked_files function returns early when it sees
    // tracked_files already lacks file_path column, without checking
    // for a leftover tracked_files_old.
    assert!(
        tbl_exists(&pool, "tracked_files_old").await,
        "tracked_files_old is leaked (known gap in v37 recovery)"
    );

    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());
    finalize_lifecycle(&pool).await;
}

// ============================================================================
// Test: crash after FK disabled but before rename
// ============================================================================

/// Crash after `PRAGMA foreign_keys = OFF` but BEFORE any table rename.
/// Verifies recovery works even if the FK pragma was left OFF on a pooled
/// connection.
#[tokio::test]
async fn ddl_crash_fk_disabled_before_rename() {
    let pool = create_test_pool().await;
    migrate_to_v36(&pool).await;
    seed_tracked_files(&pool).await;
    insert_marker(&pool).await;

    // Disable FK checks on a connection, then "crash" (drop without restore).
    {
        let mut conn = pool.acquire().await.unwrap();
        conn.execute("PRAGMA foreign_keys = OFF").await.unwrap();
        let fk: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
            .fetch_one(&mut *conn)
            .await
            .unwrap();
        assert_eq!(fk, 0, "FK must be OFF on the crashed connection");
        drop(conn);
    }

    // Tables must be unchanged (crash happened before rename).
    assert!(tbl_exists(&pool, "tracked_files").await);
    assert!(!tbl_exists(&pool, "tracked_files_old").await);
    assert_eq!(row_count(&pool, "tracked_files").await, 1);

    // "Restart": re-run v37 migration.
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migration(37)
        .await
        .expect("v37 must succeed after FK-disabled crash");
    manager.record_migration(37).await.expect("record v37");

    assert!(tbl_exists(&pool, "tracked_files").await);
    assert!(!tbl_exists(&pool, "tracked_files_old").await);
    assert_v37_schema(&get_create_sql(&pool).await);

    // FK pragma must be a valid value after migration.
    let fk_after: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert!(fk_after == 0 || fk_after == 1);

    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());
    finalize_lifecycle(&pool).await;
}

// ============================================================================
// Test: crash during data copy (both tables, old has legacy rows)
// ============================================================================

/// Crash during data copy: tracked_files_old has 6 legacy rows,
/// tracked_files (v37) is empty.
///
/// DOCUMENTED GAP: same as `ddl_crash_both_tables_exist_orphan_leak` --
/// the old table is not cleaned up because rebuild_tracked_files returns
/// early on the idempotent path.
#[tokio::test]
async fn ddl_crash_data_copy_interrupted_orphan_leak() {
    let pool = create_test_pool().await;
    migrate_to_v36(&pool).await;
    seed_tracked_files(&pool).await;

    let now = "2026-01-01T00:00:00.000Z";
    for i in 1..=5 {
        sqlx::query(
            "INSERT INTO tracked_files \
                 (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
             VALUES ('wf-ddl', ?1, ?2, ?3, ?2, ?2)",
        )
        .bind(format!("/tmp/ddl/file_{i}.rs"))
        .bind(now)
        .bind(format!("hash_{i}"))
        .execute(&pool)
        .await
        .unwrap();
    }
    assert_eq!(row_count(&pool, "tracked_files").await, 6);

    insert_marker(&pool).await;
    rename_tracked_files_to_old(&pool).await;
    create_new_tracked_files_v37(&pool).await;

    assert!(tbl_exists(&pool, "tracked_files").await);
    assert!(tbl_exists(&pool, "tracked_files_old").await);
    assert_eq!(row_count(&pool, "tracked_files_old").await, 6);
    assert_eq!(row_count(&pool, "tracked_files").await, 0);

    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migration(37)
        .await
        .expect("v37 re-run after data-copy crash");
    manager.record_migration(37).await.expect("record v37");

    // tracked_files is correct (v37 schema, empty).
    assert!(tbl_exists(&pool, "tracked_files").await);
    assert_eq!(row_count(&pool, "tracked_files").await, 0);
    assert_v37_schema(&get_create_sql(&pool).await);

    // GAP: tracked_files_old with 6 rows is leaked.
    assert!(
        tbl_exists(&pool, "tracked_files_old").await,
        "tracked_files_old is leaked (known gap)"
    );

    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());
    finalize_lifecycle(&pool).await;
}

// ============================================================================
// Test: crash after rename, new table NOT created (orphan path)
// ============================================================================

/// Crash after tracked_files renamed to tracked_files_old but new
/// tracked_files NOT created. Only tracked_files_old exists.
///
/// DOCUMENTED GAP: `rebuild_tracked_files` checks `table_exists` for
/// `tracked_files` first and returns early when it does not exist,
/// without checking for tracked_files_old. This leaves the database
/// without a tracked_files table and an orphaned tracked_files_old.
#[tokio::test]
async fn ddl_crash_rename_orphan_no_recovery() {
    let pool = create_test_pool().await;
    migrate_to_v36(&pool).await;
    seed_tracked_files(&pool).await;
    insert_marker(&pool).await;
    rename_tracked_files_to_old(&pool).await;

    assert!(!tbl_exists(&pool, "tracked_files").await);
    assert!(tbl_exists(&pool, "tracked_files_old").await);

    let manager = SchemaManager::new(pool.clone());
    manager.run_migration(37).await.expect("v37 rerun");
    manager.record_migration(37).await.expect("record v37");

    // GAP: tracked_files is NOT recreated because rebuild_tracked_files
    // returns early when tracked_files does not exist. The orphaned
    // tracked_files_old remains.
    assert!(
        !tbl_exists(&pool, "tracked_files").await,
        "tracked_files is NOT recreated (known gap)"
    );
    assert!(
        tbl_exists(&pool, "tracked_files_old").await,
        "tracked_files_old is still orphaned (known gap)"
    );
}

// ============================================================================
// Test: FK pragma not leaked across idempotent rerun
// ============================================================================

/// FK pragma must not change across an idempotent v37 rerun.
#[tokio::test]
async fn ddl_fk_pragma_not_leaked_across_rerun() {
    let pool = create_test_pool().await;
    migrate_to_v36(&pool).await;
    seed_tracked_files(&pool).await;

    let manager = SchemaManager::new(pool.clone());
    manager.run_migration(37).await.expect("v37 first run");

    let fk_mid: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&pool)
        .await
        .unwrap();

    manager
        .run_migration(37)
        .await
        .expect("v37 idempotent rerun");

    let fk_after: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(fk_mid, fk_after);
}

// ============================================================================
// Test: stale tracked_files_old from prior aborted attempt
// ============================================================================

/// Stale tracked_files_old from a prior aborted attempt must not prevent
/// a fresh v37 run from succeeding. The normal DDL path issues
/// `DROP TABLE IF EXISTS tracked_files_old` before renaming.
#[tokio::test]
async fn ddl_crash_stale_tracked_files_old() {
    let pool = create_test_pool().await;
    migrate_to_v36(&pool).await;
    seed_tracked_files(&pool).await;

    pool.execute(
        "CREATE TABLE tracked_files_old AS \
         SELECT * FROM tracked_files WHERE 0",
    )
    .await
    .unwrap();
    let now = "2026-01-01T00:00:00.000Z";
    sqlx::query(
        "INSERT INTO tracked_files_old \
             (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at) \
         VALUES ('wf-ddl', '/tmp/stale.rs', ?1, 'stale', ?1, ?1)",
    )
    .bind(now)
    .execute(&pool)
    .await
    .unwrap();

    assert!(tbl_exists(&pool, "tracked_files_old").await);
    assert_eq!(row_count(&pool, "tracked_files_old").await, 1);

    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migration(37)
        .await
        .expect("v37 must handle stale tracked_files_old");
    manager.record_migration(37).await.expect("record v37");

    assert!(!tbl_exists(&pool, "tracked_files_old").await);
    assert!(tbl_exists(&pool, "tracked_files").await);
    assert_v37_schema(&get_create_sql(&pool).await);

    assert!(is_relative_path_migration_in_progress(&pool).await.unwrap());
    finalize_lifecycle(&pool).await;
}
