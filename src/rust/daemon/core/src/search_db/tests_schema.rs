//! Tests for schema management, initialization, and basic database operations.

#![cfg(test)]

use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_create_search_db() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    let manager = SearchDbManager::new(&db_path).await.unwrap();

    assert!(db_path.exists(), "search.db should be created");
    assert_eq!(manager.path(), db_path);

    manager.close().await;
}

#[tokio::test]
async fn test_wal_mode_enabled() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let mode: String = sqlx::query_scalar("PRAGMA journal_mode")
        .fetch_one(manager.pool())
        .await
        .unwrap();

    assert_eq!(mode.to_lowercase(), "wal");

    manager.close().await;
}

#[tokio::test]
async fn test_foreign_keys_enabled() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let fk: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(manager.pool())
        .await
        .unwrap();

    assert_eq!(fk, 1, "foreign_keys should be enabled");

    manager.close().await;
}

#[tokio::test]
async fn test_schema_version_after_init() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let version = manager.get_schema_version().await.unwrap();
    assert_eq!(version, Some(SEARCH_SCHEMA_VERSION));

    manager.close().await;
}

#[tokio::test]
async fn test_schema_version_table_exists() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_schema_version')",
    )
    .fetch_one(manager.pool())
    .await
    .unwrap();

    assert!(exists, "search_schema_version table should exist");

    manager.close().await;
}

#[tokio::test]
async fn test_idempotent_initialization() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    // First init
    let manager1 = SearchDbManager::new(&db_path).await.unwrap();
    let v1 = manager1.get_schema_version().await.unwrap();
    manager1.close().await;

    // Second init -- should not fail or change version
    let manager2 = SearchDbManager::new(&db_path).await.unwrap();
    let v2 = manager2.get_schema_version().await.unwrap();
    manager2.close().await;

    assert_eq!(v1, v2, "Version should be unchanged after re-init");
}

#[tokio::test]
async fn test_concurrent_reads_during_write() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Create a simple test table
    sqlx::query("CREATE TABLE IF NOT EXISTS test_data (id INTEGER PRIMARY KEY, value TEXT)")
        .execute(manager.pool())
        .await
        .unwrap();

    // Insert some data
    sqlx::query("INSERT INTO test_data (id, value) VALUES (1, 'hello')")
        .execute(manager.pool())
        .await
        .unwrap();

    // Start a write transaction
    let pool = manager.pool().clone();
    let write_handle = tokio::spawn(async move {
        let mut tx = pool.begin().await.unwrap();
        sqlx::query("INSERT INTO test_data (id, value) VALUES (2, 'world')")
            .execute(&mut *tx)
            .await
            .unwrap();
        // Hold the transaction open briefly
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        tx.commit().await.unwrap();
    });

    // Concurrent read should succeed (WAL mode)
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM test_data")
        .fetch_one(manager.pool())
        .await
        .unwrap();

    // Should see at least the first row (WAL snapshot isolation)
    assert!(count >= 1, "Should read at least 1 row concurrently");

    write_handle.await.unwrap();
    manager.close().await;
}

#[tokio::test]
async fn test_schema_version_is_current() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let version = manager.get_schema_version().await.unwrap();
    assert_eq!(
        version,
        Some(SEARCH_SCHEMA_VERSION),
        "Schema version should be current"
    );

    manager.close().await;
}

// ============================================================================
// Branch-lineage F3: search.db v8 file_metadata.state column
// ============================================================================

/// Build a search.db pinned at schema version 7 by running migrations 1..=7
/// directly, then return the manager so a test can drive v8 in isolation.
///
/// `SearchDbManager::new` always migrates all the way up to
/// `SEARCH_SCHEMA_VERSION`, so it cannot stop at v7. Driving the migration
/// dispatcher by hand is the only way to observe the v7 -> v8 step.
async fn build_v7_search_db(db_path: &std::path::Path) -> SearchDbManager {
    let manager = SearchDbManager::with_pool(
        sqlx::SqlitePool::connect(&format!("sqlite://{}?mode=rwc", db_path.display()))
            .await
            .unwrap(),
        db_path.to_path_buf(),
    );

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS search_schema_version (\
            version INTEGER PRIMARY KEY, \
            applied_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))",
    )
    .execute(manager.pool())
    .await
    .unwrap();

    for version in 1..=7 {
        super::migrations::run_migration(manager.pool(), version)
            .await
            .unwrap();
        sqlx::query("INSERT INTO search_schema_version (version) VALUES (?1)")
            .bind(version)
            .execute(manager.pool())
            .await
            .unwrap();
    }

    manager
}

/// Whether `file_metadata` carries a column of the given name.
async fn file_metadata_has_column(pool: &sqlx::SqlitePool, column: &str) -> bool {
    sqlx::query_scalar::<_, bool>(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('file_metadata') WHERE name = ?1",
    )
    .bind(column)
    .fetch_one(pool)
    .await
    .unwrap()
}

/// T-F3-migrate-v7-v8: migrating a v7 search.db to v8 adds the `state` column,
/// and every pre-existing row reads back the DEFAULT `state='present'`.
#[tokio::test]
async fn test_f3_migrate_v7_to_v8() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    let manager = build_v7_search_db(&db_path).await;

    assert_eq!(
        manager.get_schema_version().await.unwrap(),
        Some(7),
        "fixture should be pinned at v7 before the migration"
    );
    assert!(
        !file_metadata_has_column(manager.pool(), "state").await,
        "v7 file_metadata must not yet have a state column"
    );

    // A pre-existing row that must inherit the DEFAULT after the migration.
    sqlx::query(
        "INSERT INTO file_metadata (file_id, tenant_id, branch, file_path) \
         VALUES (1, 'tenant-a', 'main', '/src/lib.rs')",
    )
    .execute(manager.pool())
    .await
    .unwrap();

    // Drive only the v7 -> v8 step.
    super::migrations::run_migration(manager.pool(), 8)
        .await
        .expect("v8 migration should succeed on a v7 database");

    assert!(
        file_metadata_has_column(manager.pool(), "state").await,
        "v8 must add the state column to file_metadata"
    );

    let state: String = sqlx::query_scalar("SELECT state FROM file_metadata WHERE file_id = 1")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(
        state, "present",
        "pre-existing rows must inherit the DEFAULT state='present'"
    );

    manager.close().await;
}

/// T-F3-dispatch-arm-reachable: opening a search.db with the current code
/// migrates cleanly to v8 without hitting the `_ =>` unknown-version error arm.
///
/// This guards the "8 => migrate_v8 arm exists" invariant: if
/// `SEARCH_SCHEMA_VERSION` were bumped to 8 but the dispatch arm were missing,
/// `run_migration(pool, 8)` would return the unknown-version error and the open
/// path below would fail.
#[tokio::test]
async fn test_f3_dispatch_arm_reachable() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");

    // Full open path: must succeed (Ok) for the target version, which is 8.
    let manager = SearchDbManager::new(&db_path)
        .await
        .expect("opening a fresh search.db must migrate to v8 without an unknown-version error");

    assert_eq!(
        manager.get_schema_version().await.unwrap(),
        Some(8),
        "open path must land on v8"
    );

    // Dispatching version 8 directly must resolve to migrate_v8, never `_ =>`.
    super::migrations::run_migration(manager.pool(), 8)
        .await
        .expect("version 8 must dispatch to migrate_v8, not the error arm");

    manager.close().await;
}
