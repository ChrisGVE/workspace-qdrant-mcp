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

    let mode: String =
        sqlx::query_scalar("PRAGMA journal_mode")
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

    let fk: i32 =
        sqlx::query_scalar("PRAGMA foreign_keys")
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
    let count: i32 =
        sqlx::query_scalar("SELECT COUNT(*) FROM test_data")
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
    assert_eq!(version, Some(SEARCH_SCHEMA_VERSION), "Schema version should be current");

    manager.close().await;
}
