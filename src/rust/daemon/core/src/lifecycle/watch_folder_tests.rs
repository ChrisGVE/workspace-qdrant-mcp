//! Tests for [`WatchFolderLifecycle`].

use super::*;
use sqlx::sqlite::SqlitePoolOptions;
use std::time::Duration;

use crate::watch_folders_schema::{CREATE_WATCH_FOLDERS_SQL, CREATE_WATCH_FOLDER_SUBMODULES_SQL};

async fn test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("in-memory pool")
}

async fn setup(pool: &SqlitePool) {
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(CREATE_WATCH_FOLDERS_SQL)
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(CREATE_WATCH_FOLDER_SUBMODULES_SQL)
        .execute(pool)
        .await
        .unwrap();
}

async fn insert_project(pool: &SqlitePool, watch_id: &str, tenant: &str, path: &str) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, \
         is_active, created_at, updated_at) \
         VALUES (?1, ?2, 'projects', ?3, 0, \
         strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
    )
    .bind(watch_id)
    .bind(path)
    .bind(tenant)
    .execute(pool)
    .await
    .unwrap();
}

async fn insert_submodule(
    pool: &SqlitePool,
    watch_id: &str,
    parent_id: &str,
    tenant: &str,
    path: &str,
) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, \
         parent_watch_id, is_active, created_at, updated_at) \
         VALUES (?1, ?2, 'projects', ?3, ?4, 0, \
         strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
    )
    .bind(watch_id)
    .bind(path)
    .bind(tenant)
    .bind(parent_id)
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO watch_folder_submodules (parent_watch_id, child_watch_id, \
         submodule_path, created_at) \
         VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
    )
    .bind(parent_id)
    .bind(watch_id)
    .bind(path)
    .execute(pool)
    .await
    .unwrap();
}

async fn is_active(pool: &SqlitePool, watch_id: &str) -> bool {
    sqlx::query_scalar::<_, bool>("SELECT is_active FROM watch_folders WHERE watch_id = ?1")
        .bind(watch_id)
        .fetch_one(pool)
        .await
        .unwrap()
}

// ── tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_activate_deactivate_project_group() {
    let pool = test_pool().await;
    setup(&pool).await;

    insert_project(&pool, "root", "t1", "/p").await;
    insert_submodule(&pool, "child1", "root", "t1", "/p/sub1").await;
    insert_submodule(&pool, "child2", "root", "t1", "/p/sub2").await;

    let lc = WatchFolderLifecycle::new(pool.clone());

    // Activate root + children
    let rows = lc.activate_project_group("root").await.unwrap();
    assert_eq!(rows, 3);
    assert!(is_active(&pool, "root").await);
    assert!(is_active(&pool, "child1").await);
    assert!(is_active(&pool, "child2").await);

    // Deactivate root + children
    let rows = lc.deactivate_project_group("root").await.unwrap();
    assert_eq!(rows, 3);
    assert!(!is_active(&pool, "root").await);
    assert!(!is_active(&pool, "child1").await);
    assert!(!is_active(&pool, "child2").await);
}

#[tokio::test]
async fn test_activate_deactivate_by_tenant() {
    let pool = test_pool().await;
    setup(&pool).await;
    insert_project(&pool, "w1", "tenant_a", "/a").await;

    let lc = WatchFolderLifecycle::new(pool.clone());

    let rows = lc.activate_by_tenant("tenant_a", "projects").await.unwrap();
    assert_eq!(rows, 1);
    assert!(is_active(&pool, "w1").await);

    let rows = lc
        .deactivate_by_tenant("tenant_a", "projects")
        .await
        .unwrap();
    assert_eq!(rows, 1);
    assert!(!is_active(&pool, "w1").await);
}

#[tokio::test]
async fn test_set_active_by_tenant() {
    let pool = test_pool().await;
    setup(&pool).await;
    insert_project(&pool, "w1", "tenant_b", "/b").await;

    let lc = WatchFolderLifecycle::new(pool.clone());

    lc.set_active_by_tenant("tenant_b", "projects", true)
        .await
        .unwrap();
    assert!(is_active(&pool, "w1").await);

    lc.set_active_by_tenant("tenant_b", "projects", false)
        .await
        .unwrap();
    assert!(!is_active(&pool, "w1").await);
}

#[tokio::test]
async fn test_set_active_by_path() {
    let pool = test_pool().await;
    setup(&pool).await;
    insert_project(&pool, "w1", "tenant_c", "/c").await;

    let lc = WatchFolderLifecycle::new(pool.clone());

    lc.set_active_by_path("/c", true).await.unwrap();
    assert!(is_active(&pool, "w1").await);

    lc.set_active_by_path("/c", false).await.unwrap();
    assert!(!is_active(&pool, "w1").await);
}

#[tokio::test]
async fn test_deactivate_by_watch_id() {
    let pool = test_pool().await;
    setup(&pool).await;
    insert_project(&pool, "w1", "tenant_d", "/d").await;

    // Activate first
    sqlx::query("UPDATE watch_folders SET is_active = 1 WHERE watch_id = 'w1'")
        .execute(&pool)
        .await
        .unwrap();
    assert!(is_active(&pool, "w1").await);

    let lc = WatchFolderLifecycle::new(pool.clone());
    lc.deactivate_by_watch_id("w1").await.unwrap();
    assert!(!is_active(&pool, "w1").await);
}

#[tokio::test]
async fn test_deactivate_orphaned_tenants() {
    let pool = test_pool().await;
    setup(&pool).await;
    insert_project(&pool, "w1", "t1", "/p1").await;
    insert_project(&pool, "w2", "t2", "/p2").await;
    insert_project(&pool, "w3", "t3", "/p3").await;

    // Activate all
    sqlx::query("UPDATE watch_folders SET is_active = 1")
        .execute(&pool)
        .await
        .unwrap();

    let lc = WatchFolderLifecycle::new(pool.clone());

    let ids = vec!["t1".to_string(), "t3".to_string()];
    let rows = lc
        .deactivate_orphaned_tenants(&ids, "projects")
        .await
        .unwrap();
    assert_eq!(rows, 2);

    assert!(!is_active(&pool, "w1").await);
    assert!(is_active(&pool, "w2").await); // untouched
    assert!(!is_active(&pool, "w3").await);
}

#[tokio::test]
async fn test_deactivate_orphaned_tenants_empty_list() {
    let pool = test_pool().await;
    setup(&pool).await;

    let lc = WatchFolderLifecycle::new(pool.clone());
    let rows = lc
        .deactivate_orphaned_tenants(&[], "projects")
        .await
        .unwrap();
    assert_eq!(rows, 0);
}

#[tokio::test]
async fn test_find_stale_active_tenants() {
    let pool = test_pool().await;
    setup(&pool).await;
    insert_project(&pool, "w1", "t1", "/p1").await;
    insert_project(&pool, "w2", "t2", "/p2").await;

    // Activate both with old timestamps
    sqlx::query(
        "UPDATE watch_folders SET is_active = 1, \
         last_activity_at = '2020-01-01T00:00:00.000Z'",
    )
    .execute(&pool)
    .await
    .unwrap();

    let lc = WatchFolderLifecycle::new(pool.clone());

    let stale = lc
        .find_stale_active_tenants("projects", "2025-01-01T00:00:00.000Z")
        .await
        .unwrap();
    assert_eq!(stale.len(), 2);

    // With a cutoff in the past, nothing should be stale
    let stale = lc
        .find_stale_active_tenants("projects", "2019-01-01T00:00:00.000Z")
        .await
        .unwrap();
    assert!(stale.is_empty());
}

#[tokio::test]
async fn test_nonexistent_tenant_returns_zero_rows() {
    let pool = test_pool().await;
    setup(&pool).await;

    let lc = WatchFolderLifecycle::new(pool.clone());
    let rows = lc
        .activate_by_tenant("nonexistent", "projects")
        .await
        .unwrap();
    assert_eq!(rows, 0);
}
