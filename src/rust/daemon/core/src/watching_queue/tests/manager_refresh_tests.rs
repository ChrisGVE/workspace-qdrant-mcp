//! Regression tests for WatchManager handling of orphaned (missing-path) and
//! archived watch folders.
//!
//! Covers the incident where a deleted worktree's watch_folders row stayed
//! `enabled = 1` and the 5-minute refresh loop retried the watcher start
//! forever ("Failed to start watcher ...: Path does not exist"). The manager
//! must auto-disable such rows after a strike threshold, and refresh must
//! apply the same `is_archived` filter as the startup loaders.

use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use tempfile::TempDir;

use crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL;
use crate::watching_queue::manager::{WatchManager, MISSING_PATH_DISABLE_THRESHOLD};
use crate::AllowedExtensions;

/// In-memory pool with the canonical production watch_folders schema.
async fn create_test_pool() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("in-memory pool");
    sqlx::query(CREATE_WATCH_FOLDERS_SQL)
        .execute(&pool)
        .await
        .expect("create watch_folders");
    pool
}

async fn insert_watch_folder(
    pool: &SqlitePool,
    watch_id: &str,
    path: &str,
    collection: &str,
    tenant_id: &str,
    is_archived: bool,
) {
    sqlx::query(
        r#"
        INSERT INTO watch_folders (
            watch_id, path, collection, tenant_id, enabled, is_archived,
            created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, 1, ?5,
                  strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                  strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        "#,
    )
    .bind(watch_id)
    .bind(path)
    .bind(collection)
    .bind(tenant_id)
    .bind(is_archived)
    .execute(pool)
    .await
    .expect("insert watch folder");
}

async fn fetch_enabled(pool: &SqlitePool, watch_id: &str) -> bool {
    sqlx::query("SELECT enabled FROM watch_folders WHERE watch_id = ?1")
        .bind(watch_id)
        .fetch_one(pool)
        .await
        .expect("fetch watch folder")
        .get::<bool, _>("enabled")
}

fn new_manager(pool: &SqlitePool) -> WatchManager {
    WatchManager::new(pool.clone(), Arc::new(AllowedExtensions::default()))
}

#[tokio::test]
async fn missing_path_watch_folder_auto_disabled_after_threshold() {
    let pool = create_test_pool().await;
    insert_watch_folder(
        &pool,
        "wf-orphan",
        "/nonexistent/wqm-orphan-watch-test",
        "projects",
        "tenant-a",
        false,
    )
    .await;
    let manager = new_manager(&pool);

    for attempt in 1..MISSING_PATH_DISABLE_THRESHOLD {
        manager.refresh_watches().await.expect("refresh");
        assert_eq!(manager.active_watcher_count().await, 0);
        assert!(
            fetch_enabled(&pool, "wf-orphan").await,
            "must stay enabled before the threshold (attempt {attempt})"
        );
    }

    manager.refresh_watches().await.expect("refresh");
    assert!(
        !fetch_enabled(&pool, "wf-orphan").await,
        "missing-path watch folder must be disabled after {MISSING_PATH_DISABLE_THRESHOLD} attempts"
    );

    // Once disabled, the refresh loop no longer sees the row at all.
    manager.refresh_watches().await.expect("refresh");
    assert_eq!(manager.active_watcher_count().await, 0);
}

#[tokio::test]
async fn startup_load_counts_missing_path_strikes() {
    let pool = create_test_pool().await;
    insert_watch_folder(
        &pool,
        "wf-orphan",
        "/nonexistent/wqm-orphan-startup-test",
        "projects",
        "tenant-a",
        false,
    )
    .await;
    let manager = new_manager(&pool);

    manager.start_all_watches().await.expect("start all");
    for _ in 1..MISSING_PATH_DISABLE_THRESHOLD {
        manager.refresh_watches().await.expect("refresh");
    }

    assert!(
        !fetch_enabled(&pool, "wf-orphan").await,
        "startup load must count as a strike toward auto-disable"
    );
}

#[tokio::test]
async fn missing_path_strikes_reset_when_path_reappears() {
    let pool = create_test_pool().await;
    let temp = TempDir::new().expect("tempdir");
    let path = temp.path().join("late-project");
    insert_watch_folder(
        &pool,
        "wf-late",
        path.to_str().expect("utf-8 path"),
        "projects",
        "tenant-a",
        false,
    )
    .await;
    let manager = new_manager(&pool);

    for _ in 1..MISSING_PATH_DISABLE_THRESHOLD {
        manager.refresh_watches().await.expect("refresh");
    }
    assert!(fetch_enabled(&pool, "wf-late").await);

    std::fs::create_dir_all(&path).expect("create watch dir");
    manager.refresh_watches().await.expect("refresh");

    assert!(
        fetch_enabled(&pool, "wf-late").await,
        "watch folder must not be disabled once the path exists again"
    );
    assert!(manager.is_watch_active("wf-late").await);
    manager.stop_all_watches().await.expect("stop");
}

#[tokio::test]
async fn library_watch_with_missing_path_auto_disabled() {
    let pool = create_test_pool().await;
    insert_watch_folder(
        &pool,
        "wf-lib",
        "/nonexistent/wqm-lib-watch-test",
        "libraries",
        "somelib",
        false,
    )
    .await;
    let manager = new_manager(&pool);

    for _ in 0..MISSING_PATH_DISABLE_THRESHOLD {
        manager.refresh_watches().await.expect("refresh");
    }

    // Library watchers run under the runtime id `lib_<tenant>`; the disable
    // must still target the row's own watch_id.
    assert!(
        !fetch_enabled(&pool, "wf-lib").await,
        "library watch folder must be disabled via its DB watch_id"
    );
}

#[tokio::test]
async fn archived_watch_folder_is_not_started() {
    let pool = create_test_pool().await;
    let temp = TempDir::new().expect("tempdir");
    insert_watch_folder(
        &pool,
        "wf-archived",
        temp.path().to_str().expect("utf-8 path"),
        "projects",
        "tenant-a",
        true,
    )
    .await;
    let manager = new_manager(&pool);

    manager.refresh_watches().await.expect("refresh");

    assert!(!manager.is_watch_active("wf-archived").await);
    assert!(
        fetch_enabled(&pool, "wf-archived").await,
        "archived rows are skipped, not disabled"
    );
}

#[tokio::test]
async fn archiving_watch_folder_stops_running_watcher() {
    let pool = create_test_pool().await;
    let temp = TempDir::new().expect("tempdir");
    insert_watch_folder(
        &pool,
        "wf-live",
        temp.path().to_str().expect("utf-8 path"),
        "projects",
        "tenant-a",
        false,
    )
    .await;
    let manager = new_manager(&pool);

    manager.refresh_watches().await.expect("refresh");
    assert!(manager.is_watch_active("wf-live").await);

    sqlx::query("UPDATE watch_folders SET is_archived = 1 WHERE watch_id = 'wf-live'")
        .execute(&pool)
        .await
        .expect("archive row");

    manager.refresh_watches().await.expect("refresh");
    assert!(
        !manager.is_watch_active("wf-live").await,
        "archiving must deregister the running watcher"
    );
}
