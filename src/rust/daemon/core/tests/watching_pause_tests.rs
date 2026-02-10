//! Integration tests for watcher pause/resume functionality (Task 543.19)
//!
//! Tests the full pause/resume lifecycle including database state,
//! in-memory flag synchronization, and poll_pause_state behavior.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tempfile::tempdir;

use workspace_qdrant_core::{
    DaemonStateManager,
    poll_pause_state,
    daemon_state::WatchFolderRecord,
};
use chrono::Utc;

/// Helper to create a test DaemonStateManager with initialized schema
async fn setup_test_db() -> (DaemonStateManager, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("pause_integration_test.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();
    (manager, temp_dir)
}

/// Helper to create a test watch folder record
fn test_watch_record(id: &str, path: &str) -> WatchFolderRecord {
    WatchFolderRecord {
        watch_id: id.to_string(),
        path: path.to_string(),
        collection: "projects".to_string(),
        tenant_id: format!("{}-tenant", id),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        library_mode: None,
        follow_symlinks: false,
        enabled: true,
        cleanup_on_disable: false,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_scan: None,
    }
}

#[tokio::test]
async fn test_pause_resume_full_lifecycle() {
    let (manager, _dir) = setup_test_db().await;

    // Register two watch folders
    manager.store_watch_folder(&test_watch_record("watch-a", "/projects/a")).await.unwrap();
    manager.store_watch_folder(&test_watch_record("watch-b", "/projects/b")).await.unwrap();

    // Verify not paused initially
    assert!(!manager.any_watchers_paused().await.unwrap());
    let paused_ids = manager.get_paused_watch_ids().await.unwrap();
    assert!(paused_ids.is_empty());

    // Pause all
    let count = manager.pause_all_watchers().await.unwrap();
    assert_eq!(count, 2);

    // Verify paused state
    assert!(manager.any_watchers_paused().await.unwrap());
    let paused_ids = manager.get_paused_watch_ids().await.unwrap();
    assert_eq!(paused_ids.len(), 2);

    // Verify pause_start_time is set
    let folder_a = manager.get_watch_folder("watch-a").await.unwrap().unwrap();
    assert!(folder_a.is_paused);
    assert!(folder_a.pause_start_time.is_some());

    // Resume all
    let count = manager.resume_all_watchers().await.unwrap();
    assert_eq!(count, 2);

    // Verify resumed state
    assert!(!manager.any_watchers_paused().await.unwrap());
    let paused_ids = manager.get_paused_watch_ids().await.unwrap();
    assert!(paused_ids.is_empty());

    // Verify pause_start_time is cleared
    let folder_a = manager.get_watch_folder("watch-a").await.unwrap().unwrap();
    assert!(!folder_a.is_paused);
    assert!(folder_a.pause_start_time.is_none());
}

#[tokio::test]
async fn test_pause_only_affects_enabled_watchers() {
    let (manager, _dir) = setup_test_db().await;

    // Register one enabled, one disabled
    manager.store_watch_folder(&test_watch_record("enabled-watch", "/projects/enabled")).await.unwrap();
    let mut disabled = test_watch_record("disabled-watch", "/projects/disabled");
    disabled.enabled = false;
    manager.store_watch_folder(&disabled).await.unwrap();

    // Pause all
    let count = manager.pause_all_watchers().await.unwrap();
    assert_eq!(count, 1); // Only enabled one paused

    // Verify disabled watch is not paused
    let disabled_folder = manager.get_watch_folder("disabled-watch").await.unwrap().unwrap();
    assert!(!disabled_folder.is_paused);
}

#[tokio::test]
async fn test_idempotent_pause_resume() {
    let (manager, _dir) = setup_test_db().await;
    manager.store_watch_folder(&test_watch_record("idem-watch", "/projects/idem")).await.unwrap();

    // First pause
    let count1 = manager.pause_all_watchers().await.unwrap();
    assert_eq!(count1, 1);

    // Second pause (already paused) - should affect 0
    let count2 = manager.pause_all_watchers().await.unwrap();
    assert_eq!(count2, 0);

    // First resume
    let count3 = manager.resume_all_watchers().await.unwrap();
    assert_eq!(count3, 1);

    // Second resume (already resumed) - should affect 0
    let count4 = manager.resume_all_watchers().await.unwrap();
    assert_eq!(count4, 0);
}

#[tokio::test]
async fn test_poll_pause_state_sync() {
    let (manager, _dir) = setup_test_db().await;
    manager.store_watch_folder(&test_watch_record("poll-watch", "/projects/poll")).await.unwrap();

    let flag = Arc::new(AtomicBool::new(false));

    // Initial poll - no change
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);
    assert!(!flag.load(Ordering::SeqCst));

    // Simulate CLI pause (direct DB write)
    sqlx::query(
        "UPDATE watch_folders SET is_paused = 1, \
         pause_start_time = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE watch_id = 'poll-watch'"
    )
    .execute(manager.pool())
    .await
    .unwrap();

    // Poll detects change
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(changed);
    assert!(flag.load(Ordering::SeqCst));

    // Poll again - no change
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);

    // Simulate CLI resume (direct DB write)
    sqlx::query(
        "UPDATE watch_folders SET is_paused = 0, pause_start_time = NULL, \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE watch_id = 'poll-watch'"
    )
    .execute(manager.pool())
    .await
    .unwrap();

    // Poll detects change
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(changed);
    assert!(!flag.load(Ordering::SeqCst));
}

#[tokio::test]
async fn test_pause_state_persists_across_manager_instances() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("persistence_test.db");

    // Create and pause with first instance
    {
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();
        manager.store_watch_folder(&test_watch_record("persist-watch", "/projects/persist")).await.unwrap();
        manager.pause_all_watchers().await.unwrap();
    }

    // Verify with second instance (simulates daemon restart)
    {
        let manager = DaemonStateManager::new(&db_path).await.unwrap();
        manager.initialize().await.unwrap();

        assert!(manager.any_watchers_paused().await.unwrap());

        let flag = AtomicBool::new(false);
        poll_pause_state(manager.pool(), &flag).await.unwrap();
        assert!(flag.load(Ordering::SeqCst));
    }
}
