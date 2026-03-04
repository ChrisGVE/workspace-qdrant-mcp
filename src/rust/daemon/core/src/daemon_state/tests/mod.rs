//! Tests for daemon state management.

use super::*;
use chrono::Utc;
use tempfile::tempdir;

mod activation_tests;
mod lifecycle_tests;
mod operational_state_tests;
mod registration_tests;
mod submodule_tests;
mod watch_folder_tests;

/// Build a `WatchFolderRecord` with sensible defaults.
/// Override any field via the returned struct.
fn make_test_watch_folder(watch_id: &str, path: &str, tenant_id: &str) -> WatchFolderRecord {
    WatchFolderRecord {
        watch_id: watch_id.to_string(),
        path: path.to_string(),
        collection: "projects".to_string(),
        tenant_id: tenant_id.to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: false,
        last_activity_at: None,
        is_paused: false,
        pause_start_time: None,
        is_archived: false,
        last_commit_hash: None,
        is_git_tracked: false,
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
async fn test_daemon_state_creation() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("daemon_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    assert!(db_path.exists());
}
