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

#[tokio::test]
async fn test_daemon_state_creation() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("daemon_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    assert!(db_path.exists());
}
