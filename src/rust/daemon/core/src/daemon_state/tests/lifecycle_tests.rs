//! Pause/resume, deactivate inactive projects, and archive/unarchive tests.

use super::*;

#[tokio::test]
async fn test_any_watchers_paused() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("pause_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // No watchers at all - should return false
    assert!(!manager.any_watchers_paused().await.unwrap());

    let record = make_test_watch_folder("pause-test-001", "/projects/pause-test", "pause-test-tenant");
    manager.store_watch_folder(&record).await.unwrap();

    assert!(!manager.any_watchers_paused().await.unwrap());

    let paused = manager.pause_all_watchers().await.unwrap();
    assert_eq!(paused, 1);
    assert!(manager.any_watchers_paused().await.unwrap());

    let resumed = manager.resume_all_watchers().await.unwrap();
    assert_eq!(resumed, 1);
    assert!(!manager.any_watchers_paused().await.unwrap());
}

#[tokio::test]
async fn test_poll_pause_state() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("poll_pause_test.db");

    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let flag = std::sync::atomic::AtomicBool::new(false);

    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);
    assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));

    let record = make_test_watch_folder("poll-test-001", "/projects/poll-test", "poll-test-tenant");
    manager.store_watch_folder(&record).await.unwrap();

    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);

    manager.pause_all_watchers().await.unwrap();

    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(changed);
    assert!(flag.load(std::sync::atomic::Ordering::SeqCst));

    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);

    manager.resume_all_watchers().await.unwrap();

    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(changed);
    assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test]
async fn test_deactivate_inactive_projects() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_inactivity.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let old_project = WatchFolderRecord {
        is_active: true,
        last_activity_at: Some(Utc::now() - chrono::Duration::hours(13)),
        ..make_test_watch_folder("old-project", "/projects/old", "old-tenant")
    };
    manager.store_watch_folder(&old_project).await.unwrap();

    let recent_project = WatchFolderRecord {
        is_active: true,
        last_activity_at: Some(Utc::now() - chrono::Duration::minutes(30)),
        ..make_test_watch_folder("recent-project", "/projects/recent", "recent-tenant")
    };
    manager.store_watch_folder(&recent_project).await.unwrap();

    let deactivated = manager.deactivate_inactive_projects(43200).await.unwrap();
    assert_eq!(deactivated, 1, "Only the 13h-old project should be deactivated");

    let old = manager.get_watch_folder("old-project").await.unwrap().unwrap();
    assert!(!old.is_active);

    let recent = manager.get_watch_folder("recent-project").await.unwrap().unwrap();
    assert!(recent.is_active);
}

#[tokio::test]
async fn test_deactivate_inactive_skips_null_activity() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_inactivity_null.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let null_project = WatchFolderRecord {
        is_active: true,
        ..make_test_watch_folder("null-project", "/projects/null", "null-tenant")
    };
    manager.store_watch_folder(&null_project).await.unwrap();

    let deactivated = manager.deactivate_inactive_projects(43200).await.unwrap();
    assert_eq!(deactivated, 0);

    let record = manager.get_watch_folder("null-project").await.unwrap().unwrap();
    assert!(record.is_active);
}

#[tokio::test]
async fn test_archive_unarchive_watch_folder() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("archive_test.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    let record = WatchFolderRecord {
        is_active: true,
        ..make_test_watch_folder("archive-test-001", "/projects/archive-test", "archive-test-tenant")
    };
    manager.store_watch_folder(&record).await.unwrap();

    let archived = manager.archive_watch_folder("archive-test-001").await.unwrap();
    assert!(archived);
    let r = manager.get_watch_folder("archive-test-001").await.unwrap().unwrap();
    assert!(r.is_archived);

    let archived_again = manager.archive_watch_folder("archive-test-001").await.unwrap();
    assert!(!archived_again);

    let unarchived = manager.unarchive_watch_folder("archive-test-001").await.unwrap();
    assert!(unarchived);
    let r = manager.get_watch_folder("archive-test-001").await.unwrap().unwrap();
    assert!(!r.is_archived);

    let unarchived_again = manager.unarchive_watch_folder("archive-test-001").await.unwrap();
    assert!(!unarchived_again);
}
