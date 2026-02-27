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

    // Add an enabled, non-paused watcher
    let record = WatchFolderRecord {
        watch_id: "pause-test-001".to_string(),
        path: "/projects/pause-test".to_string(),
        collection: "projects".to_string(),
        tenant_id: "pause-test-tenant".to_string(),
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
    };
    manager.store_watch_folder(&record).await.unwrap();

    // Not paused yet
    assert!(!manager.any_watchers_paused().await.unwrap());

    // Pause all watchers
    let paused = manager.pause_all_watchers().await.unwrap();
    assert_eq!(paused, 1);
    assert!(manager.any_watchers_paused().await.unwrap());

    // Resume
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

    // No change when no watchers
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);
    assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));

    // Add enabled watcher
    let record = WatchFolderRecord {
        watch_id: "poll-test-001".to_string(),
        path: "/projects/poll-test".to_string(),
        collection: "projects".to_string(),
        tenant_id: "poll-test-tenant".to_string(),
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
    };
    manager.store_watch_folder(&record).await.unwrap();

    // Still no change (not paused)
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);

    // Pause via DB
    manager.pause_all_watchers().await.unwrap();

    // Flag should change from false -> true
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(changed);
    assert!(flag.load(std::sync::atomic::Ordering::SeqCst));

    // Poll again - no change (already true)
    let changed = poll_pause_state(manager.pool(), &flag).await.unwrap();
    assert!(!changed);

    // Resume via DB
    manager.resume_all_watchers().await.unwrap();

    // Flag should change from true -> false
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

    // Create an active project with old last_activity_at
    let old_project = WatchFolderRecord {
        watch_id: "old-project".to_string(),
        path: "/projects/old".to_string(),
        collection: "projects".to_string(),
        tenant_id: "old-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: Some(Utc::now() - chrono::Duration::hours(13)),
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
    };
    manager.store_watch_folder(&old_project).await.unwrap();

    // Create an active project with recent activity
    let recent_project = WatchFolderRecord {
        watch_id: "recent-project".to_string(),
        path: "/projects/recent".to_string(),
        collection: "projects".to_string(),
        tenant_id: "recent-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
        last_activity_at: Some(Utc::now() - chrono::Duration::minutes(30)),
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
    };
    manager.store_watch_folder(&recent_project).await.unwrap();

    // 12-hour timeout (43200 seconds)
    let deactivated = manager.deactivate_inactive_projects(43200).await.unwrap();
    assert_eq!(deactivated, 1, "Only the 13h-old project should be deactivated");

    // Verify old project is deactivated
    let old = manager.get_watch_folder("old-project").await.unwrap().unwrap();
    assert!(!old.is_active);

    // Verify recent project is still active
    let recent = manager.get_watch_folder("recent-project").await.unwrap().unwrap();
    assert!(recent.is_active);
}

#[tokio::test]
async fn test_deactivate_inactive_skips_null_activity() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_inactivity_null.db");
    let manager = DaemonStateManager::new(&db_path).await.unwrap();
    manager.initialize().await.unwrap();

    // Create active project with NULL last_activity_at (never had a session)
    let null_project = WatchFolderRecord {
        watch_id: "null-project".to_string(),
        path: "/projects/null".to_string(),
        collection: "projects".to_string(),
        tenant_id: "null-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
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
    };
    manager.store_watch_folder(&null_project).await.unwrap();

    // Should not deactivate projects with NULL last_activity_at
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
        watch_id: "archive-test-001".to_string(),
        path: "/projects/archive-test".to_string(),
        collection: "projects".to_string(),
        tenant_id: "archive-test-tenant".to_string(),
        parent_watch_id: None,
        submodule_path: None,
        git_remote_url: None,
        remote_hash: None,
        disambiguation_path: None,
        is_active: true,
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
    };
    manager.store_watch_folder(&record).await.unwrap();

    // Archive
    let archived = manager.archive_watch_folder("archive-test-001").await.unwrap();
    assert!(archived);
    let r = manager.get_watch_folder("archive-test-001").await.unwrap().unwrap();
    assert!(r.is_archived);

    // Idempotent: archiving again returns false
    let archived_again = manager.archive_watch_folder("archive-test-001").await.unwrap();
    assert!(!archived_again);

    // Unarchive
    let unarchived = manager.unarchive_watch_folder("archive-test-001").await.unwrap();
    assert!(unarchived);
    let r = manager.get_watch_folder("archive-test-001").await.unwrap().unwrap();
    assert!(!r.is_archived);

    // Idempotent: unarchiving again returns false
    let unarchived_again = manager.unarchive_watch_folder("archive-test-001").await.unwrap();
    assert!(!unarchived_again);
}
