//! Tests for watcher-related WriteActor commands:
//! PauseWatchers, ResumeWatchers, EnableWatch, DisableWatch,
//! ArchiveWatch, UnarchiveWatch, WatchLibrary.

use crate::write_actor::commands::*;

use super::common::setup_test_db;

// ── PauseWatchers / ResumeWatchers tests ─────────────────────────────

#[tokio::test]
async fn pause_and_resume_watchers() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    // Insert two enabled, unpaused watchers
    for i in 0..2 {
        sqlx::query(
            "INSERT INTO watch_folders \
             (watch_id, path, collection, tenant_id, enabled, is_paused, created_at, updated_at) \
             VALUES (?1, ?2, 'projects', ?3, 1, 0, ?4, ?4)",
        )
        .bind(format!("w-{}", i))
        .bind(format!("/tmp/proj-{}", i))
        .bind(format!("t-{}", i))
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Insert one disabled watcher that should NOT be paused
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, enabled, is_paused, created_at, updated_at) \
         VALUES ('w-disabled', '/tmp/disabled', 'projects', 't-d', 0, 0, ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Pause
    let paused = handle.pause_watchers().await.unwrap();
    assert_eq!(paused, 2);

    // Verify paused state
    let paused_count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(paused_count, 2);

    // Resume
    let resumed = handle.resume_watchers().await.unwrap();
    assert_eq!(resumed, 2);

    // Verify resumed
    let paused_count =
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM watch_folders WHERE is_paused = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(paused_count, 0);
}

// ── EnableWatch / DisableWatch tests ─────────────────────────────────

#[tokio::test]
async fn enable_and_disable_watch() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, enabled, created_at, updated_at) \
         VALUES ('watch-toggle', '/tmp/toggle', 'projects', 't1', 1, ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Disable
    let affected = handle
        .disable_watch(WatchIdData {
            watch_id: "watch-toggle".into(),
        })
        .await
        .unwrap();
    assert_eq!(affected, 1);

    let enabled = sqlx::query_scalar::<_, i64>(
        "SELECT enabled FROM watch_folders WHERE watch_id = 'watch-toggle'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(enabled, 0);

    // Enable
    let affected = handle
        .enable_watch(WatchIdData {
            watch_id: "watch-toggle".into(),
        })
        .await
        .unwrap();
    assert_eq!(affected, 1);

    let enabled = sqlx::query_scalar::<_, i64>(
        "SELECT enabled FROM watch_folders WHERE watch_id = 'watch-toggle'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(enabled, 1);
}

// ── ArchiveWatch tests ───────────────────────────────────────────────

#[tokio::test]
async fn archive_and_unarchive_watch() {
    let (pool, handle) = setup_test_db().await;

    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, enabled, is_archived, created_at, updated_at) \
         VALUES ('w-arch', '/tmp/arch', 'projects', 't1', 1, 0, ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Archive
    let result = handle
        .archive_watch(ArchiveWatchData {
            watch_id: "w-arch".into(),
            cascade_submodules: false,
        })
        .await
        .unwrap();
    assert_eq!(result.affected_count, 1);

    let archived = sqlx::query_scalar::<_, i64>(
        "SELECT is_archived FROM watch_folders WHERE watch_id = 'w-arch'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(archived, 1);

    // Unarchive
    let affected = handle
        .unarchive_watch(WatchIdData {
            watch_id: "w-arch".into(),
        })
        .await
        .unwrap();
    assert_eq!(affected, 1);

    let archived = sqlx::query_scalar::<_, i64>(
        "SELECT is_archived FROM watch_folders WHERE watch_id = 'w-arch'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(archived, 0);
}

// ── WatchLibrary tests ───────────────────────────────────────────────

#[tokio::test]
async fn watch_library_creates_new_and_reactivates() {
    let (pool, handle) = setup_test_db().await;

    // First call creates a new library watch
    let result = handle
        .watch_library(WatchLibraryData {
            tag: "new-lib".into(),
            path: "/tmp/new-lib".into(),
            mode: "full".into(),
        })
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.is_new);
    assert_eq!(result.watch_id, "lib-new-lib");

    // Second call reactivates existing
    let result = handle
        .watch_library(WatchLibraryData {
            tag: "new-lib".into(),
            path: "/tmp/new-lib".into(),
            mode: "incremental".into(),
        })
        .await
        .unwrap();

    assert!(result.success);
    assert!(!result.is_new);

    // Verify mode was updated
    let mode = sqlx::query_scalar::<_, String>(
        "SELECT library_mode FROM watch_folders WHERE watch_id = 'lib-new-lib'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(mode, "incremental");
}
