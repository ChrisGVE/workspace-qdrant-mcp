//! Tests for library-related WriteActor commands:
//! AddLibrary, RemoveLibrary, SetIncremental.

use crate::write_actor::commands::*;

use super::common::setup_test_db;

// ── AddLibrary / RemoveLibrary tests ─────────────────────────────────

#[tokio::test]
async fn add_and_remove_library() {
    let (pool, handle) = setup_test_db().await;

    // Add
    let add_result = handle
        .add_library(AddLibraryData {
            tag: "my-lib".into(),
            path: "/tmp/my-lib".into(),
            mode: "full".into(),
        })
        .await
        .unwrap();

    assert!(add_result.success);
    assert_eq!(add_result.watch_id, "lib-my-lib");

    // Verify it exists
    let exists = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE watch_id = 'lib-my-lib'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(exists, 1);

    // Adding same tag again should fail
    let dup = handle
        .add_library(AddLibraryData {
            tag: "my-lib".into(),
            path: "/tmp/my-lib-2".into(),
            mode: "full".into(),
        })
        .await
        .unwrap();
    assert!(!dup.success);

    // Remove
    let rm_result = handle
        .remove_library(RemoveLibraryData {
            tag: "my-lib".into(),
        })
        .await
        .unwrap();

    assert!(rm_result.success);

    // Verify gone
    let exists = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM watch_folders WHERE watch_id = 'lib-my-lib'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(exists, 0);
}

#[tokio::test]
async fn remove_nonexistent_library_errors() {
    let (_pool, handle) = setup_test_db().await;

    let result = handle
        .remove_library(RemoveLibraryData {
            tag: "ghost".into(),
        })
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ── SetIncremental tests ─────────────────────────────────────────────

/// End-to-end test: relative paths from gRPC flow correctly through the
/// WriteActor to update the `incremental` flag on `tracked_files` rows
/// matched by `relative_path` (post-v37 schema).
#[tokio::test]
async fn set_incremental_matches_relative_path() {
    let (pool, handle) = setup_test_db().await;
    let now = wqm_common::timestamps::now_utc();

    // Seed a watch_folder and two tracked_files with relative_path values.
    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('lib-docs', '/home/user/docs', 'libraries', 'docs', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, relative_path, created_at, updated_at) \
         VALUES ('lib-docs', 'chapter1/intro.md', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, relative_path, created_at, updated_at) \
         VALUES ('lib-docs', 'chapter2/deep.md', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Set incremental on one existing file and one non-existent file.
    let result = handle
        .set_incremental(SetIncrementalData {
            file_paths: vec!["chapter1/intro.md".into(), "nonexistent/file.rs".into()],
            clear: false,
            watch_folder_id: None,
        })
        .await
        .unwrap();

    assert_eq!(result.updated, 1, "one file should be updated");
    assert_eq!(result.not_found, 1, "one file should be not found");

    // Verify the flag is actually set in the database.
    let incremental: i32 = sqlx::query_scalar(
        "SELECT incremental FROM tracked_files WHERE relative_path = 'chapter1/intro.md'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(incremental, 1, "incremental flag should be set to 1");

    // The other file should still have incremental = 0.
    let incremental2: i32 = sqlx::query_scalar(
        "SELECT incremental FROM tracked_files WHERE relative_path = 'chapter2/deep.md'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(incremental2, 0, "untouched file should remain 0");
}

/// Clearing the incremental flag resets it to 0.
#[tokio::test]
async fn set_incremental_clear_resets_flag() {
    let (pool, handle) = setup_test_db().await;
    let now = wqm_common::timestamps::now_utc();

    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('lib-src', '/home/user/src', 'libraries', 'src', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Insert a file with incremental already set to 1.
    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, relative_path, incremental, created_at, updated_at) \
         VALUES ('lib-src', 'lib/core.rs', 1, ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Clear the flag.
    let result = handle
        .set_incremental(SetIncrementalData {
            file_paths: vec!["lib/core.rs".into()],
            clear: true,
            watch_folder_id: None,
        })
        .await
        .unwrap();

    assert_eq!(result.updated, 1);
    assert_eq!(result.not_found, 0);

    let incremental: i32 = sqlx::query_scalar(
        "SELECT incremental FROM tracked_files WHERE relative_path = 'lib/core.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(incremental, 0, "incremental flag should be cleared to 0");
}

/// Absolute paths (which the old code expected) should NOT match any rows,
/// proving the SQL now correctly uses relative_path directly.
#[tokio::test]
async fn set_incremental_absolute_path_does_not_match() {
    let (pool, handle) = setup_test_db().await;
    let now = wqm_common::timestamps::now_utc();

    sqlx::query(
        "INSERT INTO watch_folders \
         (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES ('lib-abs', '/home/user/project', 'libraries', 'proj', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO tracked_files \
         (watch_folder_id, relative_path, created_at, updated_at) \
         VALUES ('lib-abs', 'src/main.rs', ?1, ?1)",
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    // Attempt to set incremental using an absolute path (the old bug).
    // This should NOT match because the SQL now compares against
    // tracked_files.relative_path directly.
    let result = handle
        .set_incremental(SetIncrementalData {
            file_paths: vec!["/home/user/project/src/main.rs".into()],
            clear: false,
            watch_folder_id: None,
        })
        .await
        .unwrap();

    assert_eq!(
        result.not_found, 1,
        "absolute path should not match relative_path column"
    );
    assert_eq!(result.updated, 0);

    // Verify the flag was NOT changed.
    let incremental: i32 = sqlx::query_scalar(
        "SELECT incremental FROM tracked_files WHERE relative_path = 'src/main.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(incremental, 0, "incremental flag should remain unchanged");
}
