//! Tests for library-related WriteActor commands:
//! AddLibrary, RemoveLibrary.

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
