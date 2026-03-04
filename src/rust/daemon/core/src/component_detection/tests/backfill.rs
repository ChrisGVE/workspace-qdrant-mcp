use super::create_backfill_schema;
use super::super::*;
use std::fs;
use tempfile::TempDir;

/// Create an in-memory pool with backfill schema, insert a watch folder, and
/// return the pool plus the stringified path of the temp dir.
async fn setup_backfill_pool(dir: &TempDir) -> (sqlx::SqlitePool, String) {
    let pool = sqlx::SqlitePool::connect("sqlite::memory:")
        .await
        .unwrap();
    create_backfill_schema(&pool).await;

    let path_str = dir.path().to_string_lossy().to_string();
    sqlx::query("INSERT INTO watch_folders (watch_id, path) VALUES ('wf1', ?1)")
        .bind(&path_str)
        .execute(&pool)
        .await
        .unwrap();

    (pool, path_str)
}

/// Insert tracked files with NULL component for the given watch folder.
async fn insert_tracked_files_for(
    pool: &sqlx::SqlitePool,
    watch_id: &str,
    path_str: &str,
    rels: &[&str],
) {
    for rel in rels {
        let abs = format!("{}/{}", path_str, rel);
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, relative_path)
             VALUES (?1, ?2, ?3)",
        )
        .bind(watch_id)
        .bind(&abs)
        .bind(rel)
        .execute(pool)
        .await
        .unwrap();
    }
}

/// Insert tracked files with NULL component for watch folder 'wf1'.
async fn insert_tracked_files(pool: &sqlx::SqlitePool, path_str: &str, rels: &[&str]) {
    insert_tracked_files_for(pool, "wf1", path_str, rels).await;
}

#[tokio::test]
async fn test_backfill_components() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        "[workspace]\nmembers = [\"alpha\", \"beta\"]\n",
    )
    .unwrap();
    fs::create_dir_all(dir.path().join("alpha")).unwrap();
    fs::create_dir_all(dir.path().join("beta")).unwrap();

    let (pool, path_str) = setup_backfill_pool(&dir).await;
    insert_tracked_files(&pool, &path_str, &["alpha/src/lib.rs", "beta/src/main.rs", "README.md"])
        .await;

    let stats = backfill_components(&pool, 100, false, None).await.unwrap();
    assert_eq!(stats.folders_processed, 1);
    assert_eq!(stats.files_updated, 2);
    assert_eq!(stats.files_unmatched, 1);

    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'alpha/src/lib.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("alpha"));

    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'beta/src/main.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("beta"));

    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'README.md'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(row.0.is_none());

    let count: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM project_components WHERE watch_folder_id = 'wf1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count.0, 2);
}

#[tokio::test]
async fn test_backfill_components_force() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        "[workspace]\nmembers = [\"alpha\", \"beta\"]\n",
    )
    .unwrap();
    fs::create_dir_all(dir.path().join("alpha")).unwrap();
    fs::create_dir_all(dir.path().join("beta")).unwrap();

    let (pool, path_str) = setup_backfill_pool(&dir).await;
    insert_tracked_files(&pool, &path_str, &["alpha/src/lib.rs", "beta/src/main.rs"]).await;

    // First backfill assigns components
    let stats = backfill_components(&pool, 100, false, None).await.unwrap();
    assert_eq!(stats.files_updated, 2);

    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'alpha/src/lib.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("alpha"));

    // Rename workspace member: alpha -> gamma
    fs::write(
        dir.path().join("Cargo.toml"),
        "[workspace]\nmembers = [\"gamma\", \"beta\"]\n",
    )
    .unwrap();
    fs::create_dir_all(dir.path().join("gamma")).unwrap();

    sqlx::query(
        "UPDATE tracked_files SET relative_path = 'gamma/src/lib.rs' \
         WHERE relative_path = 'alpha/src/lib.rs'",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Non-force should NOT update (component already non-NULL)
    let stats = backfill_components(&pool, 100, false, None).await.unwrap();
    assert_eq!(stats.files_updated, 0);

    // Force backfill SHOULD reassign
    let stats = backfill_components(&pool, 100, true, None).await.unwrap();
    assert_eq!(stats.files_updated, 2);

    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'gamma/src/lib.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("gamma"));

    // Stale "alpha" component should be cleaned up
    let count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM project_components \
         WHERE watch_folder_id = 'wf1' AND component_name = 'alpha'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(count.0, 0);

    let count: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM project_components WHERE watch_folder_id = 'wf1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count.0, 2);
}

#[tokio::test]
async fn test_backfill_components_tenant_filter() {
    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();

    for dir in [&dir_a, &dir_b] {
        fs::write(
            dir.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"crate-x\"]\n",
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("crate-x")).unwrap();
    }

    let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
    create_backfill_schema(&pool).await;

    let path_a = dir_a.path().to_string_lossy().to_string();
    let path_b = dir_b.path().to_string_lossy().to_string();

    for (wid, path, tid) in [("wf_a", &path_a, "tenant_a"), ("wf_b", &path_b, "tenant_b")] {
        sqlx::query("INSERT INTO watch_folders (watch_id, path, tenant_id) VALUES (?1, ?2, ?3)")
            .bind(wid)
            .bind(path)
            .bind(tid)
            .execute(&pool)
            .await
            .unwrap();
    }

    insert_tracked_files_for(&pool, "wf_a", &path_a, &["crate-x/src/lib.rs"]).await;
    insert_tracked_files_for(&pool, "wf_b", &path_b, &["crate-x/src/lib.rs"]).await;

    let stats = backfill_components(&pool, 100, false, Some("tenant_a"))
        .await
        .unwrap();
    assert_eq!(stats.folders_processed, 1);
    assert_eq!(stats.files_updated, 1);

    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE watch_folder_id = 'wf_a'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("crate-x"));

    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE watch_folder_id = 'wf_b'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(row.0.is_none());
}
