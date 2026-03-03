use super::create_backfill_schema;
use super::super::*;
use std::fs;
use tempfile::TempDir;

#[tokio::test]
async fn test_backfill_components() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"
[workspace]
members = ["alpha", "beta"]
"#,
    )
    .unwrap();
    fs::create_dir_all(dir.path().join("alpha")).unwrap();
    fs::create_dir_all(dir.path().join("beta")).unwrap();

    // Create in-memory SQLite with required schema
    let pool = sqlx::SqlitePool::connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::query(
        "CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL DEFAULT 'projects',
            tenant_id TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            is_archived INTEGER NOT NULL DEFAULT 0
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE tracked_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            relative_path TEXT,
            component TEXT,
            UNIQUE(watch_folder_id, file_path)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE project_components (
            component_id TEXT PRIMARY KEY,
            watch_folder_id TEXT NOT NULL,
            component_name TEXT NOT NULL,
            base_path TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'auto',
            patterns TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(watch_folder_id, component_name)
        )",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Insert a watch folder pointing to our temp dir
    let path_str = dir.path().to_string_lossy().to_string();
    sqlx::query("INSERT INTO watch_folders (watch_id, path) VALUES ('wf1', ?1)")
        .bind(&path_str)
        .execute(&pool)
        .await
        .unwrap();

    // Insert tracked files with NULL component
    for rel in &["alpha/src/lib.rs", "beta/src/main.rs", "README.md"] {
        let abs = format!("{}/{}", path_str, rel);
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, relative_path)
             VALUES ('wf1', ?1, ?2)",
        )
        .bind(&abs)
        .bind(rel)
        .execute(&pool)
        .await
        .unwrap();
    }

    let stats = backfill_components(&pool, 100, false, None).await.unwrap();
    assert_eq!(stats.folders_processed, 1);
    assert_eq!(stats.files_updated, 2); // alpha/src/lib.rs + beta/src/main.rs
    assert_eq!(stats.files_unmatched, 1); // README.md at root

    // Verify the component column was set
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

    // README.md should still be NULL
    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'README.md'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(row.0.is_none());

    // Components should be persisted
    let count: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM project_components WHERE watch_folder_id = 'wf1'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count.0, 2); // alpha + beta
}

#[tokio::test]
async fn test_backfill_components_force() {
    let dir = TempDir::new().unwrap();
    // Start with alpha + beta workspace
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"
[workspace]
members = ["alpha", "beta"]
"#,
    )
    .unwrap();
    fs::create_dir_all(dir.path().join("alpha")).unwrap();
    fs::create_dir_all(dir.path().join("beta")).unwrap();

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

    // Insert tracked files with NULL component
    for rel in &["alpha/src/lib.rs", "beta/src/main.rs"] {
        let abs = format!("{}/{}", path_str, rel);
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, relative_path)
             VALUES ('wf1', ?1, ?2)",
        )
        .bind(&abs)
        .bind(rel)
        .execute(&pool)
        .await
        .unwrap();
    }

    // First backfill (non-force) assigns components
    let stats = backfill_components(&pool, 100, false, None).await.unwrap();
    assert_eq!(stats.files_updated, 2);

    // Verify alpha/src/lib.rs has component "alpha"
    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'alpha/src/lib.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("alpha"));

    // Now rename workspace member: alpha -> gamma
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"
[workspace]
members = ["gamma", "beta"]
"#,
    )
    .unwrap();
    fs::create_dir_all(dir.path().join("gamma")).unwrap();

    // Update relative_path to simulate a moved file
    sqlx::query(
        "UPDATE tracked_files SET relative_path = 'gamma/src/lib.rs' \
         WHERE relative_path = 'alpha/src/lib.rs'",
    )
    .execute(&pool)
    .await
    .unwrap();

    // Non-force backfill should NOT update (component is already non-NULL)
    let stats = backfill_components(&pool, 100, false, None).await.unwrap();
    assert_eq!(stats.files_updated, 0);

    // Force backfill SHOULD reassign
    let stats = backfill_components(&pool, 100, true, None).await.unwrap();
    assert_eq!(stats.files_updated, 2);

    // Verify gamma/src/lib.rs now has component "gamma"
    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE relative_path = 'gamma/src/lib.rs'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("gamma"));

    // Stale "alpha" component should be cleaned up by force
    let count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM project_components \
         WHERE watch_folder_id = 'wf1' AND component_name = 'alpha'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(count.0, 0);

    // "gamma" and "beta" should exist
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

    // Both dirs have workspace definitions
    for dir in [&dir_a, &dir_b] {
        fs::write(
            dir.path().join("Cargo.toml"),
            r#"
[workspace]
members = ["crate-x"]
"#,
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("crate-x")).unwrap();
    }

    let pool = sqlx::SqlitePool::connect("sqlite::memory:")
        .await
        .unwrap();
    create_backfill_schema(&pool).await;

    let path_a = dir_a.path().to_string_lossy().to_string();
    let path_b = dir_b.path().to_string_lossy().to_string();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, tenant_id) VALUES ('wf_a', ?1, 'tenant_a')",
    )
    .bind(&path_a)
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, tenant_id) VALUES ('wf_b', ?1, 'tenant_b')",
    )
    .bind(&path_b)
    .execute(&pool)
    .await
    .unwrap();

    // Insert tracked files for both tenants
    let abs_a = format!("{}/crate-x/src/lib.rs", path_a);
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, relative_path)
         VALUES ('wf_a', ?1, 'crate-x/src/lib.rs')",
    )
    .bind(&abs_a)
    .execute(&pool)
    .await
    .unwrap();

    let abs_b = format!("{}/crate-x/src/lib.rs", path_b);
    sqlx::query(
        "INSERT INTO tracked_files (watch_folder_id, file_path, relative_path)
         VALUES ('wf_b', ?1, 'crate-x/src/lib.rs')",
    )
    .bind(&abs_b)
    .execute(&pool)
    .await
    .unwrap();

    // Backfill only tenant_a
    let stats = backfill_components(&pool, 100, false, Some("tenant_a"))
        .await
        .unwrap();
    assert_eq!(stats.folders_processed, 1);
    assert_eq!(stats.files_updated, 1);

    // tenant_a's file should have component assigned
    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE watch_folder_id = 'wf_a'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(row.0.as_deref(), Some("crate-x"));

    // tenant_b's file should still be NULL
    let row: (Option<String>,) = sqlx::query_as(
        "SELECT component FROM tracked_files WHERE watch_folder_id = 'wf_b'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(row.0.is_none());
}
