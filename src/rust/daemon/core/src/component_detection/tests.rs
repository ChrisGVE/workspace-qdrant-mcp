use super::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_path_to_component_id() {
    assert_eq!(path_to_component_id("daemon/core"), "daemon.core");
    assert_eq!(path_to_component_id("cli"), "cli");
    assert_eq!(path_to_component_id("src/typescript/mcp"), "src.typescript.mcp");
    assert_eq!(path_to_component_id("trailing/"), "trailing");
    assert_eq!(path_to_component_id("/leading"), "leading");
}

#[test]
fn test_parse_cargo_members_basic() {
    let content = r#"
[workspace]
resolver = "2"
members = [
    "daemon/core",
    "daemon/grpc",
    "cli",
]
"#;
    let members = parse_cargo_members(content);
    assert_eq!(members, vec!["daemon/core", "daemon/grpc", "cli"]);
}

#[test]
fn test_parse_cargo_members_inline() {
    let content = r#"
[workspace]
members = ["a", "b"]
"#;
    let members = parse_cargo_members(content);
    assert_eq!(members, vec!["a", "b"]);
}

#[test]
fn test_parse_cargo_members_with_comments() {
    let content = r#"
[workspace]
members = [
    "a",
    # "commented-out",
    "b",
]
"#;
    let members = parse_cargo_members(content);
    assert_eq!(members, vec!["a", "b"]);
}

#[test]
fn test_parse_cargo_members_no_workspace() {
    let content = r#"
[package]
name = "my-crate"
"#;
    assert!(parse_cargo_members(content).is_empty());
}

#[test]
fn test_file_matches_component() {
    let comp = ComponentInfo {
        id: "daemon.core".into(),
        base_path: "daemon/core".into(),
        patterns: vec!["daemon/core/**".into()],
        source: ComponentSource::Cargo,
    };

    assert!(file_matches_component("daemon/core/src/lib.rs", &comp));
    assert!(file_matches_component("daemon/core", &comp));
    assert!(!file_matches_component("daemon/grpc/src/lib.rs", &comp));
    assert!(!file_matches_component("daemon/core_extra/foo.rs", &comp));
}

#[test]
fn test_component_matches_filter() {
    assert!(component_matches_filter("daemon.core", "daemon.core"));
    assert!(component_matches_filter("daemon.core", "daemon"));
    assert!(!component_matches_filter("daemon.core", "cli"));
    assert!(!component_matches_filter("daemon", "daemon.core"));
}

#[test]
fn test_assign_component_most_specific() {
    let mut components = ComponentMap::new();
    components.insert(
        "daemon".into(),
        ComponentInfo {
            id: "daemon".into(),
            base_path: "daemon".into(),
            patterns: vec!["daemon/**".into()],
            source: ComponentSource::Cargo,
        },
    );
    components.insert(
        "daemon.core".into(),
        ComponentInfo {
            id: "daemon.core".into(),
            base_path: "daemon/core".into(),
            patterns: vec!["daemon/core/**".into()],
            source: ComponentSource::Cargo,
        },
    );

    let result = assign_component("daemon/core/src/lib.rs", &components);
    assert_eq!(result.unwrap().id, "daemon.core");

    let result = assign_component("daemon/grpc/src/lib.rs", &components);
    assert_eq!(result.unwrap().id, "daemon");

    let result = assign_component("cli/src/main.rs", &components);
    assert!(result.is_none());
}

#[test]
fn test_detect_cargo_workspace() {
    let dir = TempDir::new().unwrap();
    let cargo_toml = dir.path().join("Cargo.toml");
    fs::write(
        &cargo_toml,
        r#"
[workspace]
resolver = "2"
members = ["crate-a", "crate-b"]
"#,
    )
    .unwrap();

    // Create the member directories (not strictly needed for detection,
    // but validates the path computation)
    fs::create_dir_all(dir.path().join("crate-a")).unwrap();
    fs::create_dir_all(dir.path().join("crate-b")).unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);
    assert!(components.contains_key("crate-a"));
    assert!(components.contains_key("crate-b"));
    assert_eq!(components["crate-a"].source, ComponentSource::Cargo);
}

#[test]
fn test_detect_cargo_workspace_nested() {
    let dir = TempDir::new().unwrap();
    // No root Cargo.toml, but one in src/rust/
    let nested = dir.path().join("src/rust");
    fs::create_dir_all(&nested).unwrap();
    fs::write(
        nested.join("Cargo.toml"),
        r#"
[workspace]
members = ["daemon/core", "cli"]
"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);

    // Component IDs are based on member path, not full path
    assert!(components.contains_key("daemon.core"));
    assert!(components.contains_key("cli"));

    // base_path should include the relative prefix
    assert_eq!(components["daemon.core"].base_path, "src/rust/daemon/core");
    assert_eq!(components["cli"].base_path, "src/rust/cli");
}

#[test]
fn test_detect_npm_workspace() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("package.json"),
        r#"{"workspaces": ["packages/ui", "packages/api"]}"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);
    assert!(components.contains_key("packages.ui"));
    assert!(components.contains_key("packages.api"));
    assert_eq!(components["packages.ui"].source, ComponentSource::Npm);
}

#[test]
fn test_detect_npm_workspace_glob() {
    let dir = TempDir::new().unwrap();
    let pkgs = dir.path().join("packages");
    fs::create_dir_all(pkgs.join("alpha")).unwrap();
    fs::create_dir_all(pkgs.join("beta")).unwrap();
    // Create a file that should be ignored (not a dir)
    fs::write(pkgs.join("README.md"), "").unwrap();

    fs::write(
        dir.path().join("package.json"),
        r#"{"workspaces": ["packages/*"]}"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    assert_eq!(components.len(), 2);
    assert!(components.contains_key("packages.alpha"));
    assert!(components.contains_key("packages.beta"));
}

#[test]
fn test_detect_directory_fallback() {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join("src")).unwrap();
    fs::create_dir_all(dir.path().join("tests")).unwrap();
    fs::create_dir_all(dir.path().join(".git")).unwrap();
    fs::create_dir_all(dir.path().join("node_modules")).unwrap();
    fs::write(dir.path().join("README.md"), "").unwrap();

    let components = detect_components(dir.path());
    assert!(components.contains_key("src"));
    assert!(components.contains_key("tests"));
    assert!(!components.contains_key(".git"));
    assert!(!components.contains_key("node_modules"));
    assert_eq!(components["src"].source, ComponentSource::Directory);
}

#[test]
fn test_cargo_takes_priority_over_npm() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"
[workspace]
members = ["shared"]
"#,
    )
    .unwrap();
    fs::write(
        dir.path().join("package.json"),
        r#"{"workspaces": ["shared", "web"]}"#,
    )
    .unwrap();

    let components = detect_components(dir.path());
    // "shared" should be Cargo (takes priority)
    assert_eq!(components["shared"].source, ComponentSource::Cargo);
    // "web" should be npm (no conflict)
    assert!(components.contains_key("web"));
    assert_eq!(components["web"].source, ComponentSource::Npm);
}

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

/// Helper to create the in-memory test schema for backfill tests.
async fn create_backfill_schema(pool: &sqlx::SqlitePool) {
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
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
    .await
    .unwrap();
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
