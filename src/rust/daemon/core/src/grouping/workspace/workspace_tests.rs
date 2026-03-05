use super::*;
use sqlx::sqlite::SqlitePoolOptions;
use std::fs;
use tempfile::TempDir;

// ---- Parsing tests ---------------------------------------------------------

#[test]
fn test_parse_cargo_workspace_basic() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // Create workspace Cargo.toml
    fs::write(
        root.join("Cargo.toml"),
        r#"
[workspace]
members = [
    "crate-a",
    "crate-b",
]
"#,
    )
    .unwrap();

    // Create member directories
    fs::create_dir_all(root.join("crate-a")).unwrap();
    fs::create_dir_all(root.join("crate-b")).unwrap();

    let info = detect_cargo_workspace(&root.join("crate-a")).unwrap();
    assert_eq!(info.workspace_type, "cargo");
    assert_eq!(info.workspace_root, root);
    assert_eq!(info.members.len(), 2);
}

#[test]
fn test_parse_cargo_workspace_glob() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(
        root.join("Cargo.toml"),
        r#"
[workspace]
members = ["packages/*"]
"#,
    )
    .unwrap();

    fs::create_dir_all(root.join("packages/lib-a")).unwrap();
    fs::create_dir_all(root.join("packages/lib-b")).unwrap();

    let info = detect_cargo_workspace(&root.join("packages/lib-a")).unwrap();
    assert_eq!(info.members.len(), 2);
}

#[test]
fn test_parse_cargo_workspace_inline() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"a\", \"b\"]\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("a")).unwrap();
    fs::create_dir_all(root.join("b")).unwrap();

    let info = detect_cargo_workspace(&root.join("a")).unwrap();
    assert_eq!(info.members.len(), 2);
}

#[test]
fn test_no_workspace() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // Regular Cargo.toml (no workspace section)
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"solo\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    assert!(detect_cargo_workspace(root).is_none());
}

#[test]
fn test_parse_npm_workspace() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(
        root.join("package.json"),
        r#"{
  "name": "monorepo",
  "workspaces": ["packages/*"]
}"#,
    )
    .unwrap();

    fs::create_dir_all(root.join("packages/app")).unwrap();
    fs::create_dir_all(root.join("packages/lib")).unwrap();

    let info = detect_npm_workspace(&root.join("packages/app")).unwrap();
    assert_eq!(info.workspace_type, "npm");
    assert_eq!(info.members.len(), 2);
}

#[test]
fn test_parse_npm_workspace_object() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(
        root.join("package.json"),
        r#"{
  "name": "monorepo",
  "workspaces": { "packages": ["services/*"] }
}"#,
    )
    .unwrap();

    fs::create_dir_all(root.join("services/api")).unwrap();

    let info = detect_npm_workspace(&root.join("services/api")).unwrap();
    assert_eq!(info.workspace_type, "npm");
    assert_eq!(info.members.len(), 1);
}

#[test]
fn test_no_npm_workspace() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(
        root.join("package.json"),
        r#"{"name": "standalone", "version": "1.0.0"}"#,
    )
    .unwrap();

    assert!(detect_npm_workspace(root).is_none());
}

#[test]
fn test_parse_go_workspace() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(
        root.join("go.work"),
        "go 1.21\n\nuse (\n\t./cmd\n\t./pkg\n)\n",
    )
    .unwrap();

    fs::create_dir_all(root.join("cmd")).unwrap();
    fs::create_dir_all(root.join("pkg")).unwrap();

    let info = detect_go_workspace(&root.join("cmd")).unwrap();
    assert_eq!(info.workspace_type, "go");
    assert_eq!(info.members.len(), 2);
}

#[test]
fn test_parse_go_workspace_single_use() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(root.join("go.work"), "go 1.21\n\nuse ./app\n").unwrap();
    fs::create_dir_all(root.join("app")).unwrap();

    let info = detect_go_workspace(&root.join("app")).unwrap();
    assert_eq!(info.members.len(), 1);
}

#[test]
fn test_no_go_workspace() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // go.mod only (no go.work) -> not a workspace
    fs::write(root.join("go.mod"), "module example.com/myapp\n\ngo 1.21\n").unwrap();

    assert!(detect_go_workspace(root).is_none());
}

#[test]
fn test_workspace_id_deterministic() {
    let path = Path::new("/home/user/projects/my-workspace");
    let id1 = generate_workspace_id(path);
    let id2 = generate_workspace_id(path);
    assert_eq!(id1, id2);
}

#[test]
fn test_workspace_id_unique() {
    let id1 = generate_workspace_id(Path::new("/a"));
    let id2 = generate_workspace_id(Path::new("/b"));
    assert_ne!(id1, id2);
}

#[test]
fn test_extract_toml_array_strings() {
    let result = extract_toml_array_strings(r#"["foo", "bar/baz"]"#);
    assert_eq!(result, vec!["foo", "bar/baz"]);
}

// ---- SQLite integration tests ----------------------------------------------

async fn setup_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();

    // Minimal watch_folders table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS watch_folders (
            watch_id TEXT PRIMARY KEY,
            folder_path TEXT NOT NULL,
            watch_type TEXT NOT NULL DEFAULT 'project',
            tenant_id TEXT NOT NULL,
            git_remote_url TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        "#,
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(schema::CREATE_PROJECT_GROUPS_SQL)
        .execute(&pool)
        .await
        .unwrap();
    for idx in schema::CREATE_PROJECT_GROUPS_INDEXES_SQL {
        sqlx::query(idx).execute(&pool).await.unwrap();
    }

    pool
}

#[tokio::test]
async fn test_compute_workspace_groups_cargo() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // Create a Cargo workspace
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"app\", \"lib\"]\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("app")).unwrap();
    fs::create_dir_all(root.join("lib")).unwrap();

    let pool = setup_pool().await;

    // Register two projects in the workspace
    let app_path = root.join("app").to_string_lossy().to_string();
    let lib_path = root.join("lib").to_string_lossy().to_string();

    sqlx::query("INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)")
        .bind("w1")
        .bind(&app_path)
        .bind("tenant-app")
        .execute(&pool)
        .await
        .unwrap();

    sqlx::query("INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)")
        .bind("w2")
        .bind(&lib_path)
        .bind("tenant-lib")
        .execute(&pool)
        .await
        .unwrap();

    let groups = compute_workspace_groups(&pool).await.unwrap();
    assert_eq!(groups, 1);

    let members = schema::get_group_members(&pool, "tenant-app")
        .await
        .unwrap();
    assert_eq!(members.len(), 2);
    assert!(members.contains(&"tenant-app".to_string()));
    assert!(members.contains(&"tenant-lib".to_string()));
}

#[tokio::test]
async fn test_update_single_project_workspace() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"svc-a\", \"svc-b\"]\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("svc-a")).unwrap();
    fs::create_dir_all(root.join("svc-b")).unwrap();

    let pool = setup_pool().await;

    // Register svc-a first
    let svc_a_path = root.join("svc-a").to_string_lossy().to_string();
    sqlx::query("INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)")
        .bind("w1")
        .bind(&svc_a_path)
        .bind("tenant-a")
        .execute(&pool)
        .await
        .unwrap();

    // Then register svc-b and update groups
    let svc_b_path = root.join("svc-b").to_string_lossy().to_string();
    sqlx::query("INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)")
        .bind("w2")
        .bind(&svc_b_path)
        .bind("tenant-b")
        .execute(&pool)
        .await
        .unwrap();

    let added = update_project_workspace_group(&pool, "tenant-b", &root.join("svc-b"))
        .await
        .unwrap();
    assert!(added);

    // Both should be in the same group
    let members = schema::get_group_members(&pool, "tenant-a").await.unwrap();
    assert_eq!(members.len(), 2);
}

#[tokio::test]
async fn test_no_workspace_no_group() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // No workspace manifests
    let pool = setup_pool().await;

    let path = root.to_string_lossy().to_string();
    sqlx::query("INSERT INTO watch_folders (watch_id, folder_path, tenant_id) VALUES (?, ?, ?)")
        .bind("w1")
        .bind(&path)
        .bind("tenant-solo")
        .execute(&pool)
        .await
        .unwrap();

    let groups = compute_workspace_groups(&pool).await.unwrap();
    assert_eq!(groups, 0);
}
