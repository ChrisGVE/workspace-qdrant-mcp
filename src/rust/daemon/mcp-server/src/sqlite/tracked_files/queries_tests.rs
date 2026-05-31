//! Hermetic tests for `tracked_files::queries`.
//!
//! Extracted into a sibling file (via `#[path]`) to keep `queries.rs` under
//! the 500-line limit.  Each test builds an ephemeral WAL database and opens
//! it READ-ONLY, exercising the same read path the live MCP server uses.

use super::*;
use rusqlite::{Connection, OpenFlags};
use tempfile::TempDir;

fn make_db(dir: &TempDir) -> (std::path::PathBuf, Connection) {
    let path = dir.path().join("state.db");
    let setup = Connection::open(&path).unwrap();
    setup
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
                 CREATE TABLE tracked_files (
                     file_id         TEXT PRIMARY KEY,
                     watch_folder_id TEXT NOT NULL,
                     base_point      TEXT,
                     relative_path   TEXT NOT NULL,
                     file_type       TEXT,
                     language        TEXT,
                     extension       TEXT,
                     is_test         INTEGER NOT NULL DEFAULT 0,
                     branches        TEXT NOT NULL DEFAULT '[]',
                     component       TEXT
                 );
                 CREATE TABLE watch_folders (
                     watch_id         TEXT PRIMARY KEY,
                     tenant_id        TEXT NOT NULL,
                     path             TEXT NOT NULL,
                     collection       TEXT NOT NULL,
                     parent_watch_id  TEXT,
                     submodule_path   TEXT,
                     git_remote_url   TEXT
                 );
                 CREATE TABLE project_components (
                     component_id    TEXT PRIMARY KEY,
                     watch_folder_id TEXT NOT NULL,
                     component_name  TEXT NOT NULL,
                     base_path       TEXT NOT NULL,
                     source          TEXT NOT NULL DEFAULT 'cargo',
                     patterns        TEXT
                 );",
        )
        .unwrap();
    drop(setup);
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    (path, conn)
}

fn insert_file(
    path: &std::path::Path,
    id: &str,
    wfid: &str,
    rel_path: &str,
    file_type: Option<&str>,
    lang: Option<&str>,
    ext: Option<&str>,
    is_test: i64,
    branches: &str,
) {
    let setup = Connection::open(path).unwrap();
    setup
        .execute(
            "INSERT INTO tracked_files
                 (file_id, watch_folder_id, relative_path, file_type, language, extension, is_test, branches)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![id, wfid, rel_path, file_type, lang, ext, is_test, branches],
        )
        .unwrap();
}

#[test]
fn none_conn_returns_empty() {
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w1".to_string(),
        ..Default::default()
    };
    assert!(list_tracked_files(None, &opts).is_empty());
    assert_eq!(count_tracked_files(None, &opts), 0);
    assert!(list_submodules(None, "w1").is_empty());
    assert!(list_project_components(None, "w1").is_empty());
}

#[test]
fn list_tracked_files_basic() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_file(
        &path,
        "f1",
        "w1",
        "src/main.rs",
        Some("code"),
        Some("rust"),
        Some("rs"),
        0,
        "[]",
    );
    insert_file(
        &path,
        "f2",
        "w1",
        "src/lib.rs",
        Some("code"),
        Some("rust"),
        Some("rs"),
        0,
        "[]",
    );
    insert_file(&path, "f3", "w2", "other.rs", None, None, None, 0, "[]");

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w1".to_string(),
        ..Default::default()
    };
    let result = list_tracked_files(Some(&conn), &opts);
    assert_eq!(result.len(), 2);
    // ORDER BY relative_path ASC
    assert_eq!(result[0].relative_path, "src/lib.rs");
    assert_eq!(result[1].relative_path, "src/main.rs");
}

#[test]
fn list_tracked_files_exclude_tests() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_file(&path, "f1", "w1", "src/main.rs", None, None, None, 0, "[]");
    insert_file(&path, "f2", "w1", "src/tests.rs", None, None, None, 1, "[]");

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w1".to_string(),
        include_tests: Some(false),
        ..Default::default()
    };
    let result = list_tracked_files(Some(&conn), &opts);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].relative_path, "src/main.rs");
}

#[test]
fn list_tracked_files_language_filter() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_file(&path, "f1", "w1", "a.rs", None, Some("rust"), None, 0, "[]");
    insert_file(
        &path,
        "f2",
        "w1",
        "b.py",
        None,
        Some("python"),
        None,
        0,
        "[]",
    );

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w1".to_string(),
        language: Some("rust".to_string()),
        ..Default::default()
    };
    let result = list_tracked_files(Some(&conn), &opts);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].relative_path, "a.rs");
}

#[test]
fn count_tracked_files_correct() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    for i in 0..5 {
        insert_file(
            &path,
            &format!("f{i}"),
            "w1",
            &format!("f{i}.rs"),
            None,
            None,
            None,
            0,
            "[]",
        );
    }
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w1".to_string(),
        ..Default::default()
    };
    assert_eq!(count_tracked_files(Some(&conn), &opts), 5);
}

#[test]
fn list_tracked_files_branch_filter() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_file(
        &path,
        "f1",
        "w1",
        "a.rs",
        None,
        None,
        None,
        0,
        r#"["main","feat"]"#,
    );
    insert_file(
        &path,
        "f2",
        "w1",
        "b.rs",
        None,
        None,
        None,
        0,
        r#"["feat"]"#,
    );
    insert_file(
        &path,
        "f3",
        "w1",
        "c.rs",
        None,
        None,
        None,
        0,
        r#"["other"]"#,
    );

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w1".to_string(),
        branch: Some("main".to_string()),
        ..Default::default()
    };
    let result = list_tracked_files(Some(&conn), &opts);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].relative_path, "a.rs");
}

#[test]
fn list_tracked_files_component_base_paths() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_file(
        &path,
        "f1",
        "w1",
        "src/rust/a.rs",
        None,
        None,
        None,
        0,
        "[]",
    );
    insert_file(&path, "f2", "w1", "src/ts/b.ts", None, None, None, 0, "[]");
    insert_file(
        &path,
        "f3",
        "w1",
        "docs/readme.md",
        None,
        None,
        None,
        0,
        "[]",
    );

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w1".to_string(),
        component_base_paths: Some(vec!["src/rust".to_string(), "src/ts".to_string()]),
        ..Default::default()
    };
    let result = list_tracked_files(Some(&conn), &opts);
    assert_eq!(result.len(), 2);
}

#[test]
fn list_submodules_basic() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    let setup = Connection::open(&path).unwrap();
    setup
        .execute(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, parent_watch_id, submodule_path, git_remote_url)
                 VALUES ('sub1','t1','/proj/sub','projects','parent1','vendor/lib','https://github.com/user/lib.git')",
            [],
        )
        .unwrap();
    drop(setup);

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let result = list_submodules(Some(&conn), "parent1");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].submodule_path, "vendor/lib");
    assert_eq!(result[0].repo_name, "lib");
}

#[test]
fn list_project_components_basic() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    let setup = Connection::open(&path).unwrap();
    setup
        .execute(
            "INSERT INTO project_components (component_id, watch_folder_id, component_name, base_path, source)
                 VALUES ('c1','w1','daemon','src/rust/daemon','cargo')",
            [],
        )
        .unwrap();
    drop(setup);

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let result = list_project_components(Some(&conn), "w1");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].component_name, "daemon");
    assert_eq!(result[0].base_path, "src/rust/daemon");
    assert_eq!(result[0].source, "cargo");
}

#[test]
fn extract_repo_name_https_url() {
    assert_eq!(
        extract_repo_name(Some("https://github.com/user/myrepo.git"), "vendor/myrepo"),
        "myrepo"
    );
}

#[test]
fn extract_repo_name_ssh_url() {
    assert_eq!(
        extract_repo_name(Some("git@github.com:user/myrepo.git"), "vendor/myrepo"),
        "myrepo"
    );
}

#[test]
fn extract_repo_name_fallback() {
    assert_eq!(extract_repo_name(None, "vendor/some-lib"), "some-lib");
}

#[test]
fn missing_table_returns_empty() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.db");
    let setup = Connection::open(&path).unwrap();
    setup.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
    drop(setup);
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let opts = ListTrackedFilesOptions {
        watch_folder_id: "w".to_string(),
        ..Default::default()
    };
    assert!(list_tracked_files(Some(&conn), &opts).is_empty());
    assert_eq!(count_tracked_files(Some(&conn), &opts), 0);
    assert!(list_submodules(Some(&conn), "w").is_empty());
    assert!(list_project_components(Some(&conn), "w").is_empty());
}
