//! Tests for `session::project_detect`.

use std::fs;
use std::path::PathBuf;

use tempfile::TempDir;

use super::{detect_branch, detect_project, find_git_root, find_project_root, get_git_remote_url};
use crate::sqlite::manager::StateManager;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Create a minimal git repository in a tempdir.
fn make_git_repo(dir: &TempDir) -> PathBuf {
    let root = dir.path().to_path_buf();
    let git_dir = root.join(".git");
    fs::create_dir_all(git_dir.join("refs").join("heads")).unwrap();
    fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
    root
}

/// Write a `.git/config` with an origin remote.
fn write_git_config_with_origin(git_dir: &std::path::Path, url: &str) {
    let content = format!(
        "[core]\n\trepositoryformatversion = 0\n[remote \"origin\"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
    );
    fs::write(git_dir.join("config"), content).unwrap();
}

// ─────────────────────────────────────────────────────────────────────────────
// find_git_root
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn find_git_root_finds_immediate_git_dir() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    assert_eq!(find_git_root(&root), Some(root.clone()));
}

#[test]
fn find_git_root_finds_ancestor() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    let subdir = root.join("src").join("module");
    fs::create_dir_all(&subdir).unwrap();
    assert_eq!(find_git_root(&subdir), Some(root.clone()));
}

#[test]
fn find_git_root_returns_none_outside_repo() {
    let dir = TempDir::new().unwrap();
    // No .git directory — should return None.
    assert_eq!(find_git_root(dir.path()), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// find_project_root
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn find_project_root_finds_git_marker() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    let subdir = root.join("deep").join("path");
    fs::create_dir_all(&subdir).unwrap();
    assert_eq!(find_project_root(&subdir), Some(root.clone()));
}

#[test]
fn find_project_root_finds_cargo_toml() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().to_path_buf();
    fs::write(root.join("Cargo.toml"), "[package]\n").unwrap();
    let subdir = root.join("src");
    fs::create_dir_all(&subdir).unwrap();
    assert_eq!(find_project_root(&subdir), Some(root.clone()));
}

#[test]
fn find_project_root_returns_none_when_no_marker() {
    let dir = TempDir::new().unwrap();
    // Plain directory with no markers.
    assert_eq!(find_project_root(dir.path()), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// detect_branch
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn detect_branch_symbolic_ref() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    // make_git_repo writes "ref: refs/heads/main"
    assert_eq!(detect_branch(&root), "main");
}

#[test]
fn detect_branch_feature_branch() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    fs::write(
        root.join(".git").join("HEAD"),
        "ref: refs/heads/feat/my-feature\n",
    )
    .unwrap();
    assert_eq!(detect_branch(&root), "feat/my-feature");
}

#[test]
fn detect_branch_detached_head() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    let sha = "abcdef1234567890abcdef1234567890abcdef12";
    fs::write(root.join(".git").join("HEAD"), sha).unwrap();
    // Returns first 8 chars of SHA
    assert_eq!(detect_branch(&root), &sha[..8]);
}

#[test]
fn detect_branch_no_git_dir() {
    let dir = TempDir::new().unwrap();
    assert_eq!(detect_branch(dir.path()), "default");
}

#[test]
fn detect_branch_unreadable_head() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    // Write garbage format (not a ref or SHA)
    fs::write(root.join(".git").join("HEAD"), "not-a-valid-head\n").unwrap();
    assert_eq!(detect_branch(&root), "default");
}

#[test]
fn detect_branch_from_subdirectory() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    fs::write(root.join(".git").join("HEAD"), "ref: refs/heads/develop\n").unwrap();
    let subdir = root.join("src");
    fs::create_dir_all(&subdir).unwrap();
    assert_eq!(detect_branch(&subdir), "develop");
}

// ─────────────────────────────────────────────────────────────────────────────
// get_git_remote_url
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn get_git_remote_url_returns_origin() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    write_git_config_with_origin(&root.join(".git"), "git@github.com:user/repo.git");
    assert_eq!(
        get_git_remote_url(&root),
        Some("git@github.com:user/repo.git".to_string())
    );
}

#[test]
fn get_git_remote_url_https_remote() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    write_git_config_with_origin(&root.join(".git"), "https://github.com/user/repo.git");
    assert_eq!(
        get_git_remote_url(&root),
        Some("https://github.com/user/repo.git".to_string())
    );
}

#[test]
fn get_git_remote_url_no_remote_returns_none() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    // No .git/config → None
    assert_eq!(get_git_remote_url(&root), None);
}

#[test]
fn get_git_remote_url_config_without_origin_returns_none() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    fs::write(
        root.join(".git").join("config"),
        "[core]\n\trepositoryformatversion = 0\n[remote \"upstream\"]\n\turl = git@github.com:upstream/repo.git\n",
    )
    .unwrap();
    assert_eq!(get_git_remote_url(&root), None);
}

#[test]
fn get_git_remote_url_multiple_remotes_returns_origin() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    fs::write(
        root.join(".git").join("config"),
        "[core]\n\trepositoryformatversion = 0\n\
         [remote \"upstream\"]\n\turl = git@github.com:upstream/repo.git\n\
         [remote \"origin\"]\n\turl = git@github.com:user/repo.git\n",
    )
    .unwrap();
    assert_eq!(
        get_git_remote_url(&root),
        Some("git@github.com:user/repo.git".to_string())
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// detect_project
// ─────────────────────────────────────────────────────────────────────────────

/// Build a minimal SQLite state.db with watch_folders entries.
fn make_state_db_with_project(
    dir: &TempDir,
    tenant_id: &str,
    project_path: &str,
) -> std::path::PathBuf {
    let db_path = dir.path().join("state.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE watch_folders (
             tenant_id TEXT NOT NULL,
             path TEXT NOT NULL,
             collection TEXT NOT NULL DEFAULT 'projects'
         )",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES (?1, ?2, 'projects')",
        rusqlite::params![tenant_id, project_path],
    )
    .unwrap();
    drop(conn);
    db_path
}

#[test]
fn detect_project_finds_root_and_branch() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);

    let db_path = make_state_db_with_project(&dir, "tid_abc123", root.to_str().unwrap());
    let state_mgr = StateManager::open_at(&db_path);

    let info = detect_project(&root, &state_mgr).unwrap();

    assert_eq!(info.project_path, root);
    assert_eq!(info.branch, "main");
    assert_eq!(info.project_id, Some("tid_abc123".to_string()));
}

#[test]
fn detect_project_no_project_id_when_not_in_db() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);

    // State manager pointing at non-existent db → degraded
    let state_mgr = StateManager::open_at("/nonexistent/state.db");

    let info = detect_project(&root, &state_mgr).unwrap();
    assert_eq!(info.project_id, None);
}

#[test]
fn detect_project_includes_git_remote() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    write_git_config_with_origin(&root.join(".git"), "git@github.com:user/repo.git");

    let state_mgr = StateManager::open_at("/nonexistent/state.db");
    let info = detect_project(&root, &state_mgr).unwrap();

    assert_eq!(
        info.git_remote,
        Some("git@github.com:user/repo.git".to_string())
    );
}

#[test]
fn detect_project_no_git_remote_when_missing() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);

    let state_mgr = StateManager::open_at("/nonexistent/state.db");
    let info = detect_project(&root, &state_mgr).unwrap();
    assert_eq!(info.git_remote, None);
}

#[test]
fn detect_project_uses_default_branch_outside_git() {
    let dir = TempDir::new().unwrap();
    // No .git directory — place a Cargo.toml so find_project_root finds something
    let root = dir.path().to_path_buf();
    fs::write(root.join("Cargo.toml"), "[package]\n").unwrap();

    let state_mgr = StateManager::open_at("/nonexistent/state.db");
    let info = detect_project(&root, &state_mgr).unwrap();
    assert_eq!(info.branch, "default");
}

#[test]
fn detect_project_from_subdirectory() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir);
    let subdir = root.join("src").join("module");
    fs::create_dir_all(&subdir).unwrap();

    let state_mgr = StateManager::open_at("/nonexistent/state.db");
    let info = detect_project(&subdir, &state_mgr).unwrap();

    // Should find the git root, not the subdir
    assert_eq!(info.project_path, root);
    assert_eq!(info.branch, "main");
}

/// `detect_project` resolves `project_id` cwd-direct (GitHub #84): with a deeper
/// markerless registered project, a cwd under it must resolve to the DEEPER
/// tenant, while `project_path` still reports the marker root. Looking up by the
/// marker root (the old behavior) would return the wrong ancestor tenant.
#[test]
fn detect_project_resolves_project_id_cwd_direct_for_markerless_nested() {
    let dir = TempDir::new().unwrap();
    let root = make_git_repo(&dir); // ancestor has a `.git` marker
    let nest = root.join("sandbox"); // markerless nested registered project
    let cwd = nest.join("src");
    fs::create_dir_all(&cwd).unwrap();

    let db_path = dir.path().join("state.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE watch_folders (
             tenant_id TEXT NOT NULL,
             path TEXT NOT NULL,
             collection TEXT NOT NULL DEFAULT 'projects'
         )",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_ROOT', ?1, 'projects')",
        rusqlite::params![root.to_str().unwrap()],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (tenant_id, path, collection) VALUES ('T_NEST', ?1, 'projects')",
        rusqlite::params![nest.to_str().unwrap()],
    )
    .unwrap();
    drop(conn);

    let state_mgr = StateManager::open_at(&db_path);
    let info = detect_project(&cwd, &state_mgr).unwrap();

    // project_id is cwd-direct → deeper tenant; project_path is the marker root.
    assert_eq!(info.project_id.as_deref(), Some("T_NEST"));
    assert_eq!(info.project_path, root);
}
