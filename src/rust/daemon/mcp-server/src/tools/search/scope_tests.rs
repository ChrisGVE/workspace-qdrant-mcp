//! Tests for the SQLite-bound base-point resolution adapter.
//!
//! The SQLite-free scope helpers (decay, filter, path-segment matching,
//! degraded-reason) are tested in `wqm_client::search::scope`.

use super::*;
use crate::tools::search::types::SearchScope;

// ── resolve_base_points ─────────────────────────────────────────────────────

/// Build a temp state.db with a watch folder + the given base points.
fn make_base_points_db(
    dir: &tempfile::TempDir,
    tenant: &str,
    watch_id: &str,
    base_points: &[&str],
) -> std::path::PathBuf {
    let db_path = dir.path().join("state.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE watch_folders (watch_id TEXT, tenant_id TEXT, collection TEXT, parent_watch_id TEXT);
         CREATE TABLE tracked_files (base_point TEXT, watch_folder_id TEXT);",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO watch_folders (watch_id, tenant_id, collection, parent_watch_id) VALUES (?1, ?2, 'projects', NULL)",
        rusqlite::params![watch_id, tenant],
    )
    .unwrap();
    for bp in base_points {
        conn.execute(
            "INSERT INTO tracked_files (base_point, watch_folder_id) VALUES (?1, ?2)",
            rusqlite::params![bp, watch_id],
        )
        .unwrap();
    }
    drop(conn);
    db_path
}

#[test]
fn base_points_under_cap_returned() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = make_base_points_db(&dir, "T", "w1", &["/a", "/b"]);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, count) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/a/x"),
    );
    let mut got = bp.unwrap();
    got.sort();
    assert_eq!(got, vec!["/a".to_string(), "/b".to_string()]);
    assert!(!degraded);
    assert_eq!(count, None);
}

#[test]
fn base_points_non_project_scope_is_none() {
    let dir = tempfile::TempDir::new().unwrap();
    let db = make_base_points_db(&dir, "T", "w1", &["/a"]);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, _, _) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::All,
        std::path::Path::new("/a"),
    );
    assert_eq!(bp, None);
}

#[test]
fn base_points_over_cap_narrows_to_primary() {
    let dir = tempfile::TempDir::new().unwrap();
    // 501 distinct base points; one is a prefix of cwd.
    let owned: Vec<String> = (0..501).map(|i| format!("/bp/{i}")).collect();
    let mut refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    refs.push("/primary/here");
    let db = make_base_points_db(&dir, "T", "w1", &refs);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, _) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/primary/here/sub"),
    );
    assert_eq!(bp, Some(vec!["/primary/here".to_string()]));
    assert!(!degraded);
}

#[test]
fn base_points_over_cap_no_primary_degrades() {
    let dir = tempfile::TempDir::new().unwrap();
    let owned: Vec<String> = (0..520).map(|i| format!("/bp/{i}")).collect();
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    let db = make_base_points_db(&dir, "T", "w1", &refs);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, count) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/unrelated/cwd"),
    );
    assert_eq!(bp, None);
    assert!(degraded);
    assert_eq!(count, Some(520));
}

#[test]
fn base_points_over_cap_sibling_prefix_does_not_false_match() {
    // cwd `/repo-a/...` must NOT match the sibling base point `/repo` (raw
    // string prefix would). With no real prefix match → degrade (Finding #3).
    let dir = tempfile::TempDir::new().unwrap();
    let owned: Vec<String> = (0..520).map(|i| format!("/bp/{i}")).collect();
    let mut refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    refs.push("/repo");
    let db = make_base_points_db(&dir, "T", "w1", &refs);
    let conn = rusqlite::Connection::open(&db).unwrap();
    let (bp, degraded, _) = resolve_base_points(
        Some(&conn),
        Some("T"),
        SearchScope::Project,
        std::path::Path::new("/repo-a/src"),
    );
    assert_eq!(bp, None);
    assert!(degraded);
}
