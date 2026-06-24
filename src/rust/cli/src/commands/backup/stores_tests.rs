//! Tests for store discovery and VACUUM INTO copy (AC-F20.1).

use std::fs;
use tempfile::TempDir;

use super::{discover_stores, total_store_bytes, vacuum_into};

fn make_sqlite_db(path: &std::path::Path) {
    let conn = rusqlite::Connection::open(path).expect("open");
    conn.execute_batch("CREATE TABLE t (x INTEGER);")
        .expect("create table");
}

/// AC-F20.1: discover_stores finds state.db.
#[test]
fn t_f20_stores_discovers_state_db() {
    let dir = TempDir::new().expect("tempdir");
    let state = dir.path().join("state.db");
    make_sqlite_db(&state);

    let stores = discover_stores(dir.path());
    assert!(
        stores.iter().any(|s| s.rel_path == "state.db"),
        "expected state.db in discovered stores: {:?}",
        stores.iter().map(|s| &s.rel_path).collect::<Vec<_>>()
    );
}

/// AC-F20.1: discover_stores finds per-project store.db files.
#[test]
fn t_f20_stores_discovers_project_stores() {
    let dir = TempDir::new().expect("tempdir");
    let proj_dir = dir.path().join("projects").join("tenant123");
    fs::create_dir_all(&proj_dir).expect("mkdir");
    make_sqlite_db(&proj_dir.join("store.db"));

    let stores = discover_stores(dir.path());
    assert!(
        stores
            .iter()
            .any(|s| s.rel_path == "projects/tenant123/store.db"),
        "expected project store in discovered stores"
    );
    let proj = stores
        .iter()
        .find(|s| s.rel_path == "projects/tenant123/store.db")
        .unwrap();
    assert_eq!(proj.tenant_id.as_deref(), Some("tenant123"));
}

/// AC-F20.1: discover_stores finds global/store.db and libraries/store.db.
#[test]
fn t_f20_stores_discovers_global_and_libraries() {
    let dir = TempDir::new().expect("tempdir");

    let global_dir = dir.path().join("global");
    fs::create_dir_all(&global_dir).expect("mkdir global");
    make_sqlite_db(&global_dir.join("store.db"));

    let lib_dir = dir.path().join("libraries");
    fs::create_dir_all(&lib_dir).expect("mkdir libraries");
    make_sqlite_db(&lib_dir.join("store.db"));

    let stores = discover_stores(dir.path());
    assert!(
        stores.iter().any(|s| s.rel_path == "global/store.db"),
        "expected global/store.db"
    );
    assert!(
        stores.iter().any(|s| s.rel_path == "libraries/store.db"),
        "expected libraries/store.db"
    );
}

/// AC-F20.1: absent optional stores are silently skipped.
#[test]
fn t_f20_stores_absent_stores_skipped() {
    let dir = TempDir::new().expect("tempdir");
    // No files at all -- discover_stores must return an empty vec, not error.
    let stores = discover_stores(dir.path());
    assert!(stores.is_empty(), "expected empty when no stores present");
}

/// AC-F20.1: total_store_bytes sums correctly.
#[test]
fn t_f20_stores_total_bytes_sums_sizes() {
    let dir = TempDir::new().expect("tempdir");
    let state = dir.path().join("state.db");
    make_sqlite_db(&state);

    let stores = discover_stores(dir.path());
    let total = total_store_bytes(&stores);
    let actual = fs::metadata(&state).unwrap().len();
    assert_eq!(total, actual, "total_store_bytes must match file size");
}

/// AC-F20.1: vacuum_into produces a readable copy of the source database.
#[test]
fn t_f20_stores_vacuum_into_copies_db() {
    let dir = TempDir::new().expect("tempdir");
    let src = dir.path().join("source.db");
    let dst = dir.path().join("copy.db");

    let conn = rusqlite::Connection::open(&src).expect("open src");
    conn.execute_batch(
        "CREATE TABLE kv (k TEXT, v TEXT); INSERT INTO kv VALUES ('hello', 'world');",
    )
    .expect("populate");
    drop(conn);

    vacuum_into(&src, &dst).expect("vacuum_into");

    // The copy must be readable and contain the row.
    let copy_conn = rusqlite::Connection::open(&dst).expect("open copy");
    let val: String = copy_conn
        .query_row("SELECT v FROM kv WHERE k = 'hello'", [], |r| r.get(0))
        .expect("query copy");
    assert_eq!(val, "world");
}
