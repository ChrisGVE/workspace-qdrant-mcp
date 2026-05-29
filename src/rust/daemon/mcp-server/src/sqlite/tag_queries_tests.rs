//! Tests for tag_queries — split from tag_queries.rs to satisfy the 500-line limit.

use super::*;
use rusqlite::{Connection, OpenFlags};
use tempfile::TempDir;

fn make_db(dir: &TempDir) -> (std::path::PathBuf, Connection) {
    let path = dir.path().join("state.db");
    let setup = Connection::open(&path).unwrap();
    setup
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
             CREATE TABLE tags (
                 tag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                 doc_id     TEXT NOT NULL,
                 tag        TEXT NOT NULL,
                 tag_type   TEXT NOT NULL,
                 score      REAL NOT NULL DEFAULT 1.0,
                 basket_id  TEXT,
                 collection TEXT NOT NULL,
                 tenant_id  TEXT
             );
             CREATE TABLE keyword_baskets (
                 basket_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                 tag_id        INTEGER NOT NULL,
                 keywords_json TEXT NOT NULL DEFAULT '[]',
                 tenant_id     TEXT
             );
             CREATE TABLE canonical_tags (
                 canonical_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                 canonical_name TEXT NOT NULL,
                 level          INTEGER NOT NULL DEFAULT 0,
                 parent_id      INTEGER,
                 collection     TEXT NOT NULL,
                 tenant_id      TEXT
             );",
        )
        .unwrap();
    drop(setup);
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    (path, conn)
}

fn insert_tag(
    path: &std::path::Path,
    doc_id: &str,
    tag: &str,
    tag_type: &str,
    score: f64,
    collection: &str,
    tenant_id: Option<&str>,
) -> i64 {
    let setup = Connection::open(path).unwrap();
    setup
        .execute(
            "INSERT INTO tags (doc_id, tag, tag_type, score, collection, tenant_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![doc_id, tag, tag_type, score, collection, tenant_id],
        )
        .unwrap();
    setup.last_insert_rowid()
}

#[test]
fn none_conn_returns_empty() {
    assert!(get_matching_tags(None, "rust", "projects", None).is_empty());
    assert!(get_keyword_baskets_for_tags(None, &[1, 2]).is_empty());
    assert!(list_tags(None, "projects", None, 50).is_empty());
    assert!(get_tag_hierarchy(None, "projects", None).is_empty());
}

#[test]
fn tokenize_query_basic() {
    let tokens = tokenize_query("Hello World foo");
    // "foo" has length 3 — included; all lowercased
    assert!(tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
    assert!(tokens.contains(&"foo".to_string()));
}

#[test]
fn tokenize_query_strips_short_tokens() {
    let tokens = tokenize_query("ab rust");
    assert!(!tokens.contains(&"ab".to_string())); // length 2 — filtered
    assert!(tokens.contains(&"rust".to_string()));
}

#[test]
fn get_matching_tags_finds_concept_tags() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_tag(
        &path,
        "doc1",
        "rust-async",
        "concept",
        0.9,
        "projects",
        None,
    );
    insert_tag(&path, "doc2", "python", "concept", 0.5, "projects", None);
    insert_tag(&path, "doc3", "rust-io", "keyword", 0.8, "projects", None); // wrong type

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let result = get_matching_tags(Some(&conn), "rust", "projects", None);
    // only concept tags matching "rust" — rust-io is tag_type='keyword', excluded
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tag, "rust-async");
}

#[test]
fn get_matching_tags_tenant_filter() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_tag(
        &path,
        "d1",
        "async-rust",
        "concept",
        0.9,
        "projects",
        Some("t1"),
    );
    insert_tag(
        &path,
        "d2",
        "async-python",
        "concept",
        0.8,
        "projects",
        Some("t2"),
    );

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let result = get_matching_tags(Some(&conn), "async", "projects", Some("t1"));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tag, "async-rust");
}

#[test]
fn get_matching_tags_empty_query_returns_empty() {
    let dir = TempDir::new().unwrap();
    let (_, conn) = make_db(&dir);
    let result = get_matching_tags(Some(&conn), "", "projects", None);
    assert!(result.is_empty());
}

#[test]
fn get_matching_tags_short_token_returns_empty() {
    let dir = TempDir::new().unwrap();
    let (_, conn) = make_db(&dir);
    // "ab" is only 2 chars — filtered out → empty tokens → empty result
    let result = get_matching_tags(Some(&conn), "ab", "projects", None);
    assert!(result.is_empty());
}

#[test]
fn get_keyword_baskets_for_tags_returns_keywords() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    let setup = Connection::open(&path).unwrap();
    setup
        .execute(
            "INSERT INTO keyword_baskets (tag_id, keywords_json) VALUES (42, '[\"foo\",\"bar\"]')",
            [],
        )
        .unwrap();
    drop(setup);

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let result = get_keyword_baskets_for_tags(Some(&conn), &[42]);
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].keywords,
        vec!["foo".to_string(), "bar".to_string()]
    );
}

#[test]
fn get_keyword_baskets_empty_ids_returns_empty() {
    let dir = TempDir::new().unwrap();
    let (_, conn) = make_db(&dir);
    let result = get_keyword_baskets_for_tags(Some(&conn), &[]);
    assert!(result.is_empty());
}

#[test]
fn list_tags_groups_by_tag() {
    let dir = TempDir::new().unwrap();
    let (path, conn) = make_db(&dir);
    drop(conn);
    insert_tag(&path, "d1", "rust", "concept", 0.9, "projects", None);
    insert_tag(&path, "d2", "rust", "concept", 0.7, "projects", None);
    insert_tag(&path, "d3", "python", "concept", 0.8, "projects", None);

    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let result = list_tags(Some(&conn), "projects", None, 50);
    assert_eq!(result.len(), 2);
    // "rust" has 2 docs, should be first (ORDER BY doc_count DESC)
    assert_eq!(result[0].tag, "rust");
    assert_eq!(result[0].doc_count, 2);
}

#[test]
fn missing_table_returns_empty_for_all() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.db");
    let setup = Connection::open(&path).unwrap();
    setup.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
    drop(setup);
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    assert!(get_matching_tags(Some(&conn), "rust", "projects", None).is_empty());
    assert!(get_keyword_baskets_for_tags(Some(&conn), &[1]).is_empty());
    assert!(list_tags(Some(&conn), "projects", None, 50).is_empty());
    assert!(get_tag_hierarchy(Some(&conn), "projects", None).is_empty());
}
