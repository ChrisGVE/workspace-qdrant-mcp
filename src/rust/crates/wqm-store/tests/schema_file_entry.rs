//! `P04-GT002-WO020` — `C-sch-file-entry`.
//!
//! The test that matters is the negative one: many rows may share a `file_hash`.
//! It reads like a non-assertion, and it is the single fact that separates this
//! declaration from the one rev12 DATA MF-1 ruled out.

use rusqlite::Connection;
use wqm_store::schema::files::FILE_ENTRY_SQL;

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.execute_batch(FILE_ENTRY_SQL)
        .expect("file_entry DDL applies");
    conn
}

fn record(conn: &Connection, keep: &[u8], relpath: &str, hash: &[u8]) -> rusqlite::Result<usize> {
    let name = relpath.rsplit('/').next().unwrap_or(relpath);
    conn.execute(
        "INSERT INTO file_entry VALUES (?1, ?2, ?3, ?4, 0)",
        rusqlite::params![keep, relpath, name, hash],
    )
}

#[test]
fn ddl_applies_and_is_idempotent() {
    let conn = applied();
    conn.execute_batch(FILE_ENTRY_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('file_entry') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        cols,
        vec!["keep_id", "relpath", "name", "file_hash", "custody"]
    );
}

#[test]
fn many_rows_may_share_one_file_hash() {
    // The dedup case, and the reason the key is NOT file_hash: keying on content
    // would collapse these three onto one name and relpath, and deport would lose
    // two of them without anything failing.
    let conn = applied();
    let hash = vec![0x11u8; 32];
    record(&conn, b"keep-a", "docs/readme.md", &hash).expect("first instance");
    record(&conn, b"keep-a", "docs/copy.md", &hash).expect("same keep, second location");
    record(&conn, b"keep-b", "docs/readme.md", &hash).expect("another keep entirely");

    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM file_entry WHERE file_hash = ?1",
            [&hash],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 3, "three physical instances, one content hash");
}

#[test]
fn the_key_is_the_file_instance() {
    let conn = applied();
    record(&conn, b"keep-a", "docs/readme.md", &[1; 32]).unwrap();
    assert!(
        record(&conn, b"keep-a", "docs/readme.md", &[2; 32]).is_err(),
        "one row per (keep, relpath) — even when the content differs"
    );
}

#[test]
fn every_recorded_column_is_mandatory() {
    // Recorded truth, not filesystem truth: a NULL here would mean deport is
    // missing an input, which is the failure the family exists to prevent.
    let conn = applied();
    for col in ["keep_id", "relpath", "name", "file_hash", "custody"] {
        let notnull: i64 = conn
            .query_row(
                "SELECT \"notnull\" FROM pragma_table_info('file_entry') WHERE name = ?1",
                [col],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(notnull, 1, "{col} must be NOT NULL");
    }
}
