//! `P04-GT002-WO026` — `C-sch-lib-references`.
//!
//! The two document references have different delete rules, and that difference is
//! the design rather than an inconsistency. Deleting the CITING document deletes
//! the citation; deleting the CITED document *unresolves* it. Both are exercised,
//! because a test of only one would pass against a schema that had cascaded both.

use rusqlite::Connection;
use wqm_store::schema::library::LIB_REFERENCES_SQL;

/// The `items_libraries` stand-in: `A-tbl-items` is out of this GT's scope.
const ITEM_TABLE_DOUBLE: &str = "CREATE TABLE items_libraries (item_id INTEGER PRIMARY KEY);";

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.pragma_update(None, "foreign_keys", true)
        .expect("enable FK enforcement");
    conn.execute_batch(ITEM_TABLE_DOUBLE)
        .expect("FK-target double");
    conn.execute_batch(LIB_REFERENCES_SQL)
        .expect("lib_references DDL applies");
    conn
}

fn an_item(conn: &Connection, id: i64) {
    conn.execute("INSERT INTO items_libraries VALUES (?1)", [id])
        .expect("item row");
}

fn cite(conn: &Connection, src: i64, norm: &[u8], resolved: Option<i64>) -> i64 {
    conn.execute(
        "INSERT INTO lib_references (src_doc, kind, title, norm_key, resolved_doc)
         VALUES (?1, 'book', 'A Title', ?2, ?3)",
        rusqlite::params![src, norm, resolved],
    )
    .expect("reference row");
    conn.last_insert_rowid()
}

#[test]
fn ddl_applies_and_is_idempotent() {
    let conn = applied();
    conn.execute_batch(LIB_REFERENCES_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('lib_references') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        cols,
        vec![
            "ref_id",
            "src_doc",
            "kind",
            "title",
            "author",
            "section",
            "year",
            "isbn",
            "publication",
            "ref",
            "page",
            "url",
            "norm_key",
            "resolved_doc",
            "resolution_method",
            "resolution_confidence",
            "ambiguous",
        ]
    );
}

#[test]
fn the_raw_fields_are_sparse_tolerant_across_kinds() {
    // A reference's kind is data, not a schema decision, so one table holds all
    // three shapes and every per-kind field is nullable.
    let conn = applied();
    an_item(&conn, 1);
    conn.execute(
        "INSERT INTO lib_references (src_doc, kind, title, author, year, isbn)
         VALUES (1, 'book', 'T', 'A', 1999, '123')",
        [],
    )
    .expect("a book row fills the book fields");
    conn.execute(
        "INSERT INTO lib_references (src_doc, kind, url) VALUES (1, 'website', 'https://x')",
        [],
    )
    .expect("a website row fills only url");
}

#[test]
fn deleting_the_citing_document_deletes_the_citation() {
    let conn = applied();
    an_item(&conn, 1);
    cite(&conn, 1, b"norm", None);

    conn.execute("DELETE FROM items_libraries WHERE item_id = 1", [])
        .unwrap();
    let left: i64 = conn
        .query_row("SELECT count(*) FROM lib_references", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        left, 0,
        "a reference has no meaning without the document that made it"
    );
}

#[test]
fn deleting_the_cited_document_unresolves_the_citation_rather_than_deleting_it() {
    let conn = applied();
    an_item(&conn, 1);
    an_item(&conn, 2);
    let id = cite(&conn, 1, b"norm", Some(2));

    conn.execute("DELETE FROM items_libraries WHERE item_id = 2", [])
        .unwrap();

    let (surviving, resolved): (i64, Option<i64>) = conn
        .query_row(
            "SELECT count(*), max(resolved_doc) FROM lib_references WHERE ref_id = ?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(surviving, 1, "the citation survives its target");
    assert_eq!(resolved, None, "and returns to the orphan wait-set");
}

#[test]
fn the_orphan_index_is_partial_so_it_shrinks_as_references_resolve() {
    let conn = applied();
    let sql: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'ix_lib_references_orphan'",
            [],
            |r| r.get(0),
        )
        .expect("the orphan index exists");
    assert!(
        sql.contains("WHERE resolved_doc IS NULL"),
        "a full index on norm_key would answer the same query and carry every \
         resolved row forever; got {sql}"
    );
}
