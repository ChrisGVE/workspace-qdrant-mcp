//! `P04-GT002-WO025` — `C-sch-lib-keywords`.
//!
//! One of these tests asserts an absence — that there is exactly one score column.
//! v0.1 carried three, and they retire with the extractor that gave them meaning;
//! an absence nobody checks is an absence that quietly comes back.

use rusqlite::Connection;
use wqm_store::schema::library::LIB_KEYWORDS_SQL;

/// The `items_libraries` stand-in, as in the sibling tests: `A-tbl-items` is out
/// of this GT's scope, so the FK target is a documented double.
const ITEM_TABLE_DOUBLE: &str = "CREATE TABLE items_libraries (item_id INTEGER PRIMARY KEY);";

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.pragma_update(None, "foreign_keys", true)
        .expect("enable FK enforcement");
    conn.execute_batch(ITEM_TABLE_DOUBLE)
        .expect("FK-target double");
    conn.execute_batch(LIB_KEYWORDS_SQL)
        .expect("lib_keywords DDL applies");
    conn
}

fn an_item(conn: &Connection, id: i64) {
    conn.execute("INSERT INTO items_libraries VALUES (?1)", [id])
        .expect("item row");
}

fn keyword(conn: &Connection, doc: i64, kw: &str, score: f64) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO lib_keywords VALUES (?1, ?2, ?3)",
        rusqlite::params![doc, kw, score],
    )
}

#[test]
fn ddl_applies_and_is_idempotent() {
    let conn = applied();
    conn.execute_batch(LIB_KEYWORDS_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('lib_keywords') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(cols, vec!["document_id", "keyword", "score"]);
}

#[test]
fn there_is_exactly_one_score_column() {
    // v0.1's `semantic_score` / `lexical_score` / `stability_count` were artifacts
    // of the extractor that produced them and retire with it (B2, §11.2 J-7).
    // Carrying them forward would preserve a shape whose meaning had been retired.
    let conn = applied();
    let scoreish: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('lib_keywords')")
        .unwrap()
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|c| c.contains("score") || c.contains("count"))
        .collect();
    assert_eq!(
        scoreish,
        vec!["score"],
        "one score column, and no v0.1 residue"
    );
}

#[test]
fn one_score_per_document_and_keyword() {
    let conn = applied();
    an_item(&conn, 1);
    keyword(&conn, 1, "sqlite", 0.9).expect("first keyword");
    assert!(
        keyword(&conn, 1, "sqlite", 0.2).is_err(),
        "the composite PK forbids a duplicate"
    );

    an_item(&conn, 2);
    keyword(&conn, 2, "sqlite", 0.2).expect("the same keyword under another document is legal");
}

#[test]
fn keywords_are_indexed_for_a_corpus_scan_and_die_with_their_document() {
    let conn = applied();
    let indexes: Vec<String> = conn
        .prepare("SELECT name FROM pragma_index_list('lib_keywords')")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        indexes.iter().any(|i| i == "ix_lib_keywords_keyword"),
        "induction reads the corpus as a bounded stream; found {indexes:?}"
    );

    an_item(&conn, 1);
    keyword(&conn, 1, "sqlite", 0.9).unwrap();
    conn.execute("DELETE FROM items_libraries WHERE item_id = 1", [])
        .unwrap();
    let left: i64 = conn
        .query_row("SELECT count(*) FROM lib_keywords", [], |r| r.get(0))
        .unwrap();
    assert_eq!(left, 0, "DATA-01: keywords are scoped by their document");
}
