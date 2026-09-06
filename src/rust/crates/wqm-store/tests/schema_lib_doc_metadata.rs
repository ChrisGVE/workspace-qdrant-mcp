//! `P04-GT002-WO024` — `C-sch-lib-doc-metadata`.
//!
//! `items_libraries` is declared by `A-tbl-items`, which is out of `P04-GT002`'s
//! scope. [`item_table_double`] is a **test double** standing in for the FK target
//! so the declared cascade can be exercised; it is two columns and no more, so it
//! cannot be mistaken for the real declaration when that one lands.

use rusqlite::Connection;
use wqm_store::schema::library::LIB_DOC_METADATA_SQL;

/// The minimal `items_libraries` stand-in this test needs, and nothing else.
const fn item_table_double() -> &'static str {
    "CREATE TABLE items_libraries (item_id INTEGER PRIMARY KEY, label TEXT);"
}

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.pragma_update(None, "foreign_keys", true)
        .expect("enable FK enforcement");
    conn.execute_batch(item_table_double())
        .expect("FK-target double");
    conn.execute_batch(LIB_DOC_METADATA_SQL)
        .expect("lib_doc_metadata DDL applies");
    conn
}

fn an_item(conn: &Connection, id: i64) {
    conn.execute("INSERT INTO items_libraries VALUES (?1, 'x')", [id])
        .expect("item row");
}

fn describe(conn: &Connection, doc: i64, norm: &[u8]) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO lib_doc_metadata (document_id, kind, title, norm_key)
         VALUES (?1, 'book', 'A Title', ?2)",
        rusqlite::params![doc, norm],
    )
}

#[test]
fn ddl_applies_and_is_idempotent_with_the_fourteen_declared_columns() {
    let conn = applied();
    conn.execute_batch(LIB_DOC_METADATA_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('lib_doc_metadata') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        cols,
        vec![
            "document_id",
            "kind",
            "title",
            "authors",
            "year",
            "isbn",
            "doi",
            "publication",
            "pub_ref",
            "page",
            "curated_class_raw",
            "title_method",
            "content_start_offset",
            "norm_key",
        ]
    );
}

#[test]
fn one_metadata_row_per_document_and_the_document_must_exist() {
    let conn = applied();
    an_item(&conn, 1);
    describe(&conn, 1, b"norm").expect("first description");
    assert!(
        describe(&conn, 1, b"norm").is_err(),
        "document_id is the PK — one row per document"
    );
    assert!(
        describe(&conn, 99, b"norm").is_err(),
        "metadata for an absent item is rejected"
    );
}

#[test]
fn deleting_the_library_item_takes_its_metadata_with_it() {
    // DATA-01. The row is scoped by this reference, which is why the table needs no
    // tenant or collection column of its own.
    let conn = applied();
    an_item(&conn, 1);
    describe(&conn, 1, b"norm").unwrap();

    conn.execute("DELETE FROM items_libraries WHERE item_id = 1", [])
        .unwrap();

    let left: i64 = conn
        .query_row("SELECT count(*) FROM lib_doc_metadata", [], |r| r.get(0))
        .unwrap();
    assert_eq!(left, 0);
}

#[test]
fn norm_key_is_indexed_because_matching_happens_on_the_normalised_form() {
    let conn = applied();
    let indexes: Vec<String> = conn
        .prepare("SELECT name FROM pragma_index_list('lib_doc_metadata')")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        indexes.iter().any(|i| i == "ix_lib_doc_metadata_norm_key"),
        "the N37 normaliser's key must be indexed; found {indexes:?}"
    );
}
