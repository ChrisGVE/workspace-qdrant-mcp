//! `P04-GT002-WO019` — `C-sch-doc-classification`.
//!
//! Two structural claims are made by this table and both are tested against a
//! connection with foreign keys ON: an assignment cannot name a category absent
//! from its round (the composite FK), and a document cannot hold two primaries in
//! one round (the partial unique index).
//!
//! `items_libraries` is **not** declared by this crate — `A-tbl-items` is out of
//! `P04-GT002`'s scope. The fixture below is a test double standing in for the FK
//! target, and is deliberately minimal so it cannot be mistaken for a declaration.

use rusqlite::Connection;
use wqm_store::schema::classification::{DOC_CLASSIFICATION_SQL, TAXONOMY_CATEGORY_SQL};

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.pragma_update(None, "foreign_keys", true)
        .expect("enable FK enforcement");
    conn.execute_batch(TAXONOMY_CATEGORY_SQL)
        .expect("parent DDL applies");
    conn.execute_batch(DOC_CLASSIFICATION_SQL)
        .expect("doc_classification DDL applies");
    conn
}

fn a_category(conn: &Connection, key: &[u8], round: i64) {
    conn.execute(
        "INSERT INTO taxonomy_category
             (category_key, name, description, level, parent_id, is_frozen, is_utility, round)
         VALUES (?1, ?2, NULL, 0, NULL, 0, 0, ?3)",
        rusqlite::params![
            key,
            format!("{}-{round}", String::from_utf8_lossy(key)),
            round
        ],
    )
    .expect("category row");
}

fn assign(
    conn: &Connection,
    doc: i64,
    key: &[u8],
    round: i64,
    primary: bool,
) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO doc_classification VALUES (?1, ?2, ?3, 0.5, 'induced-Jaccard', ?4)",
        rusqlite::params![doc, key, round, primary as i64],
    )
}

#[test]
fn ddl_applies_and_is_idempotent() {
    let conn = applied();
    conn.execute_batch(DOC_CLASSIFICATION_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('doc_classification') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        cols,
        vec![
            "document_id",
            "category_key",
            "round",
            "confidence",
            "method",
            "is_primary"
        ]
    );
}

#[test]
fn an_assignment_to_a_category_absent_from_its_round_is_impossible() {
    // DATA-04, the store half of a two-sided defense. The other half is N53 never
    // proposing one; this half survives a producer that stops carrying it.
    let conn = applied();
    a_category(&conn, b"k1", 1);

    assign(&conn, 10, b"k1", 1, true).expect("the category exists in round 1");
    assert!(
        assign(&conn, 10, b"k1", 2, true).is_err(),
        "the same lineage key in a round that has no such category must be rejected"
    );
    assert!(
        assign(&conn, 10, b"absent", 1, false).is_err(),
        "an unknown lineage key must be rejected"
    );
}

#[test]
fn one_primary_per_document_per_round() {
    let conn = applied();
    a_category(&conn, b"k1", 1);
    a_category(&conn, b"k2", 1);
    a_category(&conn, b"k1", 2);

    assign(&conn, 10, b"k1", 1, true).expect("the primary assignment");
    assign(&conn, 10, b"k2", 1, false).expect("a secondary assignment is unconstrained");
    assert!(
        assign(&conn, 10, b"k2", 1, true).is_err(),
        "a second primary in the same round must be rejected"
    );
    // The index is partial on `is_primary`, so the next round is unaffected.
    assign(&conn, 10, b"k1", 2, true).expect("a primary in the next round is legal");
}

#[test]
fn round_gc_of_a_category_cascades_its_assignments() {
    // DATA-02: the retention bound holds for the highest-cardinality family by
    // schema rather than by discipline.
    let conn = applied();
    a_category(&conn, b"k1", 1);
    assign(&conn, 10, b"k1", 1, true).unwrap();
    assign(&conn, 11, b"k1", 1, true).unwrap();

    conn.execute("DELETE FROM taxonomy_category WHERE round = 1", [])
        .unwrap();

    let left: i64 = conn
        .query_row("SELECT count(*) FROM doc_classification", [], |r| r.get(0))
        .unwrap();
    assert_eq!(left, 0);
}
