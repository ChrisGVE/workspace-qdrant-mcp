//! `P04-GT002-WO029` — `C-sch-ref-source-authority`.
//!
//! Two columns and no secondary index. Both of those are decisions the fragment
//! makes explicitly, so both are asserted — a table this small is exactly the kind
//! whose deliberate shape gets "improved" by a later reader who assumes the
//! absences were oversights.

use rusqlite::Connection;
use wqm_store::schema::library::REF_SOURCE_AUTHORITY_SQL;

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.execute_batch(REF_SOURCE_AUTHORITY_SQL)
        .expect("ref_source_authority DDL applies");
    conn
}

#[test]
fn ddl_applies_and_is_idempotent_with_two_columns() {
    let conn = applied();
    conn.execute_batch(REF_SOURCE_AUTHORITY_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('ref_source_authority') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(cols, vec!["source", "rank"]);
}

#[test]
fn one_rank_per_source() {
    let conn = applied();
    conn.execute("INSERT INTO ref_source_authority VALUES ('doi', 1)", [])
        .unwrap();
    assert!(
        conn.execute("INSERT INTO ref_source_authority VALUES ('doi', 2)", [])
            .is_err(),
        "a source cannot hold two precedences"
    );
    conn.execute("INSERT INTO ref_source_authority VALUES ('isbn', 2)", [])
        .expect("another source is unconstrained");
}

#[test]
fn precedence_carries_no_scope_column() {
    // Precedence is a property of the SOURCE, cutting across the whole reference
    // graph. A tenant or collection column would let it be answered differently in
    // two places, which is the one thing a precedence table must not permit.
    let conn = applied();
    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('ref_source_authority')")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    for forbidden in ["tenant_id", "collection", "document_id", "keep_id"] {
        assert!(
            !cols.iter().any(|c| c == forbidden),
            "unexpected scope column {forbidden}"
        );
    }
}

#[test]
fn there_is_no_secondary_index_and_that_is_deliberate() {
    // The table is tiny and is read by full scan whenever candidate sources must be
    // ordered. Recorded as an assertion so the absence reads as a choice.
    let conn = applied();
    let secondary: Vec<String> = conn
        .prepare("SELECT name FROM pragma_index_list('ref_source_authority') WHERE origin = 'c'")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        secondary.is_empty(),
        "no CREATE INDEX belongs here; found {secondary:?}"
    );
}
