//! `P04-GT002-WO031` — `C-sch-taxonomy-category`.
//!
//! The two `UNIQUE` constraints do different jobs, and each is tested for the job
//! it does. A test that only applied the DDL would leave a constraint that never
//! fires indistinguishable from one that does.

use rusqlite::Connection;
use wqm_store::schema::classification::TAXONOMY_CATEGORY_SQL;

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.execute_batch(TAXONOMY_CATEGORY_SQL)
        .expect("taxonomy_category DDL applies");
    conn
}

/// One category row, parameterised on the axes the constraints care about.
fn insert(conn: &Connection, key: &[u8], name: &str, round: i64) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO taxonomy_category
             (category_key, name, description, level, parent_id, is_frozen, is_utility, round)
         VALUES (?1, ?2, NULL, 0, NULL, 0, 0, ?3)",
        rusqlite::params![key, name, round],
    )
}

#[test]
fn ddl_applies_is_idempotent_and_declares_the_nine_columns() {
    let conn = applied();
    conn.execute_batch(TAXONOMY_CATEGORY_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('taxonomy_category') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        cols,
        vec![
            "category_id",
            "category_key",
            "name",
            "description",
            "level",
            "parent_id",
            "is_frozen",
            "is_utility",
            "round",
        ]
    );
}

#[test]
fn names_are_unique_within_a_round_and_rounds_coexist() {
    let conn = applied();
    insert(&conn, b"k1", "Databases", 1).expect("first round-1 category");
    assert!(
        insert(&conn, b"k2", "Databases", 1).is_err(),
        "a duplicate name inside one round must be rejected"
    );
    // Append-then-flip depends on exactly this: the next round may reuse the name.
    insert(&conn, b"k1", "Databases", 2).expect("the same name in round 2 is legal");
}

#[test]
fn the_lineage_key_is_unique_per_round_so_the_composite_fk_has_a_target() {
    let conn = applied();
    insert(&conn, b"k1", "Databases", 1).unwrap();
    assert!(
        insert(&conn, b"k1", "Storage", 1).is_err(),
        "one lineage key may appear at most once per round"
    );
    insert(&conn, b"k1", "Databases (renamed)", 2)
        .expect("the same lineage key carries into the next round");

    // The candidate key doc_classification's composite FK names. If this is not
    // declarable, that FK cannot exist — so it is asserted, not assumed.
    let indexed: i64 = conn
        .query_row(
            "SELECT count(*) FROM pragma_index_list('taxonomy_category') WHERE origin = 'u'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(indexed, 2, "both UNIQUE constraints must be present");
}

#[test]
fn the_table_carries_no_tenant_or_collection_column() {
    // One cross-tenant backbone on the libraries axis. A scope column here would
    // not narrow a query — it would fork the taxonomy.
    let conn = applied();
    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('taxonomy_category')")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    for forbidden in ["tenant_id", "collection", "keep_id"] {
        assert!(
            !cols.iter().any(|c| c == forbidden),
            "unexpected scope column {forbidden}"
        );
    }
}
