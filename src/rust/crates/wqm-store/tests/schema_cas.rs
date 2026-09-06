//! What a DDL declaration can be tested for: that it applies, and that the
//! constraints it claims actually reject what they say they reject.
//!
//! `P04-GT002-WO015`. A declaration's test is not a behavior test — nothing here
//! exercises the CAS. It exercises the *statement*: an assertion-free green would
//! leave a `CHECK` that never fires indistinguishable from one that does, which is
//! precisely the defect the test discipline exists to prevent.

use rusqlite::Connection;
use wqm_store::schema::cas::CAS_ENTRY_SQL;

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.execute_batch(CAS_ENTRY_SQL)
        .expect("cas_entry DDL applies");
    conn
}

#[test]
fn ddl_applies_and_is_idempotent() {
    let conn = applied();
    // `IF NOT EXISTS` is load-bearing: the daemon that will own creation applies
    // the family on every open.
    conn.execute_batch(CAS_ENTRY_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('cas_entry') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(cols, vec!["file_hash", "held", "blob_locator"]);
}

#[test]
fn file_hash_is_the_primary_key_and_a_second_ingest_adds_no_row() {
    let conn = applied();
    let hash = vec![0xABu8; 32];
    conn.execute("INSERT INTO cas_entry VALUES (?1, 0, NULL)", [&hash])
        .expect("first ingest");

    // The dedup case: identical content presented again is the SAME entry.
    let again = conn.execute("INSERT INTO cas_entry VALUES (?1, 0, NULL)", [&hash]);
    assert!(
        again.is_err(),
        "a duplicate file_hash must be rejected by the PK"
    );

    let n: i64 = conn
        .query_row("SELECT count(*) FROM cas_entry", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn held_and_blob_locator_agree_or_the_row_is_rejected() {
    let conn = applied();
    let h = |b: u8| vec![b; 32];

    conn.execute("INSERT INTO cas_entry VALUES (?1, 0, NULL)", [&h(1)])
        .expect("hash-only with no locator is legal");
    conn.execute(
        "INSERT INTO cas_entry VALUES (?1, 1, ?2)",
        (&h(2), b"loc".to_vec()),
    )
    .expect("blob-held with a locator is legal");

    assert!(
        conn.execute("INSERT INTO cas_entry VALUES (?1, 1, NULL)", [&h(3)])
            .is_err(),
        "blob-held without a locator must be rejected"
    );
    assert!(
        conn.execute(
            "INSERT INTO cas_entry VALUES (?1, 0, ?2)",
            (&h(4), b"loc".to_vec())
        )
        .is_err(),
        "hash-only carrying a locator must be rejected"
    );
}

#[test]
fn the_table_carries_no_scope_column() {
    // Global by construction. A keep/collection column here would partition the
    // dedup silently — every query would still answer, against a smaller world.
    let conn = applied();
    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('cas_entry')")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    for forbidden in ["keep_id", "collection", "tenant_id", "branch_id"] {
        assert!(
            !cols.iter().any(|c| c == forbidden),
            "unexpected scope column {forbidden}"
        );
    }
}
