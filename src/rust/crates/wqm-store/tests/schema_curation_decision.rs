//! `P04-GT002-WO018` — `C-sch-curation-decision`.
//!
//! The `CHECK` is the whole declaration here: exactly one target, enforced with
//! `<>` over two `IS NOT NULL` tests. Both failure directions are exercised, since
//! an untargeted decision and a doubly-targeted one fail for different reasons and
//! a `CHECK` that caught only one would still pass a test that tried only one.

use rusqlite::Connection;
use wqm_store::schema::classification::{CURATION_DECISION_OPS, CURATION_DECISION_SQL};

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.execute_batch(CURATION_DECISION_SQL)
        .expect("curation_decision DDL applies");
    conn
}

/// `(target_category_key, target_document_id)` are the two axes the CHECK reads;
/// everything else is held constant so a failure names the axis that caused it.
fn decide(
    conn: &Connection,
    key: Option<&[u8]>,
    doc: Option<i64>,
    op: &str,
) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO curation_decision
             (target_category_key, target_document_id, op, payload, authored_by, created_at)
         VALUES (?1, ?2, ?3, '{}', 'server', 0)",
        rusqlite::params![key, doc, op],
    )
}

#[test]
fn ddl_applies_and_is_idempotent() {
    let conn = applied();
    conn.execute_batch(CURATION_DECISION_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('curation_decision') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        cols,
        vec![
            "decision_id",
            "target_category_key",
            "target_document_id",
            "op",
            "payload",
            "authored_by",
            "created_at",
        ]
    );
}

#[test]
fn exactly_one_target_is_set() {
    let conn = applied();

    decide(&conn, Some(b"lineage"), None, "freeze").expect("a category-scoped op");
    decide(&conn, None, Some(42), "override").expect("a document-scoped op");

    assert!(
        decide(&conn, None, None, "freeze").is_err(),
        "an untargeted decision is rejected"
    );
    assert!(
        decide(&conn, Some(b"lineage"), Some(42), "freeze").is_err(),
        "a doubly-targeted decision is rejected"
    );
}

#[test]
fn the_target_is_a_lineage_key_not_a_per_round_surrogate() {
    // Why it matters: a decision pointing at a round-stable key stays valid across
    // reinduction by construction, with no migration step to find and rewrite it.
    let conn = applied();
    let ty: String = conn
        .query_row(
            "SELECT type FROM pragma_table_info('curation_decision')
              WHERE name = 'target_category_key'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        ty, "BLOB",
        "the lineage key is a raw-byte key, not an INTEGER surrogate"
    );
}

#[test]
fn the_op_value_set_is_a_documented_set_not_a_constraint() {
    // The fragment declares `op` as TEXT with a listed value set. It is recorded as
    // a constant, deliberately NOT as a CHECK and NOT as a Rust enum: minting a
    // closed type for it would occupy vocabulary, which is the interference this
    // GT's scope boundary exists to avoid.
    assert_eq!(
        CURATION_DECISION_OPS,
        ["freeze", "rename", "merge", "split", "override"]
    );

    let conn = applied();
    for op in CURATION_DECISION_OPS {
        decide(&conn, Some(b"lineage"), None, op).expect("every listed op stores");
    }
    decide(&conn, Some(b"lineage"), None, "not-in-the-set")
        .expect("the column does not constrain the set — the writer does");
}
