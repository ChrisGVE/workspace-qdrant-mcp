//! `P04-GT002-WO017` — `C-sch-category-keyword`.
//!
//! The cascade is the retention mechanism, so the cascade is what gets tested —
//! and `PRAGMA foreign_keys` is enabled explicitly, because SQLite leaves foreign
//! keys OFF by default and a cascade test against a connection that never turned
//! them on passes by not enforcing anything.

use rusqlite::Connection;
use wqm_store::schema::classification::{CATEGORY_KEYWORD_SQL, TAXONOMY_CATEGORY_SQL};

fn applied() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    conn.pragma_update(None, "foreign_keys", true)
        .expect("enable FK enforcement");
    conn.execute_batch(TAXONOMY_CATEGORY_SQL)
        .expect("parent DDL applies");
    conn.execute_batch(CATEGORY_KEYWORD_SQL)
        .expect("category_keyword DDL applies");
    conn
}

fn a_category(conn: &Connection, round: i64) -> i64 {
    conn.execute(
        "INSERT INTO taxonomy_category
             (category_key, name, description, level, parent_id, is_frozen, is_utility, round)
         VALUES (?1, ?2, NULL, 0, NULL, 0, 0, ?3)",
        rusqlite::params![b"key".to_vec(), format!("cat-{round}"), round],
    )
    .expect("category row");
    conn.last_insert_rowid()
}

#[test]
fn ddl_applies_and_is_idempotent() {
    let conn = applied();
    conn.execute_batch(CATEGORY_KEYWORD_SQL)
        .expect("re-applying is a no-op");

    let cols: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('category_keyword') ORDER BY cid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(cols, vec!["category_id", "keyword", "weight"]);
}

#[test]
fn one_weight_per_category_and_keyword() {
    let conn = applied();
    let cat = a_category(&conn, 1);
    conn.execute(
        "INSERT INTO category_keyword VALUES (?1, 'sqlite', 0.9)",
        [cat],
    )
    .expect("first weight");
    assert!(
        conn.execute(
            "INSERT INTO category_keyword VALUES (?1, 'sqlite', 0.4)",
            [cat]
        )
        .is_err(),
        "a duplicate keyword inside one profile must be rejected by the composite PK"
    );
    // The same keyword in a different category is a different profile entry.
    let other = a_category(&conn, 2);
    conn.execute(
        "INSERT INTO category_keyword VALUES (?1, 'sqlite', 0.4)",
        [other],
    )
    .expect("the same keyword under another category is legal");
}

#[test]
fn a_profile_cannot_exist_without_its_category() {
    let conn = applied();
    assert!(
        conn.execute(
            "INSERT INTO category_keyword VALUES (999, 'orphan', 1.0)",
            []
        )
        .is_err(),
        "a profile row naming an absent category must be rejected"
    );
}

#[test]
fn deleting_a_category_reclaims_its_profile() {
    // Retention (current + previous) holds for profiles by this cascade, not by a
    // GC routine that remembers to visit a second table.
    let conn = applied();
    let cat = a_category(&conn, 1);
    conn.execute(
        "INSERT INTO category_keyword VALUES (?1, 'sqlite', 0.9)",
        [cat],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO category_keyword VALUES (?1, 'index', 0.5)",
        [cat],
    )
    .unwrap();

    conn.execute(
        "DELETE FROM taxonomy_category WHERE category_id = ?1",
        [cat],
    )
    .unwrap();

    let left: i64 = conn
        .query_row("SELECT count(*) FROM category_keyword", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        left, 0,
        "the superseded round's profile rows go with its categories"
    );
}
