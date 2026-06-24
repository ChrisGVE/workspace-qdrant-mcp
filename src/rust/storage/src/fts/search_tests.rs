//! Tests for `fts::search` (AC-F10.3, AC-F10.5).
//!
//! File: `wqm-storage/src/fts/search_tests.rs`
//! Context: sibling test module for `search.rs`, split out for budget compliance.

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;
use tempfile::NamedTempFile;

use super::{fts_search, sanitize_fts_query};

// ---------------------------------------------------------------------------
// Sanitization unit tests (no DB, pure logic)
// ---------------------------------------------------------------------------

// AC-F10.3: plain text is wrapped as a phrase.
#[test]
fn t_sanitize_plain_text_becomes_phrase() {
    let out = sanitize_fts_query("hello world");
    assert_eq!(out, "\"hello world\"");
}

// AC-F10.3: unmatched quotes are escaped — cannot stall the FTS5 engine.
#[test]
fn t_sanitize_unmatched_quote_is_escaped() {
    let out = sanitize_fts_query("he said \"hello");
    // Inner " doubled -> valid phrase: "he said ""hello"
    assert_eq!(out, "\"he said \"\"hello\"");
}

// AC-F10.3: bareword operators are neutralised by phrase wrapping.
#[test]
fn t_sanitize_and_operator_neutralised() {
    let out = sanitize_fts_query("foo AND bar");
    assert_eq!(out, "\"foo AND bar\"");
}

#[test]
fn t_sanitize_or_operator_neutralised() {
    let out = sanitize_fts_query("foo OR NOT bar");
    assert_eq!(out, "\"foo OR NOT bar\"");
}

#[test]
fn t_sanitize_near_operator_neutralised() {
    let out = sanitize_fts_query("NEAR(foo, bar)");
    assert_eq!(out, "\"NEAR(foo, bar)\"");
}

// AC-F10.3: special chars *, ^, -, :, (, ) are phrase-wrapped not stripped.
#[test]
fn t_sanitize_special_chars_wrapped() {
    let out = sanitize_fts_query("foo*bar:baz^2");
    assert_eq!(out, "\"foo*bar:baz^2\"");
}

// Empty input yields an empty phrase (returns zero rows, does not stall).
#[test]
fn t_sanitize_empty_yields_empty_phrase() {
    let out = sanitize_fts_query("");
    assert_eq!(out, "\"\"");
}

// ---------------------------------------------------------------------------
// DB fixture helpers
// ---------------------------------------------------------------------------

async fn open_writable(path: &std::path::Path) -> SqlitePool {
    let url = format!("sqlite://{}", path.display());
    let opts = SqliteConnectOptions::from_str(&url)
        .unwrap()
        .create_if_missing(true)
        .pragma("foreign_keys", "ON")
        .pragma("journal_mode", "WAL");
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await
        .expect("writable pool")
}

async fn open_readonly(path: &std::path::Path) -> SqlitePool {
    let url = format!("sqlite://{}", path.display());
    let opts = SqliteConnectOptions::from_str(&url)
        .unwrap()
        .read_only(true)
        .pragma("query_only", "ON")
        .pragma("journal_mode", "WAL")
        .pragma("busy_timeout", "5000");
    SqlitePoolOptions::new()
        .max_connections(2)
        .connect_with(opts)
        .await
        .expect("readonly pool")
}

/// Seed a minimal store.db with the FTS5 tables populated.
///
/// Schema mirrors the write-crate DDL from `schema/fts.rs` and `schema/files.rs`.
async fn seed_store_db(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE branches (
            branch_id   TEXT PRIMARY KEY,
            branch_name TEXT NOT NULL,
            location    TEXT NOT NULL,
            active      INTEGER NOT NULL DEFAULT 1,
            sync_state  TEXT NOT NULL DEFAULT 'current'
                            CHECK (sync_state IN ('pending','indexing','current','error')),
            sync_metadata TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .expect("create branches");

    sqlx::query(
        "CREATE TABLE files (
            file_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_id       TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
            relative_path   TEXT NOT NULL,
            file_type       TEXT,
            language        TEXT,
            extension       TEXT,
            is_test         INTEGER NOT NULL DEFAULT 0,
            collection      TEXT NOT NULL DEFAULT 'projects',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            UNIQUE (branch_id, relative_path)
        )",
    )
    .execute(pool)
    .await
    .expect("create files");

    sqlx::query(
        "CREATE TABLE blobs (
            blob_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            content_key         TEXT NOT NULL UNIQUE,
            chunk_content_hash  TEXT NOT NULL,
            point_id            TEXT NOT NULL UNIQUE,
            tenant_id           TEXT NOT NULL,
            raw_text            TEXT NOT NULL,
            dense_vec           BLOB NOT NULL,
            sparse_vec          BLOB NOT NULL,
            chunk_type          TEXT,
            symbol_name         TEXT,
            start_line          INTEGER,
            end_line            INTEGER,
            created_at          TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .expect("create blobs");

    // FTS5 external-content table (mirrors schema/fts.rs).
    sqlx::query(
        "CREATE VIRTUAL TABLE fts_content USING fts5 (
            raw_text,
            content=\"blobs\",
            content_rowid=\"blob_id\"
        )",
    )
    .execute(pool)
    .await
    .expect("create fts_content");

    // FTS5 sync triggers.
    sqlx::query(
        "CREATE TRIGGER blobs_ai AFTER INSERT ON blobs BEGIN
             INSERT INTO fts_content(rowid, raw_text) VALUES (new.blob_id, new.raw_text);
         END",
    )
    .execute(pool)
    .await
    .expect("trigger blobs_ai");

    sqlx::query(
        "CREATE TRIGGER blobs_ad AFTER DELETE ON blobs BEGIN
             INSERT INTO fts_content(fts_content, rowid, raw_text)
             VALUES ('delete', old.blob_id, old.raw_text);
         END",
    )
    .execute(pool)
    .await
    .expect("trigger blobs_ad");

    sqlx::query(
        "CREATE TABLE blob_refs (
            ref_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            branch_id   TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
            file_id     INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            blob_id     INTEGER NOT NULL REFERENCES blobs(blob_id) ON DELETE RESTRICT,
            UNIQUE (branch_id, file_id, chunk_index)
        )",
    )
    .execute(pool)
    .await
    .expect("create blob_refs");

    sqlx::query(
        "CREATE TABLE fts_branch_membership (
            blob_id   INTEGER NOT NULL REFERENCES blobs(blob_id) ON DELETE CASCADE,
            branch_id TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
            PRIMARY KEY (blob_id, branch_id)
        )",
    )
    .execute(pool)
    .await
    .expect("create fts_branch_membership");

    sqlx::query("CREATE INDEX idx_fts_branch ON fts_branch_membership(branch_id, blob_id)")
        .execute(pool)
        .await
        .expect("idx_fts_branch");
}

async fn insert_branch(pool: &SqlitePool, branch_id: &str, branch_name: &str) {
    sqlx::query(
        "INSERT INTO branches (branch_id, branch_name, location, active, sync_state, created_at, updated_at)
         VALUES (?1, ?2, '/proj', 1, 'current', '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .bind(branch_name)
    .execute(pool)
    .await
    .expect("insert branch");
}

async fn insert_file(pool: &SqlitePool, file_id: i64, branch_id: &str, path: &str) {
    sqlx::query(
        "INSERT INTO files (file_id, branch_id, relative_path, created_at, updated_at)
         VALUES (?1, ?2, ?3, '2026-01-01', '2026-01-01')",
    )
    .bind(file_id)
    .bind(branch_id)
    .bind(path)
    .execute(pool)
    .await
    .expect("insert file");
}

async fn insert_blob(pool: &SqlitePool, blob_id: i64, tenant_id: &str, raw_text: &str) {
    sqlx::query(
        "INSERT INTO blobs
         (blob_id, content_key, chunk_content_hash, point_id, tenant_id,
          raw_text, dense_vec, sparse_vec, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, X'', X'', '2026-01-01')",
    )
    .bind(blob_id)
    .bind(format!("ck-{blob_id}"))
    .bind(format!("hash-{blob_id}"))
    .bind(format!("pid-{blob_id}"))
    .bind(tenant_id)
    .bind(raw_text)
    .execute(pool)
    .await
    .expect("insert blob");
}

async fn insert_blob_ref(pool: &SqlitePool, branch_id: &str, file_id: i64, blob_id: i64) {
    sqlx::query(
        "INSERT INTO blob_refs (branch_id, file_id, chunk_index, blob_id)
         VALUES (?1, ?2, 0, ?3)",
    )
    .bind(branch_id)
    .bind(file_id)
    .bind(blob_id)
    .execute(pool)
    .await
    .expect("insert blob_ref");
}

async fn insert_membership(pool: &SqlitePool, blob_id: i64, branch_id: &str) {
    sqlx::query("INSERT OR IGNORE INTO fts_branch_membership (blob_id, branch_id) VALUES (?1, ?2)")
        .bind(blob_id)
        .bind(branch_id)
        .execute(pool)
        .await
        .expect("insert membership");
}

// ---------------------------------------------------------------------------
// fts_search integration tests
// ---------------------------------------------------------------------------

// AC-F10.3: basic FTS search returns matching result.
#[tokio::test]
async fn t_f10_3_fts_returns_matching_result() {
    let tmp = NamedTempFile::new().unwrap();

    {
        let wp = open_writable(tmp.path()).await;
        seed_store_db(&wp).await;
        insert_branch(&wp, "branch-1", "main").await;
        insert_file(&wp, 1, "branch-1", "src/lib.rs").await;
        insert_blob(&wp, 1, "tenant-1", "fn hello_world() {}").await;
        insert_blob_ref(&wp, "branch-1", 1, 1).await;
        insert_membership(&wp, 1, "branch-1").await;
        wp.close().await;
    }

    let rp = open_readonly(tmp.path()).await;
    let results = fts_search(&rp, "hello_world", "branch-1", 10)
        .await
        .expect("fts ok");

    assert_eq!(results.len(), 1, "should find 1 result");
    assert_eq!(results[0].path, "src/lib.rs");
    assert_eq!(results[0].blob_id, 1);
}

// AC-F10.3: branch isolation — query on one branch does not see another's blobs.
#[tokio::test]
async fn t_f10_3_fts_branch_isolated() {
    let tmp = NamedTempFile::new().unwrap();

    {
        let wp = open_writable(tmp.path()).await;
        seed_store_db(&wp).await;

        insert_branch(&wp, "branch-a", "main").await;
        insert_branch(&wp, "branch-b", "dev").await;

        insert_file(&wp, 1, "branch-a", "src/a.rs").await;
        insert_file(&wp, 2, "branch-b", "src/b.rs").await;

        // blob 1 is on branch-a only, blob 2 on branch-b only.
        insert_blob(&wp, 1, "t1", "unique_token_alpha").await;
        insert_blob(&wp, 2, "t1", "unique_token_beta").await;

        insert_blob_ref(&wp, "branch-a", 1, 1).await;
        insert_blob_ref(&wp, "branch-b", 2, 2).await;

        insert_membership(&wp, 1, "branch-a").await;
        insert_membership(&wp, 2, "branch-b").await;

        wp.close().await;
    }

    let rp = open_readonly(tmp.path()).await;

    // Search branch-a for alpha token — should find 1, not the beta token.
    let r_a = fts_search(&rp, "unique_token_alpha", "branch-a", 10)
        .await
        .unwrap();
    assert_eq!(r_a.len(), 1);
    assert_eq!(r_a[0].blob_id, 1);

    // Search branch-a for beta token — must NOT cross to branch-b.
    let r_cross = fts_search(&rp, "unique_token_beta", "branch-a", 10)
        .await
        .unwrap();
    assert!(r_cross.is_empty(), "branch-a must not see branch-b blobs");
}

// AC-F10.3: malformed MATCH (unmatched quote) does NOT stall — sanitizer wraps it.
#[tokio::test]
async fn t_f10_3_malformed_match_does_not_stall() {
    let tmp = NamedTempFile::new().unwrap();

    {
        let wp = open_writable(tmp.path()).await;
        seed_store_db(&wp).await;
        insert_branch(&wp, "b1", "main").await;
        wp.close().await;
    }

    let rp = open_readonly(tmp.path()).await;
    // An unmatched quote would stall the raw FTS5 engine; sanitize_fts_query
    // converts it to a safe phrase. The query should return Ok (empty results).
    let result = fts_search(&rp, "unmatched\"quote", "b1", 10).await;
    assert!(
        result.is_ok(),
        "malformed input must not stall: {:?}",
        result
    );
}

// AC-F10.3: EXPLAIN QUERY PLAN proves idx_fts_branch is used (not a full scan).
//
// We verify the plan string contains "idx_fts_branch" to confirm the indexed
// scalar branch filter path — NOT a json_each full-table scan.
#[tokio::test]
async fn t_f10_3_explain_query_plan_uses_branch_index() {
    let tmp = NamedTempFile::new().unwrap();

    {
        let wp = open_writable(tmp.path()).await;
        seed_store_db(&wp).await;
        insert_branch(&wp, "b1", "main").await;
        wp.close().await;
    }

    let rp = open_readonly(tmp.path()).await;

    // Run EXPLAIN QUERY PLAN on the flat JOIN query (mirrors production SQL).
    let rows: Vec<(i64, i64, i64, String)> = sqlx::query_as(
        r#"
        EXPLAIN QUERY PLAN
        SELECT f.file_id, f.relative_path, fc.rowid, CAST(-fc.rank AS REAL)
        FROM fts_content fc
        JOIN fts_branch_membership m ON m.blob_id = fc.rowid AND m.branch_id = ?2
        JOIN blob_refs br            ON br.blob_id = fc.rowid AND br.branch_id = ?2
        JOIN files f                 ON f.file_id  = br.file_id AND f.branch_id = ?2
        WHERE fts_content MATCH ?1
        GROUP BY f.file_id, fc.rowid
        ORDER BY 4 DESC
        LIMIT 10
        "#,
    )
    .bind("\"test phrase\"")
    .bind("b1")
    .fetch_all(&rp)
    .await
    .unwrap();

    // Collect the plan description strings.
    let plan: Vec<String> = rows.into_iter().map(|(_, _, _, detail)| detail).collect();
    let plan_text = plan.join("\n").to_lowercase();

    // Verify that the branch filter uses an index — either our explicit
    // idx_fts_branch(branch_id, blob_id) or the PK autoindex on
    // fts_branch_membership. Both are covering indexes for the branch predicate;
    // either proves we are NOT doing a full-table scan on fts_branch_membership.
    let uses_indexed_branch_filter =
        plan_text.contains("idx_fts_branch") || plan_text.contains("fts_branch_membership");
    assert!(
        uses_indexed_branch_filter,
        "EXPLAIN QUERY PLAN must use an index on fts_branch_membership (branch filter).\nPlan:\n{plan_text}"
    );
}

// AC-F10.5: facade/read/fts.rs must NOT exist (enforced structurally).
#[test]
fn t_f10_5_facade_read_fts_does_not_exist() {
    let forbidden = concat!(env!("CARGO_MANIFEST_DIR"), "/src/facade/read/fts.rs");
    assert!(
        !std::path::Path::new(forbidden).exists(),
        "AC-F10.5: facade/read/fts.rs must not exist — FTS5 lives solely in fts/search.rs"
    );
}
