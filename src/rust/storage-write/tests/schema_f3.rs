//! F3 schema integration tests (AC-F3.2, AC-F3.4, Task 44).
//!
//! File: `wqm-storage-write/tests/schema_f3.rs`
//! Location: `src/rust/storage-write/tests/` (integration test suite)
//! Context: workspace-qdrant-mcp branch-storage model. Verifies the per-project
//!   `store.db` DDL against the three acceptance criteria that require live SQLite:
//!
//!   AC-F3.2 — FTS5 sync triggers (`blobs_ai` / `blobs_ad`):
//!     A GC'd blob leaves NO ghost FTS rowid; a newly inserted blob IS FTS-visible.
//!
//!   AC-F3.4 (DATA-01 / DATA-NIT-01) — Cross-tenant isolation defense-in-depth:
//!     (1) Insert via write facade with wrong tenant -> BOTH layers refuse.
//!     (2) Direct SQL bypassing facade with wrong tenant -> trigger alone refuses.
//!     (3) Trigger body reads `store_meta`, NOT `NEW.tenant_id` reflexively.
//!
//! Neighbors: [`wqm_storage_write::connection`] (factory under test),
//!   [`wqm_storage_write::schema`] (DDL applied in each test).

use sqlx::{Pool, Sqlite};
use tempfile::NamedTempFile;
use wqm_storage_write::{open_store_write, schema::ddl_statements};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Open a fresh in-file store.db and apply the full schema.
async fn fresh_store() -> (Pool<Sqlite>, NamedTempFile) {
    let tmp = NamedTempFile::new().expect("tempfile");
    let pool = open_store_write(tmp.path())
        .await
        .expect("open_store_write");
    for stmt in ddl_statements() {
        sqlx::query(stmt)
            .execute(&pool)
            .await
            .unwrap_or_else(|e| panic!("DDL failed: {e}\n  stmt: {stmt}"));
    }
    (pool, tmp)
}

/// Seed `store_meta` with the given tenant and insert a branch row.
async fn seed_meta_and_branch(pool: &Pool<Sqlite>, tenant: &str, branch_id: &str) {
    sqlx::query("INSERT INTO store_meta(tenant_id) VALUES (?)")
        .bind(tenant)
        .execute(pool)
        .await
        .expect("store_meta insert");

    sqlx::query(
        "INSERT INTO branches(branch_id, branch_name, location, created_at, updated_at) \
         VALUES (?,?,'/repo','2024-01-01','2024-01-01')",
    )
    .bind(branch_id)
    .bind("main")
    .execute(pool)
    .await
    .expect("branch insert");
}

/// Insert a blob using raw SQL (no write-facade layer) with the given tenant.
async fn insert_blob_raw(
    pool: &Pool<Sqlite>,
    content_key: &str,
    tenant_id: &str,
) -> Result<sqlx::sqlite::SqliteQueryResult, sqlx::Error> {
    sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?,?,?,?,'hello world',X'',X'','2024-01-01')",
    )
    .bind(content_key)
    .bind("hash1")
    .bind(format!("pt-{content_key}"))
    .bind(tenant_id)
    .execute(pool)
    .await
}

// ---------------------------------------------------------------------------
// AC-F3.2 — FTS5 sync triggers
// ---------------------------------------------------------------------------

/// A new blob must be immediately visible to FTS5 via `fts_content MATCH`.
#[tokio::test]
async fn fts_new_blob_is_visible() {
    let (pool, _tmp) = fresh_store().await;
    seed_meta_and_branch(&pool, "tenant-a", "b1").await;

    // Insert a blob; blobs_ai trigger fires.
    sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES ('ck1','h1','pt1','tenant-a','hello world',X'',X'','2024-01-01')",
    )
    .execute(&pool)
    .await
    .expect("blob insert");

    // FTS5 MATCH must find the blob.
    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM fts_content WHERE raw_text MATCH 'hello'")
            .fetch_one(&pool)
            .await
            .expect("fts match");

    assert_eq!(count, 1, "new blob must be FTS-visible (blobs_ai trigger)");
}

/// A GC'd (deleted) blob must leave NO ghost FTS rowid.
#[tokio::test]
async fn fts_deleted_blob_leaves_no_ghost_rowid() {
    let (pool, _tmp) = fresh_store().await;
    seed_meta_and_branch(&pool, "tenant-a", "b1").await;

    // Insert blob.
    sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES ('ck2','h2','pt2','tenant-a','unique phrase xyz',X'',X'','2024-01-01')",
    )
    .execute(&pool)
    .await
    .expect("blob insert");

    // Confirm it appears in FTS.
    let before: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM fts_content WHERE raw_text MATCH 'xyz'")
            .fetch_one(&pool)
            .await
            .expect("fts before");
    assert_eq!(before, 1, "blob must be FTS-visible before deletion");

    // Delete the blob; blobs_ad trigger fires.
    sqlx::query("DELETE FROM blobs WHERE content_key = 'ck2'")
        .execute(&pool)
        .await
        .expect("blob delete");

    // FTS must no longer see it.
    let after: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM fts_content WHERE raw_text MATCH 'xyz'")
            .fetch_one(&pool)
            .await
            .expect("fts after");

    assert_eq!(
        after, 0,
        "GC'd blob must leave no ghost FTS rowid (blobs_ad trigger)"
    );
}

/// Multiple blobs, then one deleted — only the deleted one vanishes from FTS.
#[tokio::test]
async fn fts_partial_delete_leaves_surviving_blob_visible() {
    let (pool, _tmp) = fresh_store().await;
    seed_meta_and_branch(&pool, "tenant-a", "b1").await;

    for (ck, pt, text) in [
        ("ck3", "pt3", "alpha content"),
        ("ck4", "pt4", "beta content"),
    ] {
        sqlx::query(
            "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
             raw_text, dense_vec, sparse_vec, created_at) \
             VALUES (?,?,?,?,?,X'',X'','2024-01-01')",
        )
        .bind(ck)
        .bind("hx")
        .bind(pt)
        .bind("tenant-a")
        .bind(text)
        .execute(&pool)
        .await
        .unwrap_or_else(|e| panic!("blob insert failed for {ck}: {e}"));
    }

    // Delete only 'ck3'.
    sqlx::query("DELETE FROM blobs WHERE content_key = 'ck3'")
        .execute(&pool)
        .await
        .expect("partial delete");

    let alpha: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM fts_content WHERE raw_text MATCH 'alpha'")
            .fetch_one(&pool)
            .await
            .expect("alpha fts");
    let beta: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM fts_content WHERE raw_text MATCH 'beta'")
            .fetch_one(&pool)
            .await
            .expect("beta fts");

    assert_eq!(alpha, 0, "deleted blob must not appear in FTS");
    assert_eq!(beta, 1, "surviving blob must remain FTS-visible");
}

// ---------------------------------------------------------------------------
// AC-F3.4 / DATA-NIT-01 — cross-tenant isolation defense-in-depth
// ---------------------------------------------------------------------------

/// (2) Direct SQL bypassing facade with WRONG tenant -> trigger alone refuses.
/// This proves the trigger is the structural backstop even when no Rust layer is
/// in the call path.
#[tokio::test]
async fn trigger_rejects_wrong_tenant_direct_sql() {
    let (pool, _tmp) = fresh_store().await;
    seed_meta_and_branch(&pool, "tenant-owner", "b1").await;

    // Attempt to insert a blob with a DIFFERENT tenant directly (no facade).
    let result = insert_blob_raw(&pool, "ck-cross", "tenant-intruder").await;

    assert!(
        result.is_err(),
        "BEFORE INSERT trigger must reject wrong-tenant blob even without the write facade"
    );

    // Confirm the error is an ABORT (SQLite error code 4 or message contains ABORT).
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("tenant_id mismatch") || err_msg.contains("ABORT"),
        "error must reference the trigger's RAISE(ABORT) message; got: {err_msg}"
    );
}

/// (1) Insert via the write facade path with WRONG tenant -> trigger layer refuses.
/// We simulate the facade assertion failure by attempting the INSERT at the SQL layer
/// (the Rust facade assertion would fire first in production, but the trigger is the
/// last-resort backstop). We verify both layers are present independently.
///
/// The facade guard (Rust assert) is not a live SQL function yet (it will be added
/// as part of the write-facade implementation in later features). Here we verify:
///   (a) The trigger alone guards the insert (already proven by the direct-SQL test).
///   (b) A correct-tenant insert SUCCEEDS (proving the trigger is not always-abort).
#[tokio::test]
async fn correct_tenant_insert_succeeds_wrong_tenant_fails() {
    let (pool, _tmp) = fresh_store().await;
    seed_meta_and_branch(&pool, "tenant-a", "b1").await;

    // Correct tenant -> must succeed.
    let ok = insert_blob_raw(&pool, "ck-ok", "tenant-a").await;
    assert!(
        ok.is_ok(),
        "correct-tenant blob insert must succeed; err: {:?}",
        ok.err()
    );

    // Wrong tenant -> must fail.
    let bad = insert_blob_raw(&pool, "ck-bad", "tenant-b").await;
    assert!(
        bad.is_err(),
        "wrong-tenant blob insert must be refused by the trigger"
    );
}

/// (3) Trigger reads `store_meta`, NOT `NEW.tenant_id` reflexively.
/// A reflexive trigger compares NEW.tenant_id to itself — always true, vacuously
/// accepting every insert (SEC-N02 / DATA-NIT-01 prohibition).
///
/// We verify non-vacuity by inspecting the trigger body from SQLite's schema table
/// and confirming it contains the string "store_meta" (the external reference).
#[tokio::test]
async fn trigger_body_references_store_meta_not_reflexive() {
    let (pool, _tmp) = fresh_store().await;

    let trigger_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='blobs_bi'",
    )
    .fetch_one(&pool)
    .await
    .expect("blobs_bi trigger must exist in sqlite_master");

    // Must reference store_meta (the separate source of truth).
    assert!(
        trigger_sql.contains("store_meta"),
        "trigger body must read from store_meta; got: {trigger_sql}"
    );

    // Must NOT compare NEW.tenant_id to NEW.tenant_id (reflexive tautology).
    // A reflexive check would look like: NEW.tenant_id != NEW.tenant_id (vacuously false).
    // We check that `NEW.tenant_id` appears only once (compared to store_meta, not itself).
    let new_tenant_occurrences = trigger_sql.matches("NEW.tenant_id").count();
    assert_eq!(
        new_tenant_occurrences, 1,
        "trigger must compare NEW.tenant_id to store_meta exactly once, not to itself; \
         got {new_tenant_occurrences} occurrences in: {trigger_sql}"
    );
}

/// AC-F3.4 round-trip: store_meta row correctly allows the matching tenant and
/// the trigger fires correctly when the row is present (not when store_meta is empty).
#[tokio::test]
async fn trigger_requires_store_meta_row_to_be_populated() {
    // Empty store_meta -> the trigger's SELECT returns NULL -> NULL != any string
    // is NULL in SQLite, so WHEN clause is NULL (falsy) -> trigger does NOT fire.
    // This means with no store_meta row, ANY tenant passes (open DB before registration).
    // After F13/registration populates the row, the trigger becomes effective.
    //
    // This test documents the expected behavior:
    // - Empty store_meta: insert proceeds (no row to compare against).
    // - Populated store_meta: only matching tenant passes.
    let (pool, _tmp) = fresh_store().await;

    // No store_meta row yet — branch insert still needs to succeed.
    sqlx::query(
        "INSERT INTO branches(branch_id, branch_name, location, created_at, updated_at) \
         VALUES ('b1','main','/repo','2024-01-01','2024-01-01')",
    )
    .execute(&pool)
    .await
    .expect("branch");

    // Without store_meta row: any tenant insert passes (WHEN clause evaluates NULL).
    let result_no_meta = insert_blob_raw(&pool, "ck-no-meta", "any-tenant").await;
    assert!(
        result_no_meta.is_ok(),
        "with empty store_meta, WHEN clause is NULL (falsy) so any tenant passes; \
         this is expected — F13/registration sets the row before writes begin"
    );

    // Now set store_meta.
    sqlx::query("INSERT INTO store_meta(tenant_id) VALUES ('owner-tenant')")
        .execute(&pool)
        .await
        .expect("store_meta");

    // Correct tenant still passes.
    let ok = insert_blob_raw(&pool, "ck-owner", "owner-tenant").await;
    assert!(ok.is_ok(), "owner tenant must pass after store_meta is set");

    // Wrong tenant now fails.
    let bad = insert_blob_raw(&pool, "ck-wrong", "other-tenant").await;
    assert!(
        bad.is_err(),
        "wrong tenant must be rejected after store_meta row is populated"
    );
}
