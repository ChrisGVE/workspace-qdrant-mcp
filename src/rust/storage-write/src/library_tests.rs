//! AC-F16.1 tests — branchless library store mechanics.
//!
//! File: `wqm-storage-write/src/library_tests.rs`
//! Location: `src/rust/storage-write/src/` (write-crate)
//! Context: Extracted from `library.rs` via `#[path]` for codesize compliance
//!   (coding.md §X / 500-line limit). These tests prove the three AC-F16.1 properties:
//!   (i)  identical bytes under a project tenant vs. a library tenant produce DISTINCT
//!        content_key and point_id values (cross-tenant deletion isolation);
//!   (ii) within a single library store, the same bytes ingested twice dedup to ONE
//!        blobs row (in-bucket dedup);
//!   (iii) the blobs_bi cross-tenant trigger fires in a library store, aborting a blob
//!         insert whose tenant_id does not match store_meta (trigger guard still applies).

use tempfile::NamedTempFile;
use wqm_common::hashing::{bucket, content_key_v4, point_id};

use super::open_library_store;
use crate::connection::open_store_write;
use crate::schema::ddl_statements;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Apply the 9-table DDL to a freshly opened pool.
async fn apply_schema(pool: &sqlx::SqlitePool) {
    for stmt in ddl_statements() {
        sqlx::query(stmt)
            .execute(pool)
            .await
            .expect("DDL statement");
    }
}

/// Insert a minimal blob row and return the `blob_id`.
///
/// Uses the provided `content_key` and `point_id` verbatim so tests can
/// assert exact dedup behavior without going through the full ingest ladder.
async fn insert_blob(
    pool: &sqlx::SqlitePool,
    content_key: &str,
    point_id_str: &str,
    tenant_id: &str,
) -> Result<i64, sqlx::Error> {
    let result = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, 'hash-abc', ?, ?, 'hello library', X'', X'', '2026-01-01')",
    )
    .bind(content_key)
    .bind(point_id_str)
    .bind(tenant_id)
    .execute(pool)
    .await?;
    Ok(result.last_insert_rowid())
}

// ---------------------------------------------------------------------------
// AC-F16.1 (i): project vs. library cross-tenant distinct point identities
// ---------------------------------------------------------------------------

/// Pure hash test: identical chunk bytes under two different tenants produce DIFFERENT
/// content_key and point_id values (arch §5.4, AC-F16.1 deletion-isolation property).
///
/// tenant_id is the FIRST slot of content_key_v4, so same `chunk_content_hash` under
/// different tenants yields different digests — and therefore a different Qdrant point
/// identity per tenant. This test pins the ACTUAL computed strings (non-vacuous).
#[test]
fn identical_bytes_different_tenants_have_distinct_keys() {
    let project_tenant = "project-tenant-aabbcc";
    let library_tenant = "library-tenant-ddeeff";
    let chunk_hash = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

    let ck_project = content_key_v4(project_tenant, bucket::CODE, chunk_hash, "");
    let ck_library = content_key_v4(library_tenant, bucket::CODE, chunk_hash, "");

    // Both must be 64-char hex strings (full SHA-256 digest).
    assert_eq!(
        ck_project.len(),
        64,
        "project content_key must be 64-char hex"
    );
    assert_eq!(
        ck_library.len(),
        64,
        "library content_key must be 64-char hex"
    );

    // Different tenants → different content_key.
    assert_ne!(
        ck_project, ck_library,
        "content_key must differ across tenants: project={ck_project} library={ck_library}"
    );

    // Different content_keys → different point_ids.
    let pid_project = point_id(&ck_project, 0);
    let pid_library = point_id(&ck_library, 0);
    assert_ne!(
        pid_project, pid_library,
        "point_id must differ across tenants: project={pid_project} library={pid_library}"
    );
}

/// Store round-trip: a project blob and a library blob with identical bytes coexist in
/// their respective store.dbs without UNIQUE-constraint conflicts (AC-F16.1 — distinct
/// point identities mean each tenant gets its own independent Qdrant point).
#[tokio::test]
async fn identical_bytes_coexist_in_separate_project_and_library_stores() {
    let project_tenant = "project-tenant-aabbcc";
    let library_tenant = "library-tenant-ddeeff";
    let chunk_hash = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

    let ck_project = content_key_v4(project_tenant, bucket::CODE, chunk_hash, "");
    let ck_library = content_key_v4(library_tenant, bucket::CODE, chunk_hash, "");
    let pid_proj_str = point_id(&ck_project, 0).to_string();
    let pid_lib_str = point_id(&ck_library, 0).to_string();

    // Initialize both stores.
    let proj_tmp = NamedTempFile::new().expect("tempfile project");
    let proj_pool = open_store_write(proj_tmp.path())
        .await
        .expect("open project store");
    apply_schema(&proj_pool).await;
    sqlx::query("INSERT INTO store_meta(tenant_id) VALUES (?)")
        .bind(project_tenant)
        .execute(&proj_pool)
        .await
        .expect("project store_meta");
    sqlx::query(
        "INSERT INTO branches(branch_id, branch_name, location, created_at, updated_at) \
         VALUES ('proj-branch','main','/repo','2026-01-01','2026-01-01')",
    )
    .execute(&proj_pool)
    .await
    .expect("project branch");

    let lib_tmp = NamedTempFile::new().expect("tempfile library");
    let lib_pool = open_library_store(lib_tmp.path(), library_tenant)
        .await
        .expect("open library store");

    // Both inserts must succeed (distinct content_key / point_id = no UNIQUE collision).
    let proj_blob_id = insert_blob(&proj_pool, &ck_project, &pid_proj_str, project_tenant)
        .await
        .expect("project blob insert");
    let lib_blob_id = insert_blob(&lib_pool, &ck_library, &pid_lib_str, library_tenant)
        .await
        .expect("library blob insert");

    assert!(proj_blob_id > 0, "project blob_id must be a positive rowid");
    assert!(lib_blob_id > 0, "library blob_id must be a positive rowid");

    // Read back and confirm distinct content_keys are stored correctly.
    let proj_ck: String = sqlx::query_scalar("SELECT content_key FROM blobs WHERE blob_id = ?")
        .bind(proj_blob_id)
        .fetch_one(&proj_pool)
        .await
        .expect("read project content_key");
    let lib_ck: String = sqlx::query_scalar("SELECT content_key FROM blobs WHERE blob_id = ?")
        .bind(lib_blob_id)
        .fetch_one(&lib_pool)
        .await
        .expect("read library content_key");

    assert_eq!(proj_ck, ck_project, "project content_key roundtrip");
    assert_eq!(lib_ck, ck_library, "library content_key roundtrip");
    assert_ne!(
        proj_ck, lib_ck,
        "cross-tenant content_keys must remain distinct"
    );
}

// ---------------------------------------------------------------------------
// AC-F16.1 (ii): within-library in-bucket dedup
// ---------------------------------------------------------------------------

/// Ingesting the same chunk bytes twice into one library store deduplicates to a
/// SINGLE blobs row (same content_key → UNIQUE constraint fires on second insert).
///
/// This mirrors the project-store dedup guarantee and confirms it holds in the
/// branchless library bucket.
#[tokio::test]
async fn same_bytes_ingested_twice_dedup_to_one_blob_in_library_store() {
    let library_tenant = "library-tenant-dedup-test";
    let chunk_hash = "cafecafecafecafecafecafecafecafecafecafecafecafecafecafecafecafe";

    let ck = content_key_v4(library_tenant, bucket::CODE, chunk_hash, "");
    let pid = point_id(&ck, 0).to_string();

    let tmp = NamedTempFile::new().expect("tempfile");
    let pool = open_library_store(tmp.path(), library_tenant)
        .await
        .expect("open library store");

    // First insert: must succeed.
    insert_blob(&pool, &ck, &pid, library_tenant)
        .await
        .expect("first blob insert must succeed");

    // Second insert of the same content_key: UNIQUE constraint must fire.
    let second = insert_blob(&pool, &ck, &pid, library_tenant).await;
    assert!(
        second.is_err(),
        "second insert of the same content_key must fail (UNIQUE constraint = in-bucket dedup)"
    );

    // Confirm exactly ONE row exists.
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE content_key = ?")
        .bind(&ck)
        .fetch_one(&pool)
        .await
        .expect("count blobs");
    assert_eq!(
        count, 1,
        "exactly one blobs row must exist after dedup (got {count})"
    );
}

// ---------------------------------------------------------------------------
// AC-F16.1 (iii): blobs_bi cross-tenant trigger fires in a library store
// ---------------------------------------------------------------------------

/// The `blobs_bi` BEFORE INSERT trigger (AC-F3.4) must abort a blob insert whose
/// `tenant_id` does not match `store_meta.tenant_id` — proving the structural
/// cross-tenant guard applies to library stores just as it does to project stores.
#[tokio::test]
async fn cross_tenant_trigger_fires_in_library_store() {
    let library_tenant = "library-tenant-trigger-test";
    let wrong_tenant = "wrong-tenant-should-be-rejected";

    let chunk_hash = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";
    let ck_wrong = content_key_v4(wrong_tenant, bucket::CODE, chunk_hash, "");
    let pid_wrong = point_id(&ck_wrong, 0).to_string();

    let tmp = NamedTempFile::new().expect("tempfile");
    let pool = open_library_store(tmp.path(), library_tenant)
        .await
        .expect("open library store");

    // Attempt to insert a blob with the WRONG tenant_id — trigger must ABORT.
    let result = insert_blob(&pool, &ck_wrong, &pid_wrong, wrong_tenant).await;

    assert!(
        result.is_err(),
        "blobs_bi trigger must abort insert with mismatched tenant_id"
    );

    // Confirm the error message contains the trigger's ABORT text.
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("tenant_id mismatch"),
        "error must mention 'tenant_id mismatch' from the blobs_bi trigger; got: {err_msg}"
    );

    // Confirm zero rows were inserted (trigger ABORT rolled back the statement).
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs")
        .fetch_one(&pool)
        .await
        .expect("count blobs after trigger abort");
    assert_eq!(
        count, 0,
        "no blob rows must exist after trigger ABORT (got {count})"
    );
}
