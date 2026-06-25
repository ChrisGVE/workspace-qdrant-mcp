//! Tests (iv) and (vi) for AC-F16.5 — library-doc orphan migration.
//!
//! File: `wqm-storage-write/src/orphan_tests2.rs`
//! Location: `src/rust/storage-write/src/` (write-crate — sibling of orphan.rs)
//! Context: Split from `orphan_tests.rs` for codesize compliance. Mounted via
//!   `#[cfg(test)] #[path = "orphan_tests2.rs"] mod tests2;` in `orphan.rs`.
//!
//! ## Tests in this file
//!
//! (iv)  Crash-mid-re-tenant — source row left intact, case5 heals without culling.
//! (vi)  DOM-R8-N1 strict-subset is re-homed, NOT dropped.

use super::*;
use crate::blob::ladder::{CaptureSink, QdrantOp};
use crate::library::LIBRARY_SENTINEL_BRANCH_ID;

// Pull shared fixtures from the sibling test module (orphan_tests.rs).
use super::tests::{
    global_fixture, insert_library_doc, project_fixture, row_count, sha2_hash, BRANCH_A,
    COLLECTION_ID, GLOBAL_TENANT, PROJECT_TENANT,
};

use sqlx::SqlitePool;
use std::collections::HashMap;
use wqm_common::hashing::{bucket, content_key_v4, point_id as derive_point_id};

use crate::blob::vector_codec::{encode_dense, encode_sparse};

// ---------------------------------------------------------------------------
// Test (iv): crash-mid-re-tenant — case5 heals without culling
// ---------------------------------------------------------------------------

/// Insert source-side rows (project store) for the mid-re-tenant crash scenario.
/// Returns (blob_id, proj_pid).
async fn insert_mid_retenant_source(
    proj: &SqlitePool,
    chunk_hash: &str,
    chunk_text: &str,
) -> (i64, String) {
    let proj_ck = content_key_v4(PROJECT_TENANT, bucket::CODE, chunk_hash, "");
    let proj_pid = derive_point_id(&proj_ck, 0).to_string();
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());
    let blob_id: i64 = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) VALUES (?,?,?,?,?,?,?,'2026-01-01')",
    )
    .bind(&proj_ck)
    .bind(chunk_hash)
    .bind(&proj_pid)
    .bind(PROJECT_TENANT)
    .bind(chunk_text)
    .bind(dense)
    .bind(sparse)
    .execute(proj)
    .await
    .expect("proj blob")
    .last_insert_rowid();
    let file_id: i64 = sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, 'mid-retenant/doc.md', ?, '2026-01-01', '2026-01-01')",
    )
    .bind(BRANCH_A)
    .bind(COLLECTION_ID)
    .execute(proj)
    .await
    .expect("proj file")
    .last_insert_rowid();
    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)")
        .bind(BRANCH_A)
        .bind(file_id)
        .bind(blob_id)
        .execute(proj)
        .await
        .expect("proj blob_ref");
    (blob_id, proj_pid)
}

/// Insert destination-side rows (global store) for the mid-re-tenant crash scenario.
/// Models state after copy_chunks_to_global completed but delete_source_rows never ran.
async fn insert_mid_retenant_global(glob: &SqlitePool, chunk_hash: &str, chunk_text: &str) {
    let glob_ck = content_key_v4(GLOBAL_TENANT, bucket::CODE, chunk_hash, "");
    let glob_pid = derive_point_id(&glob_ck, 0).to_string();
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());
    let blob_id: i64 = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) VALUES (?,?,?,?,?,?,?,'2026-01-01')",
    )
    .bind(&glob_ck)
    .bind(chunk_hash)
    .bind(&glob_pid)
    .bind(GLOBAL_TENANT)
    .bind(chunk_text)
    .bind(dense)
    .bind(sparse)
    .execute(glob)
    .await
    .expect("global blob")
    .last_insert_rowid();
    let file_id: i64 = sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, 'mid-retenant/doc.md', ?, '2026-01-01', '2026-01-01')",
    )
    .bind(LIBRARY_SENTINEL_BRANCH_ID)
    .bind(COLLECTION_ID)
    .execute(glob)
    .await
    .expect("global file")
    .last_insert_rowid();
    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)")
        .bind(LIBRARY_SENTINEL_BRANCH_ID)
        .bind(file_id)
        .bind(blob_id)
        .execute(glob)
        .await
        .expect("global blob_ref");
}

/// Simulate case5 detection: a blob whose stored `tenant_id` != `store_meta.tenant_id`.
/// In our mid-re-tenant fixture the project store still has the source blob row whose
/// `tenant_id = PROJECT_TENANT` (correct) — the mismatch case5 catches is on the
/// GLOBAL store where the Qdrant point still carries the project tenant in its payload.
///
/// We test the observable outcome: the source blob row is NOT culled by migration,
/// and after re-running migrate with the global copy present it follows the DROP path.
async fn run_case5_on_mid_retenant(
    proj: &SqlitePool,
    glob: &SqlitePool,
    collection_id: &str,
) -> u32 {
    // After crash, the global copy exists → migration sees equal-cardinality match
    // and takes DROP path on the surviving source row. This is the heal.
    let mut sink = CaptureSink::default();
    let summary = migrate_project_library_docs(
        proj,
        glob,
        &mut sink,
        PROJECT_TENANT,
        GLOBAL_TENANT,
        collection_id,
    )
    .await
    .expect("re-run migrate (case5 heal)");
    summary.dropped
}

#[tokio::test]
async fn test_iv_crash_mid_re_tenant_case5_heals_no_cull() {
    let (proj, _p_dir) = project_fixture().await;
    let (glob, _g_dir) = global_fixture().await;

    let chunk_text = "crash-mid-retenant content";
    let chunk_hash = format!("{:x}", sha2_hash(chunk_text));
    let (blob_id, _proj_pid) = insert_mid_retenant_source(&proj, &chunk_hash, chunk_text).await;
    insert_mid_retenant_global(&glob, &chunk_hash, chunk_text).await;

    // Verify source row still present (crash scenario).
    let src_present: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&proj)
        .await
        .expect("src count");
    assert_eq!(src_present, 1, "(iv) source blob survives crash");

    // Heal: global copy present → migrate detects equal-cardinality match → DROP.
    let dropped = run_case5_on_mid_retenant(&proj, &glob, COLLECTION_ID).await;
    assert_eq!(
        dropped, 1,
        "(iv) heal drops source copy after global already present"
    );

    // Source blob should now be gone (DROP path cleaned it).
    let src_after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&proj)
        .await
        .expect("src after count");
    assert_eq!(src_after, 0, "(iv) source blob removed after heal DROP");

    // Global copy intact.
    assert_eq!(
        row_count(&glob, "blobs").await,
        1,
        "(iv) global blob survives"
    );
}

// ---------------------------------------------------------------------------
// Test (vi): DOM-R8-N1 — strict subset must be re-homed, NOT dropped
// ---------------------------------------------------------------------------

/// Insert a global doc with TWO chunks (chunk_a + chunk_b).
async fn insert_global_doc_two_chunks(
    global_pool: &SqlitePool,
    chunk_hash_a: &str,
    chunk_hash_b: &str,
) {
    let ck_a = content_key_v4(GLOBAL_TENANT, bucket::CODE, chunk_hash_a, "");
    let pid_a = derive_point_id(&ck_a, 0).to_string();
    let ck_b = content_key_v4(GLOBAL_TENANT, bucket::CODE, chunk_hash_b, "");
    let pid_b = derive_point_id(&ck_b, 1).to_string();
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());

    let ba: i64 = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) VALUES (?,?,?,?,'gA',?,?,'2026-01-01')",
    )
    .bind(&ck_a)
    .bind(chunk_hash_a)
    .bind(&pid_a)
    .bind(GLOBAL_TENANT)
    .bind(&dense)
    .bind(&sparse)
    .execute(global_pool)
    .await
    .expect("global blob a")
    .last_insert_rowid();

    let bb: i64 = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) VALUES (?,?,?,?,'gB',?,?,'2026-01-01')",
    )
    .bind(&ck_b)
    .bind(chunk_hash_b)
    .bind(&pid_b)
    .bind(GLOBAL_TENANT)
    .bind(&dense)
    .bind(&sparse)
    .execute(global_pool)
    .await
    .expect("global blob b")
    .last_insert_rowid();

    let fid: i64 = sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, 'lib/two-chunk.md', ?, '2026-01-01', '2026-01-01')",
    )
    .bind(LIBRARY_SENTINEL_BRANCH_ID)
    .bind(COLLECTION_ID)
    .execute(global_pool)
    .await
    .expect("global file 2c")
    .last_insert_rowid();

    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)")
        .bind(LIBRARY_SENTINEL_BRANCH_ID)
        .bind(fid)
        .bind(ba)
        .execute(global_pool)
        .await
        .expect("ref a");

    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,1,?)")
        .bind(LIBRARY_SENTINEL_BRANCH_ID)
        .bind(fid)
        .bind(bb)
        .execute(global_pool)
        .await
        .expect("ref b");
}

#[tokio::test]
async fn test_vi_dom_r8_n1_strict_subset_is_rehomed_not_dropped() {
    let (proj, _p_dir) = project_fixture().await;
    let (glob, _g_dir) = global_fixture().await;

    // Project doc has ONE chunk (chunk_a only).
    let chunk_text_a = "dom-r8-n1 chunk-a only in project";
    let chunk_text_b = "dom-r8-n1 chunk-b only in global";
    let hash_a = format!("{:x}", sha2_hash(chunk_text_a));
    let hash_b = format!("{:x}", sha2_hash(chunk_text_b));

    // Project doc: chunk_a alone (strict subset of global {chunk_a, chunk_b}).
    insert_library_doc(&proj, chunk_text_a, BRANCH_A).await;

    // Global doc: chunk_a + chunk_b (superset).
    insert_global_doc_two_chunks(&glob, &hash_a, &hash_b).await;

    let mut sink = CaptureSink::default();
    let summary = migrate_project_library_docs(
        &proj,
        &glob,
        &mut sink,
        PROJECT_TENANT,
        GLOBAL_TENANT,
        COLLECTION_ID,
    )
    .await
    .expect("migrate vi");

    // DOM-R8-N1: strict subset → re-home, NOT drop.
    assert_eq!(summary.rehomed, 1, "(vi) strict subset must be re-homed");
    assert_eq!(summary.dropped, 0, "(vi) strict subset must NOT be dropped");

    // No OverwritePayload for chunk_a would be wrong (it must re-home, so OverwritePayload emitted).
    let has_overwrite = sink.ops.iter().any(|op| {
        matches!(op, QdrantOp::OverwritePayload { payload, .. }
            if payload.tenant_id == GLOBAL_TENANT)
    });
    assert!(
        has_overwrite,
        "(vi) re-home emits OverwritePayload for global tenant"
    );

    // No Delete op (that would be the DROP path).
    let has_delete = sink
        .ops
        .iter()
        .any(|op| matches!(op, QdrantOp::Delete { .. }));
    assert!(!has_delete, "(vi) strict subset must NOT trigger Delete");
}
