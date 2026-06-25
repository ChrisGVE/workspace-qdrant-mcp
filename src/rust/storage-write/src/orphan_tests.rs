//! Tests (i)-(iii),(v) for AC-F16.5 — library-doc orphan migration.
//!
//! File: `wqm-storage-write/src/orphan_tests.rs`
//! Location: `src/rust/storage-write/src/` (write-crate — sibling of orphan.rs)
//! Context: Extracted via `#[path]` for codesize compliance (coding.md §X).
//!   Tests (iv) and (vi) live in `orphan_tests2.rs` (second `#[path]` mod).
//!
//! ## Tests in this file
//!
//! (i)   Duplicate doc in global library → project copy dropped, global copy kept.
//! (ii)  Unique doc → re-homed under global library tenant, still searchable.
//! (iii) No orphaned Qdrant points or blob rows after either path.
//! (v)   Re-home emits the INFO audit-log event (SEC-F16-01).

use super::*;
use crate::blob::ladder::{CaptureSink, QdrantOp};
use crate::library::{open_library_store, LIBRARY_SENTINEL_BRANCH_ID};

use sqlx::SqlitePool;
use std::collections::HashMap;
use tempfile::TempDir;
use tracing_test::traced_test;
use wqm_common::hashing::{bucket, content_key_v4, point_id as derive_point_id};

use crate::blob::vector_codec::{encode_dense, encode_sparse};
use crate::connection::open_store_write;
use crate::schema::ddl_statements;

// ---------------------------------------------------------------------------
// Test constants (pub(crate) so orphan_tests2.rs can reuse)
// ---------------------------------------------------------------------------

pub(crate) const PROJECT_TENANT: &str = "project-tenant-uuid";
pub(crate) const GLOBAL_TENANT: &str = "global-library-tenant-uuid";
pub(crate) const COLLECTION_ID: &str = "libraries";
pub(crate) const BRANCH_A: &str = "branch-test-a";

// ---------------------------------------------------------------------------
// Fixture helpers (pub(crate) so orphan_tests2.rs can reuse)
// ---------------------------------------------------------------------------

/// Open a project store with schema + store_meta + one branch row.
pub(crate) async fn project_fixture() -> (SqlitePool, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("project.db");
    let pool = open_store_write(&path).await.expect("open project store");
    for stmt in ddl_statements() {
        sqlx::query(stmt).execute(&pool).await.expect("ddl");
    }
    sqlx::query("INSERT INTO store_meta(tenant_id) VALUES (?)")
        .bind(PROJECT_TENANT)
        .execute(&pool)
        .await
        .expect("store_meta");
    sqlx::query(
        "INSERT INTO branches(branch_id, branch_name, location, active, sync_state, \
         created_at, updated_at) VALUES (?, 'main', '/repo', 1, 'current', \
         '2026-01-01', '2026-01-01')",
    )
    .bind(BRANCH_A)
    .execute(&pool)
    .await
    .expect("branch");
    (pool, dir)
}

/// Open a global library store (sentinel branch inserted by open_library_store).
pub(crate) async fn global_fixture() -> (SqlitePool, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("global.db");
    let pool = open_library_store(&path, GLOBAL_TENANT)
        .await
        .expect("open global store");
    (pool, dir)
}

/// Insert a library-collection file in the project store, with one blob chunk.
/// Returns (file_id, blob_id, chunk_content_hash, point_id_str).
pub(crate) async fn insert_library_doc(
    pool: &SqlitePool,
    chunk_text: &str,
    branch_id: &str,
) -> (i64, i64, String, String) {
    let chunk_hash = format!("{:x}", sha2_hash(chunk_text));
    let ck = content_key_v4(PROJECT_TENANT, bucket::CODE, &chunk_hash, "");
    let pid = derive_point_id(&ck, 0).to_string();
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());

    let blob_id: i64 = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, '2026-01-01')",
    )
    .bind(&ck)
    .bind(&chunk_hash)
    .bind(&pid)
    .bind(PROJECT_TENANT)
    .bind(chunk_text)
    .bind(dense)
    .bind(sparse)
    .execute(pool)
    .await
    .expect("blob insert")
    .last_insert_rowid();

    let file_id: i64 = sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, 'lib/doc.md', ?, '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .bind(COLLECTION_ID)
    .execute(pool)
    .await
    .expect("file insert")
    .last_insert_rowid();

    sqlx::query(
        "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?, ?, 0, ?)",
    )
    .bind(branch_id)
    .bind(file_id)
    .bind(blob_id)
    .execute(pool)
    .await
    .expect("blob_ref");

    (file_id, blob_id, chunk_hash, pid)
}

/// Insert an identical doc (same chunk hash) into the global store.
pub(crate) async fn insert_global_doc(global_pool: &SqlitePool, chunk_hash: &str) {
    let ck = content_key_v4(GLOBAL_TENANT, bucket::CODE, chunk_hash, "");
    let pid = derive_point_id(&ck, 0).to_string();
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());

    let blob_id: i64 = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, ?, ?, ?, 'global-text', ?, ?, '2026-01-01')",
    )
    .bind(&ck)
    .bind(chunk_hash)
    .bind(&pid)
    .bind(GLOBAL_TENANT)
    .bind(dense)
    .bind(sparse)
    .execute(global_pool)
    .await
    .expect("global blob insert")
    .last_insert_rowid();

    let file_id: i64 = sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, 'lib/global.md', ?, '2026-01-01', '2026-01-01')",
    )
    .bind(LIBRARY_SENTINEL_BRANCH_ID)
    .bind(COLLECTION_ID)
    .execute(global_pool)
    .await
    .expect("global file insert")
    .last_insert_rowid();

    sqlx::query(
        "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?, ?, 0, ?)",
    )
    .bind(LIBRARY_SENTINEL_BRANCH_ID)
    .bind(file_id)
    .bind(blob_id)
    .execute(global_pool)
    .await
    .expect("global blob_ref");
}

/// SHA-256 of a string as a hex-formattable digest.
pub(crate) fn sha2_hash(s: &str) -> impl std::fmt::LowerHex {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    h.finalize()
}

pub(crate) async fn row_count(pool: &SqlitePool, table: &'static str) -> i64 {
    sqlx::query_scalar::<_, i64>(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(pool)
        .await
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Test (i): duplicate doc in global → project copy dropped, global copy kept
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_i_duplicate_doc_project_copy_dropped_global_survives() {
    let (proj, _p_dir) = project_fixture().await;
    let (glob, _g_dir) = global_fixture().await;

    let (file_id, blob_id, chunk_hash, proj_pid) =
        insert_library_doc(&proj, "hello library content", BRANCH_A).await;
    insert_global_doc(&glob, &chunk_hash).await;

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
    .expect("migrate i");

    assert_eq!(summary.dropped, 1, "(i) dropped count");
    assert_eq!(summary.rehomed, 0, "(i) rehomed count");

    let proj_refs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blob_refs WHERE file_id = ?")
        .bind(file_id)
        .fetch_one(&proj)
        .await
        .expect("refs count");
    assert_eq!(proj_refs, 0, "(i) project blob_refs gone after drop");

    let proj_files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files WHERE file_id = ?")
        .bind(file_id)
        .fetch_one(&proj)
        .await
        .expect("files count");
    assert_eq!(proj_files, 0, "(i) project files row gone after drop");

    let proj_blobs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM blobs WHERE blob_id = ?")
        .bind(blob_id)
        .fetch_one(&proj)
        .await
        .expect("blobs count");
    assert_eq!(proj_blobs, 0, "(i) project blob row GC'd after drop");

    assert_eq!(
        row_count(&glob, "blobs").await,
        1,
        "(i) global blob still present"
    );

    let has_delete = sink.ops.iter().any(|op| {
        matches!(op, QdrantOp::Delete { point_id, collection }
            if point_id == &proj_pid && collection == COLLECTION_ID)
    });
    assert!(
        has_delete,
        "(i) QdrantOp::Delete for project copy's point_id"
    );
}

// ---------------------------------------------------------------------------
// Test (ii): unique doc → re-homed, still searchable from global
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_ii_unique_doc_rehomed_to_global() {
    let (proj, _p_dir) = project_fixture().await;
    let (glob, _g_dir) = global_fixture().await;
    let chunk_text = "unique doc content only in project";

    let (_, _, _, proj_pid) = insert_library_doc(&proj, chunk_text, BRANCH_A).await;

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
    .expect("migrate ii");

    assert_eq!(summary.rehomed, 1, "(ii) rehomed count");
    assert_eq!(summary.dropped, 0, "(ii) dropped count");

    let glob_tenant: String =
        sqlx::query_scalar("SELECT tenant_id FROM blobs WHERE chunk_content_hash = ?")
            .bind(&format!("{:x}", sha2_hash(chunk_text)))
            .fetch_one(&glob)
            .await
            .expect("global blob tenant");
    assert_eq!(
        glob_tenant, GLOBAL_TENANT,
        "(ii) global blob has global tenant_id"
    );

    let has_overwrite = sink.ops.iter().any(|op| {
        matches!(op, QdrantOp::OverwritePayload { point_id, payload }
            if point_id == &proj_pid
            && payload.tenant_id == GLOBAL_TENANT
            && payload.branch_id == [LIBRARY_SENTINEL_BRANCH_ID])
    });
    assert!(
        has_overwrite,
        "(ii) OverwritePayload with global tenant for original point_id"
    );

    assert_eq!(
        row_count(&glob, "files").await,
        1,
        "(ii) global store has files row"
    );
}

// ---------------------------------------------------------------------------
// Test (iii): no orphaned Qdrant points or blob rows after either path
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_iii_no_orphaned_rows_after_migration() {
    let (proj, _p_dir) = project_fixture().await;
    let (glob, _g_dir) = global_fixture().await;

    // Doc A: has duplicate in global → DROP path.
    let (_, _, hash_a, _) = insert_library_doc(&proj, "chunk-a content", BRANCH_A).await;
    insert_global_doc(&glob, &hash_a).await;

    // Doc B: unique → RE-HOME path (different path to avoid UNIQUE constraint).
    insert_project_doc_at_path(&proj, "chunk-b different", "lib/doc_b.md").await;

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
    .expect("migrate iii");

    assert_eq!(summary.dropped, 1, "(iii) one doc dropped");
    assert_eq!(summary.rehomed, 1, "(iii) one doc rehomed");

    let proj_lib_files: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM files WHERE collection != 'projects'")
            .fetch_one(&proj)
            .await
            .expect("count");
    assert_eq!(proj_lib_files, 0, "(iii) no library files in project store");

    assert_eq!(
        row_count(&proj, "blob_refs").await,
        0,
        "(iii) no orphaned blob_refs"
    );
    assert_eq!(
        row_count(&glob, "blobs").await,
        2,
        "(iii) 2 blobs in global store"
    );
}

// ---------------------------------------------------------------------------
// Invariant test: re-home preserves point_id + content_key (payload-only model)
// ---------------------------------------------------------------------------
/// The destination `blobs.point_id` stored in the global store MUST equal the
/// `point_id` used in `QdrantOp::OverwritePayload`, and both MUST equal the
/// source chunk's original `point_id`. This validates the payload-only re-home
/// model (DATA-05/SEC-4): no re-embedding, no point recreation.
#[tokio::test]
async fn test_rehome_preserves_point_id_and_content_key_invariant() {
    let (proj, _p_dir) = project_fixture().await;
    let (glob, _g_dir) = global_fixture().await;
    let chunk_text = "invariant-test content";

    let (_, _, chunk_hash, src_pid) = insert_library_doc(&proj, chunk_text, BRANCH_A).await;
    let src_ck = content_key_v4(PROJECT_TENANT, bucket::CODE, &chunk_hash, "");

    let mut sink = CaptureSink::default();
    migrate_project_library_docs(
        &proj,
        &glob,
        &mut sink,
        PROJECT_TENANT,
        GLOBAL_TENANT,
        COLLECTION_ID,
    )
    .await
    .expect("migrate invariant");

    // Destination row point_id must equal source point_id (preserved verbatim).
    let glob_pid: String =
        sqlx::query_scalar("SELECT point_id FROM blobs WHERE chunk_content_hash = ?")
            .bind(&chunk_hash)
            .fetch_one(&glob)
            .await
            .expect("global point_id");
    assert_eq!(
        glob_pid, src_pid,
        "dest point_id must equal source point_id"
    );

    // Destination row content_key must equal source content_key (preserved verbatim).
    let glob_ck: String =
        sqlx::query_scalar("SELECT content_key FROM blobs WHERE chunk_content_hash = ?")
            .bind(&chunk_hash)
            .fetch_one(&glob)
            .await
            .expect("global content_key");
    assert_eq!(
        glob_ck, src_ck,
        "dest content_key must equal source content_key"
    );

    // OverwritePayload must target the same point_id.
    let overwrite_pid = sink.ops.iter().find_map(|op| {
        if let QdrantOp::OverwritePayload { point_id, .. } = op {
            Some(point_id.clone())
        } else {
            None
        }
    });
    assert_eq!(
        overwrite_pid.as_deref(),
        Some(src_pid.as_str()),
        "OverwritePayload point_id must equal source point_id"
    );
}

/// Insert a library doc at a specific path (avoids the `UNIQUE(branch_id, relative_path)` constraint).
pub(crate) async fn insert_project_doc_at_path(pool: &SqlitePool, chunk_text: &str, path: &str) {
    let chunk_hash = format!("{:x}", sha2_hash(chunk_text));
    let ck = content_key_v4(PROJECT_TENANT, bucket::CODE, &chunk_hash, "");
    let pid = derive_point_id(&ck, 0).to_string();
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());
    let blob_id: i64 = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) VALUES (?,?,?,?,?,?,?,'2026-01-01')",
    )
    .bind(&ck)
    .bind(&chunk_hash)
    .bind(&pid)
    .bind(PROJECT_TENANT)
    .bind(chunk_text)
    .bind(dense)
    .bind(sparse)
    .execute(pool)
    .await
    .expect("extra blob")
    .last_insert_rowid();
    let file_id: i64 = sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, ?, '2026-01-01', '2026-01-01')",
    )
    .bind(BRANCH_A)
    .bind(path)
    .bind(COLLECTION_ID)
    .execute(pool)
    .await
    .expect("extra file")
    .last_insert_rowid();
    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)")
        .bind(BRANCH_A)
        .bind(file_id)
        .bind(blob_id)
        .execute(pool)
        .await
        .expect("extra ref");
}

// ---------------------------------------------------------------------------
// Test (v): re-home emits INFO audit-log event (SEC-F16-01)
// ---------------------------------------------------------------------------
#[tokio::test]
#[traced_test]
async fn test_v_rehome_emits_audit_log_event() {
    let (proj, _p_dir) = project_fixture().await;
    let (glob, _g_dir) = global_fixture().await;

    insert_library_doc(&proj, "auditable unique content", BRANCH_A).await;

    let mut sink = CaptureSink::default();
    migrate_project_library_docs(
        &proj,
        &glob,
        &mut sink,
        PROJECT_TENANT,
        GLOBAL_TENANT,
        COLLECTION_ID,
    )
    .await
    .expect("migrate v");

    assert!(
        logs_contain("orphan-migrated"),
        "(v) audit log 'orphan-migrated' missing"
    );
    assert!(
        logs_contain(PROJECT_TENANT),
        "(v) audit log missing project_tenant_id"
    );
    assert!(
        logs_contain(GLOBAL_TENANT),
        "(v) audit log missing global_tenant_id"
    );
}
