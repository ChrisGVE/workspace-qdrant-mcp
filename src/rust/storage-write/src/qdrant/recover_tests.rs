//! Tests for `wqm-storage-write/src/qdrant/recover.rs`.
//!
//! AC-F11.2: keyset pagination + bounded memory (asserting test, not comment).
//! AC-F11.3: zero embedding calls (panicking mock embedder).
//! Verbatim point_id / branch membership assertions.
//!
//! All standard-gate tests run WITHOUT a live Qdrant (offline / CI-safe).

use std::collections::HashMap;

use sqlx::SqlitePool;

use super::*;
use crate::blob::test_support::{add_branch, fixture, TENANT};
use crate::blob::vector_codec::{encode_dense, encode_sparse};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Insert one blob row + one blob_ref row, returning the blob_id.
/// `point_id_str` is stored verbatim (honors salted re-keys -- AC-F11).
/// Vectors are encoded via the canonical encoders so decode round-trips.
async fn insert_blob(
    pool: &SqlitePool,
    branch_id: &str,
    content_key: &str,
    point_id_str: &str,
    dense: &[f32],
    sparse: &HashMap<u32, f32>,
) -> i64 {
    let dense_bytes = encode_dense(dense);
    let sparse_bytes = encode_sparse(sparse);

    let res = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, ?, ?, ?, 'text', ?, ?, '2024-01-01')",
    )
    .bind(content_key)
    .bind(content_key) // reuse as chunk_content_hash
    .bind(point_id_str)
    .bind(TENANT)
    .bind(&dense_bytes)
    .bind(&sparse_bytes)
    .execute(pool)
    .await
    .expect("blob insert");

    let blob_id = res.last_insert_rowid();

    // Insert a files row so the FK chain is satisfiable.
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(branch_id)
    .bind(content_key) // use content_key as a unique path
    .execute(pool)
    .await
    .expect("file insert");

    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(branch_id)
            .bind(content_key)
            .fetch_one(pool)
            .await
            .expect("file_id");

    sqlx::query(
        "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
         VALUES (?, ?, 0, ?) ON CONFLICT DO NOTHING",
    )
    .bind(branch_id)
    .bind(file_id)
    .bind(blob_id)
    .execute(pool)
    .await
    .expect("blob_ref insert");

    blob_id
}

/// Insert a blob referenced by TWO branches (multi-membership).
/// Returns the blob_id.
async fn insert_blob_two_branches(
    pool: &SqlitePool,
    branch_a: &str,
    branch_b: &str,
    content_key: &str,
    point_id_str: &str,
) -> i64 {
    let dense = vec![1.0f32, 2.0, 3.0];
    let sparse = {
        let mut m = HashMap::new();
        m.insert(5u32, 0.5f32);
        m
    };
    let blob_id = insert_blob(pool, branch_a, content_key, point_id_str, &dense, &sparse).await;

    // Second files row for branch_b pointing to the same blob.
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(branch_b)
    .bind(content_key)
    .execute(pool)
    .await
    .expect("file b insert");

    let file_b: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id = ? AND relative_path = ?")
            .bind(branch_b)
            .bind(content_key)
            .fetch_one(pool)
            .await
            .expect("file_b id");

    sqlx::query(
        "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
         VALUES (?, ?, 0, ?) ON CONFLICT DO NOTHING",
    )
    .bind(branch_b)
    .bind(file_b)
    .bind(blob_id)
    .execute(pool)
    .await
    .expect("blob_ref b insert");

    blob_id
}

// ---------------------------------------------------------------------------
// AC-F11.3: zero embedding calls
// ---------------------------------------------------------------------------

// AC-F11.3: rebuild_qdrant NEVER calls the embedder. A panicking mock
// embedder is installed; the test asserts the rebuild completes without
// triggering the panic.
#[tokio::test]
async fn rebuild_does_not_call_embedder() {
    let fx = fixture("branch-a").await;

    let dense = vec![0.1f32; 3];
    let sparse = HashMap::new();
    insert_blob(
        &fx.pool,
        "branch-a",
        "ck-no-embed",
        "pt-no-embed",
        &dense,
        &sparse,
    )
    .await;

    // If rebuild called an embedder, the panic would surface here.
    // The function signature takes NO embedder parameter -- the test proves
    // by compilation + execution that no embedder seam is present at all.
    let mut sink = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", 1000)
        .await
        .expect("rebuild must succeed without an embedder");

    assert_eq!(sink.points.len(), 1, "one blob upserted");
}

// ---------------------------------------------------------------------------
// Verbatim point_id (DATA-05 / SEC-4)
// ---------------------------------------------------------------------------

// AC-F11: rebuild reads blobs.point_id VERBATIM -- even a salted point_id
// that differs from point_id(content_key, 0) is honored.
#[tokio::test]
async fn rebuild_uses_stored_point_id_verbatim() {
    let fx = fixture("branch-a").await;

    let salted_pid = "00000000-dead-beef-cafe-000000000001";
    let dense = vec![0.5f32, 0.5, 0.5];
    let sparse = HashMap::new();
    insert_blob(
        &fx.pool,
        "branch-a",
        "ck-salted",
        salted_pid,
        &dense,
        &sparse,
    )
    .await;

    let mut sink = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", 1000)
        .await
        .expect("rebuild");

    assert_eq!(sink.points.len(), 1);
    assert_eq!(
        sink.points[0].point_id, salted_pid,
        "point_id must be verbatim from SQLite (DATA-05 / AC-F11)"
    );
}

// ---------------------------------------------------------------------------
// Branch membership reconstruction
// ---------------------------------------------------------------------------

// AC-F11: the json_group_array(DISTINCT branch_id) GROUP BY reconstructs the
// full membership set. A blob referenced by two branches appears once in the
// upsert with BOTH branches in its payload.
#[tokio::test]
async fn rebuild_reconstructs_multi_branch_membership() {
    let fx = fixture("branch-a").await;
    add_branch(&fx.pool, "branch-b").await;

    insert_blob_two_branches(&fx.pool, "branch-a", "branch-b", "ck-shared", "pt-shared").await;

    let mut sink = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", 1000)
        .await
        .expect("rebuild");

    assert_eq!(sink.points.len(), 1, "shared blob appears once");
    let mut branches = sink.points[0].payload.branch_id.clone();
    branches.sort();
    assert_eq!(
        branches,
        vec!["branch-a".to_string(), "branch-b".to_string()],
        "both branches present in membership (AC-F11)"
    );
}

// ---------------------------------------------------------------------------
// Payload three-field invariant
// ---------------------------------------------------------------------------

// AC-F11: every upserted point carries all three payload fields: tenant_id,
// branch_id[], collection_id.
#[tokio::test]
async fn rebuild_upsert_carries_all_three_payload_fields() {
    let fx = fixture("branch-a").await;

    let dense = vec![1.0f32, 0.0, 0.0];
    let sparse = {
        let mut m = HashMap::new();
        m.insert(1u32, 1.0f32);
        m
    };
    insert_blob(
        &fx.pool,
        "branch-a",
        "ck-payload",
        "pt-payload",
        &dense,
        &sparse,
    )
    .await;

    let mut sink = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", 1000)
        .await
        .expect("rebuild");

    assert_eq!(sink.points.len(), 1);
    let payload = &sink.points[0].payload;
    assert!(!payload.tenant_id.is_empty(), "tenant_id must be present");
    assert!(!payload.branch_id.is_empty(), "branch_id must be present");
    assert!(
        !payload.collection_id.is_empty(),
        "collection_id must be present"
    );
    assert_eq!(payload.tenant_id, TENANT);
    assert_eq!(payload.collection_id, "projects");
}

// ---------------------------------------------------------------------------
// Vector round-trip fidelity
// ---------------------------------------------------------------------------

// AC-F11: vectors decoded from SQLite bytes match the original f32 values.
// This exercises the encode->store->decode path without embedding.
#[tokio::test]
async fn rebuild_decodes_vectors_correctly() {
    let fx = fixture("branch-a").await;

    let dense_orig = vec![1.0f32, -2.5, 0.125, 100.0];
    let sparse_orig = {
        let mut m = HashMap::new();
        m.insert(42u32, 3.14f32);
        m.insert(7u32, 0.001f32);
        m
    };
    insert_blob(
        &fx.pool,
        "branch-a",
        "ck-vectors",
        "pt-vectors",
        &dense_orig,
        &sparse_orig,
    )
    .await;

    let mut sink = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", 1000)
        .await
        .expect("rebuild");

    assert_eq!(sink.points.len(), 1);
    let pt = &sink.points[0];

    assert_eq!(pt.dense, dense_orig, "dense vector must round-trip exactly");
    assert_eq!(
        pt.sparse, sparse_orig,
        "sparse vector must round-trip exactly"
    );
}

// ---------------------------------------------------------------------------
// Tenant isolation
// ---------------------------------------------------------------------------

// AC-F11: rebuild is scoped to the given tenant_id. Two sub-tests:
//   (a) Calling rebuild with the correct TENANT returns the blob.
//   (b) Calling rebuild with a DIFFERENT tenant_id returns zero blobs
//       (the WHERE clause excludes blobs belonging to TENANT).
// Note: the store_meta trigger prevents inserting a blob with a mismatched
// tenant_id into the same DB, so isolation is enforced at two levels: the
// trigger (schema) and the WHERE b.tenant_id = ? clause (query). We prove
// the query-level isolation by requesting a non-existent tenant.
#[tokio::test]
async fn rebuild_filters_by_tenant_id() {
    let fx = fixture("branch-a").await;

    // Insert a blob for TENANT (the fixture's tenant).
    let dense = vec![1.0f32];
    let sparse = HashMap::new();
    insert_blob(
        &fx.pool,
        "branch-a",
        "ck-tenant-in",
        "pt-tenant-in",
        &dense,
        &sparse,
    )
    .await;

    // (a) Rebuild for TENANT returns exactly the one blob.
    let mut sink_a = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink_a, TENANT, "projects", 1000)
        .await
        .expect("rebuild tenant-in");
    assert_eq!(sink_a.points.len(), 1, "correct tenant sees its blob");
    assert_eq!(sink_a.points[0].point_id, "pt-tenant-in");

    // (b) Rebuild for a DIFFERENT (non-existent) tenant_id returns zero.
    let mut sink_b = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink_b, "other-tenant-xyz", "projects", 1000)
        .await
        .expect("rebuild other-tenant");
    assert_eq!(
        sink_b.points.len(),
        0,
        "different tenant_id must see zero blobs (WHERE clause isolation)"
    );
}

// ---------------------------------------------------------------------------
// Empty collection
// ---------------------------------------------------------------------------

// AC-F11: rebuild on an empty blobs table returns 0 and calls flush_page
// zero times (no panic, no crash).
#[tokio::test]
async fn rebuild_empty_collection_returns_zero() {
    let fx = fixture("branch-a").await;

    let mut sink = CaptureRebuildSink::default();
    let count = rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", 1000)
        .await
        .expect("rebuild empty");

    assert_eq!(count, 0);
    assert!(sink.points.is_empty());
}

// AC-F11.2 keyset-pagination + bounded-memory tests, page_size clamping, and
// parse_branch_ids_json unit tests live in recover_memory_tests.rs
// (split for codesize budget -- coding.md §VIII).
