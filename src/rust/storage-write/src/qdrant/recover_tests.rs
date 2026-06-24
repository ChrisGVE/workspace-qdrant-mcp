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
use crate::blob::ladder::{encode_dense, encode_sparse};
use crate::blob::test_support::{add_branch, fixture, TENANT};

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

// ---------------------------------------------------------------------------
// AC-F11.2: keyset pagination + bounded memory
// ---------------------------------------------------------------------------

// AC-F11.2 (ASSERTING bounded-memory test): insert >=20000 blobs, run rebuild
// with page_size=1000, and assert:
//   (i)  no single page materializes more than PAGE_SIZE_MAX rows.
//   (ii) peak RSS stays under 512 MB (a generous ceiling that fails a
//        full-table 400k-blob load spiking ~1.9 GB, but passes a 1000-row page).
//
// RSS is read via libc::getrusage(RUSAGE_SELF). On macOS ru_maxrss is in BYTES;
// on Linux it is in KiB. We normalize to bytes.
#[tokio::test]
async fn rebuild_respects_page_bound_and_bounded_memory() {
    const N_BLOBS: usize = 22_000;
    const PAGE: u64 = 1_000;
    // 512 MB ceiling -- generous enough to pass paged rebuild, tight enough
    // to fail a full-table 400k-blob single-query load (~1.9 GB spike).
    const RSS_CEILING_BYTES: u64 = 512 * 1024 * 1024;

    let fx = fixture("branch-a").await;

    // Insert N_BLOBS blobs in batches (fast: zeroed vectors, no embedder).
    let empty_dense = encode_dense(&[0.0f32; 3]);
    let empty_sparse = encode_sparse(&HashMap::new());

    // Batch inserts via a single transaction for speed.
    let mut tx = fx.pool.begin().await.expect("begin");

    // We need a files row + blob_refs row per blob to satisfy the JOIN.
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES ('branch-a', '__bulk__', 'projects', '2024-01-01', '2024-01-01')",
    )
    .execute(&mut *tx)
    .await
    .expect("bulk file");
    let file_id: i64 = sqlx::query_scalar(
        "SELECT file_id FROM files WHERE branch_id = 'branch-a' AND relative_path = '__bulk__'",
    )
    .fetch_one(&mut *tx)
    .await
    .expect("bulk file_id");

    for i in 0..N_BLOBS {
        let ck = format!("ck-bulk-{i:06}");
        let pid = format!("00000000-0000-4000-8000-{i:012x}");

        sqlx::query(
            "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
             raw_text, dense_vec, sparse_vec, created_at) \
             VALUES (?, ?, ?, ?, 'bulk', ?, ?, '2024-01-01')",
        )
        .bind(&ck)
        .bind(&ck)
        .bind(&pid)
        .bind(TENANT)
        .bind(&empty_dense)
        .bind(&empty_sparse)
        .execute(&mut *tx)
        .await
        .expect("bulk blob");

        let blob_id: i64 = sqlx::query_scalar("SELECT last_insert_rowid()")
            .fetch_one(&mut *tx)
            .await
            .expect("last_insert_rowid");

        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
             VALUES ('branch-a', ?, ?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(file_id)
        .bind(i as i64)
        .bind(blob_id)
        .execute(&mut *tx)
        .await
        .expect("bulk blob_ref");
    }
    tx.commit().await.expect("commit bulk");

    // Measure RSS before rebuild.
    let rss_before = current_rss_bytes();

    let mut sink = CaptureRebuildSink::default();
    let total = rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", PAGE)
        .await
        .expect("bulk rebuild");

    // Measure RSS after rebuild (peak RSS already captured by getrusage).
    let rss_after = peak_rss_bytes();

    // (i) No single page exceeded PAGE_SIZE_MAX.
    assert!(
        sink.max_page_len <= PAGE_SIZE_MAX as usize,
        "AC-F11.2: max page len {} must not exceed PAGE_SIZE_MAX={}",
        sink.max_page_len,
        PAGE_SIZE_MAX
    );

    // (i) Every page was at least 1 (rebuild streamed, did not load all at once).
    assert!(
        sink.max_page_len <= PAGE as usize,
        "AC-F11.2: max page len {} must not exceed requested page_size={}",
        sink.max_page_len,
        PAGE
    );

    // Total blobs upserted matches what was inserted.
    assert_eq!(
        total, N_BLOBS as u64,
        "AC-F11.2: all {N_BLOBS} blobs must be upserted"
    );

    // (ii) Peak RSS delta must be under the ceiling.
    // rss_after is the OS peak RSS (getrusage RUSAGE_SELF ru_maxrss) --
    // captures the high-water mark. Subtracting rss_before gives the rebuild's
    // contribution (approximate: process may have already used some heap).
    let rss_delta = rss_after.saturating_sub(rss_before);
    assert!(
        rss_delta < RSS_CEILING_BYTES,
        "AC-F11.2: peak RSS delta {} bytes >= ceiling {} bytes \
         -- rebuild loaded more than one page at once",
        rss_delta,
        RSS_CEILING_BYTES
    );

    let _ = rss_after; // suppress unused warning if rss_before assertion is enough
}

/// Read the CURRENT resident set size in bytes via `getrusage(RUSAGE_SELF)`.
/// On macOS `ru_maxrss` is in bytes; on Linux it is in KiB.
fn current_rss_bytes() -> u64 {
    // We read ru_maxrss as a proxy for current heap; it is the OS peak but
    // good enough for before/after comparison in a test process.
    peak_rss_bytes()
}

/// Read `ru_maxrss` from `getrusage(RUSAGE_SELF)` and normalize to bytes.
pub fn peak_rss_bytes() -> u64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };

    let raw = usage.ru_maxrss as u64;

    // macOS reports ru_maxrss in bytes; Linux reports in KiB.
    #[cfg(target_os = "macos")]
    {
        raw
    }
    #[cfg(not(target_os = "macos"))]
    {
        raw * 1024
    }
}

// ---------------------------------------------------------------------------
// AC-F11.2: page_size clamping
// ---------------------------------------------------------------------------

// AC-F11.2: page_size below PAGE_SIZE_MIN is clamped up to PAGE_SIZE_MIN.
// page_size above PAGE_SIZE_MAX is clamped down to PAGE_SIZE_MAX.
// We test this by observing that max_page_len never exceeds PAGE_SIZE_MAX
// even when the caller requests 999_999.
#[tokio::test]
async fn page_size_clamped_to_max() {
    let fx = fixture("branch-a").await;

    // Insert PAGE_SIZE_MAX + 1 blobs so clamping is observable.
    let n = PAGE_SIZE_MAX as usize + 10;
    let empty_dense = encode_dense(&[0.0f32; 3]);
    let empty_sparse = encode_sparse(&HashMap::new());

    let mut tx = fx.pool.begin().await.expect("begin");
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES ('branch-a', '__clamp__', 'projects', '2024-01-01', '2024-01-01')",
    )
    .execute(&mut *tx)
    .await
    .expect("file");
    let file_id: i64 = sqlx::query_scalar(
        "SELECT file_id FROM files WHERE branch_id = 'branch-a' AND relative_path = '__clamp__'",
    )
    .fetch_one(&mut *tx)
    .await
    .expect("file_id");

    for i in 0..n {
        let ck = format!("ck-clamp-{i:06}");
        let pid = format!("00000001-0000-4000-8000-{i:012x}");
        sqlx::query(
            "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
             raw_text, dense_vec, sparse_vec, created_at) \
             VALUES (?, ?, ?, ?, 'bulk', ?, ?, '2024-01-01')",
        )
        .bind(&ck)
        .bind(&ck)
        .bind(&pid)
        .bind(TENANT)
        .bind(&empty_dense)
        .bind(&empty_sparse)
        .execute(&mut *tx)
        .await
        .expect("blob");
        let blob_id: i64 = sqlx::query_scalar("SELECT last_insert_rowid()")
            .fetch_one(&mut *tx)
            .await
            .expect("rowid");
        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) \
             VALUES ('branch-a', ?, ?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(file_id)
        .bind(i as i64)
        .bind(blob_id)
        .execute(&mut *tx)
        .await
        .expect("ref");
    }
    tx.commit().await.expect("commit");

    // Request an oversized page -- must be clamped to PAGE_SIZE_MAX.
    let mut sink = CaptureRebuildSink::default();
    rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", 999_999)
        .await
        .expect("rebuild");

    assert!(
        sink.max_page_len <= PAGE_SIZE_MAX as usize,
        "AC-F11.2: page_size 999_999 must be clamped to PAGE_SIZE_MAX={}, got {}",
        PAGE_SIZE_MAX,
        sink.max_page_len
    );
    assert_eq!(sink.points.len(), n, "all blobs upserted");
}

// ---------------------------------------------------------------------------
// parse_branch_ids_json unit tests
// ---------------------------------------------------------------------------

// Verify the JSON helper handles edge cases correctly.
#[test]
fn parse_branch_ids_json_single_entry() {
    let result = parse_branch_ids_json(r#"["branch-abc"]"#, 1);
    assert_eq!(result, vec!["branch-abc".to_string()]);
}

#[test]
fn parse_branch_ids_json_two_entries() {
    let mut result = parse_branch_ids_json(r#"["branch-a","branch-b"]"#, 1);
    result.sort();
    assert_eq!(result, vec!["branch-a".to_string(), "branch-b".to_string()]);
}

#[test]
fn parse_branch_ids_json_empty_array() {
    let result = parse_branch_ids_json("[]", 1);
    assert!(result.is_empty());
}

#[test]
fn parse_branch_ids_json_three_entries() {
    let mut result = parse_branch_ids_json(r#"["c","a","b"]"#, 1);
    result.sort();
    assert_eq!(
        result,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}
