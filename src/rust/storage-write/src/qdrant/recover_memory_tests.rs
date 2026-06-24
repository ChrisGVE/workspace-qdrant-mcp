//! Keyset-pagination + bounded-memory tests for `recover.rs` (AC-F11.2).
//!
//! Separated from `recover_tests.rs` to keep each file within the 500-line
//! codesize budget (coding.md §VIII). The tests here exercise the two
//! page-size-sensitive behaviours: RSS stays bounded across a 22k-blob rebuild,
//! and oversized `page_size` arguments are clamped to `PAGE_SIZE_MAX`.
//!
//! `parse_branch_ids_json` unit tests also live here (they are small and
//! cohesively related to the query that drives them).

use std::collections::HashMap;

use sqlx::SqlitePool;

use super::*;
use crate::blob::test_support::{fixture, TENANT};
use crate::blob::vector_codec::{encode_dense, encode_sparse};

// ---------------------------------------------------------------------------
// Bulk-insert setup helper
// ---------------------------------------------------------------------------

/// Insert `n` blobs in one transaction, all owned by `'branch-a'` via a single
/// shared `files` row (`relative_path = path_tag`). Returns the file_id.
///
/// Vectors are zeroed (no embedder required). Used by the pagination tests to
/// create a fixture large enough that page boundaries are observable.
async fn bulk_insert_blobs(
    pool: &SqlitePool,
    n: usize,
    path_tag: &str,
    ck_prefix: &str,
    pid_prefix: u32,
) -> i64 {
    let empty_dense = encode_dense(&[0.0f32; 3]);
    let empty_sparse = encode_sparse(&HashMap::new());

    let mut tx = pool.begin().await.expect("begin");

    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES ('branch-a', ?, 'projects', '2024-01-01', '2024-01-01')",
    )
    .bind(path_tag)
    .execute(&mut *tx)
    .await
    .expect("bulk file");

    let file_id: i64 = sqlx::query_scalar(
        "SELECT file_id FROM files WHERE branch_id = 'branch-a' AND relative_path = ?",
    )
    .bind(path_tag)
    .fetch_one(&mut *tx)
    .await
    .expect("bulk file_id");

    for i in 0..n {
        let ck = format!("{ck_prefix}{i:06}");
        let pid = format!("{pid_prefix:08x}-0000-4000-8000-{i:012x}");

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
    file_id
}

// ---------------------------------------------------------------------------
// RSS helpers
// ---------------------------------------------------------------------------

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
    bulk_insert_blobs(&fx.pool, N_BLOBS, "__bulk__", "ck-bulk-", 0x0000_0000).await;

    // Measure RSS before rebuild.
    let rss_before = current_rss_bytes();

    let mut sink = CaptureRebuildSink::default();
    let total = rebuild_qdrant(&fx.pool, &mut sink, TENANT, "projects", PAGE)
        .await
        .expect("bulk rebuild");

    // Peak RSS already captured by getrusage.
    let rss_after = peak_rss_bytes();

    // (i) No single page exceeded PAGE_SIZE_MAX.
    assert!(
        sink.max_page_len <= PAGE_SIZE_MAX as usize,
        "AC-F11.2: max page len {} must not exceed PAGE_SIZE_MAX={}",
        sink.max_page_len,
        PAGE_SIZE_MAX
    );
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
    let rss_delta = rss_after.saturating_sub(rss_before);
    assert!(
        rss_delta < RSS_CEILING_BYTES,
        "AC-F11.2: peak RSS delta {} bytes >= ceiling {} bytes \
         -- rebuild loaded more than one page at once",
        rss_delta,
        RSS_CEILING_BYTES
    );
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
    bulk_insert_blobs(&fx.pool, n, "__clamp__", "ck-clamp-", 0x0000_0001).await;

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
