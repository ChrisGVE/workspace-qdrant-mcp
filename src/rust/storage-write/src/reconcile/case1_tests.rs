//! Tests for reconcile case 1 (AC-F15.1 -- missing Qdrant membership heal).
//!
//! All tests use temp SQLite + `MockQdrantReader` (no live Qdrant).

use super::*;
use crate::blob::ladder::{CaptureSink, QdrantOp};
use crate::blob::test_support::{add_branch, fixture, TENANT};
use crate::blob::vector_codec::{encode_dense, encode_sparse};
use crate::reconcile::seams::MockQdrantReader;
use std::collections::HashMap;

const BRANCH_A: &str = "branch-a";
const BRANCH_B: &str = "branch-b";
const COLL_ID: &str = "projects";
const COLL_NAME: &str = "projects";

/// Insert a blob row with empty vectors and return blob_id.
async fn insert_blob(pool: &sqlx::SqlitePool, point_id: &str) -> i64 {
    let dense_bytes = encode_dense(&[]);
    let sparse_bytes = encode_sparse(&HashMap::new());
    let result = sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, 'hash', ?, ?, 'text', ?, ?, '2026-01-01')",
    )
    .bind(format!("ck-{point_id}"))
    .bind(point_id)
    .bind(TENANT)
    .bind(dense_bytes)
    .bind(sparse_bytes)
    .execute(pool)
    .await
    .expect("blob insert");
    result.last_insert_rowid()
}

/// Insert a blob_ref linking `blob_id` to `branch_id` via a files row.
async fn insert_ref(pool: &sqlx::SqlitePool, branch_id: &str, blob_id: i64, path: &str) {
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .bind(path)
    .execute(pool)
    .await
    .expect("file insert");
    let file_id: i64 =
        sqlx::query_scalar("SELECT file_id FROM files WHERE branch_id=? AND relative_path=?")
            .bind(branch_id)
            .bind(path)
            .fetch_one(pool)
            .await
            .expect("file_id");
    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES (?,?,0,?)")
        .bind(branch_id)
        .bind(file_id)
        .bind(blob_id)
        .execute(pool)
        .await
        .expect("blob_ref insert");
}

// AC-F15.1 case 1: blob present in SQLite + Qdrant -- enqueues OverwritePayload only.
#[tokio::test]
async fn case1_blob_present_in_qdrant_enqueues_overwrite_only() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-a1").await;
    insert_ref(&fx.pool, BRANCH_A, blob_id, "a.rs").await;

    let reader = MockQdrantReader::with_existing(["pt-a1"]);
    let mut sink = CaptureSink::default();

    run_case1(&fx.pool, &mut sink, &reader, 0, COLL_ID, COLL_NAME)
        .await
        .expect("case1");

    // Only OverwritePayload; no Upsert because the point exists.
    let overwrite_count = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::OverwritePayload { .. }))
        .count();
    let upsert_count = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::Upsert { .. }))
        .count();
    assert_eq!(overwrite_count, 1);
    assert_eq!(upsert_count, 0, "point present -> no Upsert enqueued");
}

// AC-F15.1 case 1 sub-case: blob present in SQLite but absent from Qdrant --
// enqueues both OverwritePayload and Upsert (re-upsert from durable vectors).
#[tokio::test]
async fn case1_blob_absent_from_qdrant_enqueues_upsert_too() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-b1").await;
    insert_ref(&fx.pool, BRANCH_A, blob_id, "b.rs").await;

    let reader = MockQdrantReader::all_absent(); // pt-b1 does NOT exist in Qdrant
    let mut sink = CaptureSink::default();

    run_case1(&fx.pool, &mut sink, &reader, 0, COLL_ID, COLL_NAME)
        .await
        .expect("case1");

    let overwrite_count = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::OverwritePayload { .. }))
        .count();
    let upsert_count = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::Upsert { .. }))
        .count();
    assert_eq!(overwrite_count, 1, "always enqueue OverwritePayload");
    assert_eq!(
        upsert_count, 1,
        "absent point -> also Upsert from durable vectors"
    );
}

// AC-F15.4: incremental pass only touches blobs above the watermark.
#[tokio::test]
async fn case1_incremental_watermark_scopes_scan() {
    let fx = fixture(BRANCH_A).await;

    // Insert two blobs with sequentially assigned IDs.
    let old_id = insert_blob(&fx.pool, "pt-old").await;
    insert_ref(&fx.pool, BRANCH_A, old_id, "old.rs").await;

    let new_id = insert_blob(&fx.pool, "pt-new").await;
    insert_ref(&fx.pool, BRANCH_A, new_id, "new.rs").await;

    // Watermark set to old_id -- incremental pass sees only new_id.
    let reader = MockQdrantReader::with_existing(["pt-old", "pt-new"]);
    let mut sink = CaptureSink::default();

    run_case1(&fx.pool, &mut sink, &reader, old_id, COLL_ID, COLL_NAME)
        .await
        .expect("case1");

    // Only one OverwritePayload (for new_id); old_id is below the watermark.
    let overwrite_count = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::OverwritePayload { .. }))
        .count();
    assert_eq!(
        overwrite_count, 1,
        "incremental pass must not touch blobs at or below the watermark"
    );
}

// Membership payload is correct: two branches for one blob.
#[tokio::test]
async fn case1_payload_carries_full_membership_set() {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;
    let blob_id = insert_blob(&fx.pool, "pt-shared").await;
    insert_ref(&fx.pool, BRANCH_A, blob_id, "shared.rs").await;
    insert_ref(&fx.pool, BRANCH_B, blob_id, "shared.rs").await;

    let reader = MockQdrantReader::with_existing(["pt-shared"]);
    let mut sink = CaptureSink::default();

    run_case1(&fx.pool, &mut sink, &reader, 0, COLL_ID, COLL_NAME)
        .await
        .expect("case1");

    let op = sink
        .ops
        .iter()
        .find(|op| matches!(op, QdrantOp::OverwritePayload { .. }))
        .expect("OverwritePayload");

    if let QdrantOp::OverwritePayload { payload, .. } = op {
        let mut branches = payload.branch_id.clone();
        branches.sort();
        assert_eq!(
            branches,
            vec![BRANCH_A.to_string(), BRANCH_B.to_string()],
            "payload must carry full membership set"
        );
        assert_eq!(payload.tenant_id, TENANT);
        assert_eq!(payload.collection_id, COLL_ID);
    }
}

// FULL mode (watermark=0) scans all blobs.
#[tokio::test]
async fn case1_full_mode_scans_all_blobs() {
    let fx = fixture(BRANCH_A).await;
    let id1 = insert_blob(&fx.pool, "pt-f1").await;
    insert_ref(&fx.pool, BRANCH_A, id1, "f1.rs").await;
    let id2 = insert_blob(&fx.pool, "pt-f2").await;
    insert_ref(&fx.pool, BRANCH_A, id2, "f2.rs").await;

    let reader = MockQdrantReader::with_existing(["pt-f1", "pt-f2"]);
    let mut sink = CaptureSink::default();

    run_case1(&fx.pool, &mut sink, &reader, 0, COLL_ID, COLL_NAME)
        .await
        .expect("case1 full");

    let overwrite_count = sink
        .ops
        .iter()
        .filter(|op| matches!(op, QdrantOp::OverwritePayload { .. }))
        .count();
    assert_eq!(overwrite_count, 2, "FULL mode must scan all blobs");
}
