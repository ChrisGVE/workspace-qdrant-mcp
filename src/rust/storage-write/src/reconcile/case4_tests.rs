//! Tests for reconcile case 4 (AC-F15.1 / AC-F15.4 -- FTS branch-membership drift).

use super::*;
use crate::blob::test_support::{add_branch, fixture, TENANT};
use crate::blob::vector_codec::{encode_dense, encode_sparse};
use std::collections::HashMap;

const BRANCH_A: &str = "branch-a";
const BRANCH_B: &str = "branch-b";

/// Insert a blob and return its blob_id.
async fn insert_blob(pool: &sqlx::SqlitePool, point_id: &str) -> i64 {
    let dense = encode_dense(&[]);
    let sparse = encode_sparse(&HashMap::new());
    sqlx::query(
        "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
         raw_text, dense_vec, sparse_vec, created_at) \
         VALUES (?, 'h', ?, ?, 't', ?, ?, '2026-01-01')",
    )
    .bind(format!("ck-{point_id}"))
    .bind(point_id)
    .bind(TENANT)
    .bind(dense)
    .bind(sparse)
    .execute(pool)
    .await
    .expect("blob insert")
    .last_insert_rowid()
}

/// Insert a `blob_refs` row WITHOUT inserting `fts_branch_membership`
/// (simulating the crash after blob_refs write but before FTS write).
async fn insert_ref_no_fts(pool: &sqlx::SqlitePool, branch_id: &str, blob_id: i64, path: &str) {
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
    sqlx::query("INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES(?,?,0,?)")
        .bind(branch_id)
        .bind(file_id)
        .bind(blob_id)
        .execute(pool)
        .await
        .expect("ref insert");
    // Deliberately do NOT insert into fts_branch_membership.
}

/// Count rows in `fts_branch_membership` for a given blob_id.
async fn fts_count(pool: &sqlx::SqlitePool, blob_id: i64) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM fts_branch_membership WHERE blob_id=?")
        .bind(blob_id)
        .fetch_one(pool)
        .await
        .expect("fts count")
}

// AC-F15.1 case 4: drifted FTS row is inserted by the pass.
#[tokio::test]
async fn case4_missing_fts_row_is_repaired() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-c4").await;
    insert_ref_no_fts(&fx.pool, BRANCH_A, blob_id, "f.rs").await;

    // Before the pass: no FTS membership row.
    assert_eq!(fts_count(&fx.pool, blob_id).await, 0);

    run_case4(&fx.pool, 0).await.expect("case4");

    // After the pass: FTS membership row exists.
    assert_eq!(
        fts_count(&fx.pool, blob_id).await,
        1,
        "FTS row must be inserted"
    );
}

// AC-F15.4: existing FTS row is not duplicated (ON CONFLICT DO NOTHING).
#[tokio::test]
async fn case4_existing_fts_row_is_not_duplicated() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-c4-dup").await;
    insert_ref_no_fts(&fx.pool, BRANCH_A, blob_id, "g.rs").await;

    // Pre-insert the FTS row (already correct state).
    sqlx::query("INSERT INTO fts_branch_membership(blob_id, branch_id) VALUES(?, ?)")
        .bind(blob_id)
        .bind(BRANCH_A)
        .execute(&fx.pool)
        .await
        .expect("pre-insert fts");

    run_case4(&fx.pool, 0).await.expect("case4");

    // Count must still be 1 (not 2).
    assert_eq!(
        fts_count(&fx.pool, blob_id).await,
        1,
        "ON CONFLICT DO NOTHING must not create duplicate"
    );
}

// AC-F15.4: incremental watermark scopes scan to blobs above watermark.
#[tokio::test]
async fn case4_incremental_watermark_scopes_scan() {
    let fx = fixture(BRANCH_A).await;

    let old_id = insert_blob(&fx.pool, "pt-old").await;
    insert_ref_no_fts(&fx.pool, BRANCH_A, old_id, "old.rs").await;

    let new_id = insert_blob(&fx.pool, "pt-new").await;
    insert_ref_no_fts(&fx.pool, BRANCH_A, new_id, "new.rs").await;

    // Watermark = old_id: pass sees only new_id.
    run_case4(&fx.pool, old_id)
        .await
        .expect("case4 incremental");

    assert_eq!(
        fts_count(&fx.pool, old_id).await,
        0,
        "old blob at watermark must not be touched"
    );
    assert_eq!(
        fts_count(&fx.pool, new_id).await,
        1,
        "new blob above watermark must be repaired"
    );
}

// FULL mode (watermark=0) repairs all blobs.
#[tokio::test]
async fn case4_full_mode_repairs_all_blobs() {
    let fx = fixture(BRANCH_A).await;
    add_branch(&fx.pool, BRANCH_B).await;

    let id1 = insert_blob(&fx.pool, "pt-all1").await;
    insert_ref_no_fts(&fx.pool, BRANCH_A, id1, "p1.rs").await;
    let id2 = insert_blob(&fx.pool, "pt-all2").await;
    insert_ref_no_fts(&fx.pool, BRANCH_B, id2, "p2.rs").await;

    run_case4(&fx.pool, 0).await.expect("case4 full");

    assert_eq!(fts_count(&fx.pool, id1).await, 1);
    assert_eq!(fts_count(&fx.pool, id2).await, 1);
}

// max_blob_id returned is correct.
#[tokio::test]
async fn case4_returns_correct_max_blob_id() {
    let fx = fixture(BRANCH_A).await;

    let id1 = insert_blob(&fx.pool, "pt-mx1").await;
    insert_ref_no_fts(&fx.pool, BRANCH_A, id1, "mx1.rs").await;
    let id2 = insert_blob(&fx.pool, "pt-mx2").await;
    insert_ref_no_fts(&fx.pool, BRANCH_A, id2, "mx2.rs").await;

    let max = run_case4(&fx.pool, 0).await.expect("case4");
    assert_eq!(
        max, id2,
        "max blob_id returned must be the highest blob scanned"
    );
}
