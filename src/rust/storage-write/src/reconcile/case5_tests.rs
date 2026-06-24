//! Tests for reconcile case 5 (AC-F15.6 -- cross-DB tenant-mismatch heal).
//!
//! Three required tests per AC-F15.6:
//!   (i)  stale-payload fixture -> reconcile re-PUTs to correct tenant, does NOT delete.
//!   (ii) transient-window fixture (both stores hold row, payload=destination) -> NO-OP.
//!   (iii) empty migration journal -> no full-collection scan (zero reader calls).

use super::*;
use crate::blob::ladder::{CaptureSink, QdrantOp};
use crate::blob::test_support::{fixture, TENANT};
use crate::blob::vector_codec::{encode_dense, encode_sparse};
use crate::reconcile::seams::MockQdrantReader;
use crate::reconcile::watermark::ReconcileWatermark;
use std::collections::HashMap;

const BRANCH_A: &str = "branch-a";
const COLL_ID: &str = "projects";

/// Watermark that signals a recent tenant-move (case-5 MUST run).
fn watermark_with_move() -> ReconcileWatermark {
    ReconcileWatermark {
        tenant_id: TENANT.to_owned(),
        last_reconcile_at: Some("2026-06-24T00:00:00Z".to_owned()),
        max_seen_blob_id: 0,
        last_tenant_move_at: Some("2026-06-25T00:00:00Z".to_owned()),
    }
}

/// Watermark with no tenant-move (case-5 MUST be skipped).
fn watermark_empty_journal() -> ReconcileWatermark {
    ReconcileWatermark {
        tenant_id: TENANT.to_owned(),
        last_reconcile_at: Some("2026-06-25T00:00:00Z".to_owned()),
        max_seen_blob_id: 0,
        last_tenant_move_at: None,
    }
}

/// Insert a minimal blob, return (blob_id, point_id).
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

/// Insert file + blob_ref for (branch, blob_id).
async fn insert_ref(pool: &sqlx::SqlitePool, branch_id: &str, blob_id: i64, path: &str) {
    sqlx::query(
        "INSERT INTO files(branch_id, relative_path, collection, created_at, updated_at) \
         VALUES (?, ?, 'projects', '2026-01-01', '2026-01-01')",
    )
    .bind(branch_id)
    .bind(path)
    .execute(pool)
    .await
    .expect("file");
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
        .expect("ref");
}

// ---------------------------------------------------------------------------
// AC-F15.6 test (i): stale-payload fixture
// ---------------------------------------------------------------------------
// Store T holds blob_refs row. Qdrant payload says tenant=T' (wrong/stale).
// No store matches T'. Reconcile must re-PUT the point to T and NOT delete it.
#[tokio::test]
async fn case5_stale_payload_is_reput_not_deleted() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-stale").await;
    insert_ref(&fx.pool, BRANCH_A, blob_id, "stale.rs").await;

    // Reader: point exists, but payload tenant is "tenant-wrong" (stale).
    let reader = MockQdrantReader::all_present(["pt-stale"], [("pt-stale", "tenant-wrong")]);

    // Candidate: this store owns the blob (has blob_refs row).
    let candidates = vec![TenantMismatchCandidate {
        point_id: "pt-stale".to_owned(),
        store_tenant_id: TENANT.to_owned(),
        blob_id,
        collection_id: COLL_ID.to_owned(),
    }];

    let wm = watermark_with_move();
    let mut sink = CaptureSink::default();
    let healed = run_case5(&fx.pool, &mut sink, &reader, &wm, &candidates, COLL_ID)
        .await
        .expect("case5 stale");

    assert_eq!(healed, 1, "stale payload must be healed");

    // OverwritePayload enqueued (not Delete).
    let has_overwrite = sink.ops.iter().any(|op| {
        matches!(op, QdrantOp::OverwritePayload { point_id, payload }
            if point_id == "pt-stale" && payload.tenant_id == TENANT)
    });
    assert!(
        has_overwrite,
        "must enqueue OverwritePayload with correct tenant"
    );

    let has_delete = sink
        .ops
        .iter()
        .any(|op| matches!(op, QdrantOp::Delete { .. }));
    assert!(!has_delete, "must NOT enqueue Delete (additive-first)");
}

// ---------------------------------------------------------------------------
// AC-F15.6 test (ii): transient-window fixture
// ---------------------------------------------------------------------------
// BOTH stores hold a blob_refs row (copy-then-delete crash-C2 window).
// Payload already says tenant=destination. Reconcile must be a NO-OP.
// We model this as: reader returns payload tenant == store_tenant (destination).
#[tokio::test]
async fn case5_transient_window_is_noop_no_oscillation() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-transient").await;
    insert_ref(&fx.pool, BRANCH_A, blob_id, "transient.rs").await;

    // Reader: payload tenant == store's tenant (the PUT already landed for destination).
    let reader = MockQdrantReader::all_present(
        ["pt-transient"],
        [("pt-transient", TENANT)], // payload already correct
    );

    let candidates = vec![TenantMismatchCandidate {
        point_id: "pt-transient".to_owned(),
        store_tenant_id: TENANT.to_owned(),
        blob_id,
        collection_id: COLL_ID.to_owned(),
    }];

    let wm = watermark_with_move();
    let mut sink = CaptureSink::default();
    let healed = run_case5(&fx.pool, &mut sink, &reader, &wm, &candidates, COLL_ID)
        .await
        .expect("case5 transient");

    assert_eq!(healed, 0, "transient window: no heal needed");
    assert!(
        sink.ops.is_empty(),
        "transient window: no ops must be enqueued (no oscillation)"
    );
}

// ---------------------------------------------------------------------------
// AC-F15.6 test (iii): empty migration journal -> no full-collection scan
// ---------------------------------------------------------------------------
// When watermark.tenant_move_since_last_pass() is false, case-5 must return
// immediately without calling reader.payload_tenant_id at all.
// We verify this by using a candidate list; if case-5 called the reader, it
// would see a stale payload and enqueue an OverwritePayload. Since we assert
// zero ops, the reader was never called.
#[tokio::test]
async fn case5_empty_journal_skips_entirely_no_scan() {
    let fx = fixture(BRANCH_A).await;
    let blob_id = insert_blob(&fx.pool, "pt-nojrn").await;
    insert_ref(&fx.pool, BRANCH_A, blob_id, "nojrn.rs").await;

    // Reader would return stale payload IF called.
    let reader = MockQdrantReader::all_present(["pt-nojrn"], [("pt-nojrn", "tenant-wrong")]);

    let candidates = vec![TenantMismatchCandidate {
        point_id: "pt-nojrn".to_owned(),
        store_tenant_id: TENANT.to_owned(),
        blob_id,
        collection_id: COLL_ID.to_owned(),
    }];

    // Watermark with NO tenant-move (empty journal).
    let wm = watermark_empty_journal();
    let mut sink = CaptureSink::default();
    let healed = run_case5(&fx.pool, &mut sink, &reader, &wm, &candidates, COLL_ID)
        .await
        .expect("case5 empty journal");

    assert_eq!(healed, 0, "empty journal: case-5 must be entirely skipped");
    assert!(
        sink.ops.is_empty(),
        "empty journal: zero ops (no scan performed)"
    );
}
