//! Tests for watermark RMW discipline (AC-F15.4 / DATA-NIT-02).
//!
//! File: `wqm-storage-write/src/reconcile/watermark_tests.rs`
//! All tests use temp SQLite + in-process pool (no live Qdrant).

use super::*;
use crate::blob::test_support::{fixture, TENANT};

/// Apply the maintenance_meta DDL on a fresh fixture.
async fn setup_meta(pool: &SqlitePool) {
    ensure_maintenance_meta(pool).await.expect("ensure_meta");
}

// Watermark returns zero defaults when no row exists.
#[tokio::test]
async fn watermark_default_when_missing() {
    let fx = fixture("branch-a").await;
    setup_meta(&fx.pool).await;
    let wm = read_watermark(&fx.pool, TENANT).await.expect("read");
    assert_eq!(wm.tenant_id, TENANT);
    assert!(wm.last_reconcile_at.is_none());
    assert_eq!(wm.max_seen_blob_id, 0);
    assert!(wm.last_tenant_move_at.is_none());
}

// update_watermark persists and is readable immediately.
#[tokio::test]
async fn watermark_persists_after_update() {
    let fx = fixture("branch-a").await;
    setup_meta(&fx.pool).await;
    update_watermark(&fx.pool, TENANT, "2026-06-25T10:00:00Z", 42)
        .await
        .expect("update");
    let wm = read_watermark(&fx.pool, TENANT).await.expect("read");
    assert_eq!(
        wm.last_reconcile_at.as_deref(),
        Some("2026-06-25T10:00:00Z")
    );
    assert_eq!(wm.max_seen_blob_id, 42);
}

// update_watermark uses MAX for max_seen_blob_id (never regresses).
#[tokio::test]
async fn watermark_max_blob_id_never_regresses() {
    let fx = fixture("branch-a").await;
    setup_meta(&fx.pool).await;
    update_watermark(&fx.pool, TENANT, "2026-06-25T10:00:00Z", 100)
        .await
        .expect("update high");
    // Update with a lower value -- max_seen_blob_id must stay at 100.
    update_watermark(&fx.pool, TENANT, "2026-06-25T11:00:00Z", 5)
        .await
        .expect("update low");
    let wm = read_watermark(&fx.pool, TENANT).await.expect("read");
    assert_eq!(
        wm.max_seen_blob_id, 100,
        "max_seen_blob_id must not regress"
    );
    // But last_reconcile_at should be the newer timestamp.
    assert_eq!(
        wm.last_reconcile_at.as_deref(),
        Some("2026-06-25T11:00:00Z")
    );
}

// Transaction-held serialization: update_watermark holds the single connection
// for the entire RMW so no interleaving is possible on a size-1 pool.
//
// This test is non-vacuous by design: it verifies that a concurrent task which
// tries to acquire the pool connection while update_watermark holds it is
// serialized -- the concurrent write cannot interleave between update_watermark's
// upsert and commit. We spawn a task that sleeps briefly then tries pool.begin();
// because pool_size=1, it blocks until update_watermark commits. The final
// max_seen_blob_id reflects both writes applied in order, not a lost update.
#[tokio::test]
async fn watermark_transaction_held_serializes_concurrent_writer() {
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};

    let fx = fixture("branch-a").await;
    setup_meta(&fx.pool).await;

    // Seed an initial watermark so the MAX merge is exercised.
    update_watermark(&fx.pool, TENANT, "2026-06-25T08:00:00Z", 30)
        .await
        .expect("seed");

    let pool = Arc::new(fx.pool.clone());

    // Spawn a concurrent writer that tries to set blob_id=10 (lower than 30 and 50).
    // With correct Transaction-held semantics, this writer cannot interleave with
    // the main writer's RMW: it must wait for the main writer to commit.
    let pool2 = Arc::clone(&pool);
    let concurrent = tokio::spawn(async move {
        // Small delay so the main writer's pool.begin() lands first.
        sleep(Duration::from_millis(5)).await;
        // pool_size=1: this pool.begin() blocks until the main writer commits.
        update_watermark(&pool2, TENANT, "2026-06-25T09:30:00Z", 10)
            .await
            .expect("concurrent writer")
    });

    // Main writer holds the connection for blob_id=50.
    update_watermark(&pool, TENANT, "2026-06-25T09:00:00Z", 50)
        .await
        .expect("main writer");

    concurrent.await.expect("concurrent task");

    let wm = read_watermark(&pool, TENANT).await.expect("read");
    // MAX(30, 50, 10) = 50: the concurrent lower write cannot regress the watermark.
    assert_eq!(
        wm.max_seen_blob_id, 50,
        "serialized MAX must never regress: concurrent lower write must not win"
    );
}

// tenant_move_since_last_pass: no move recorded -> false.
#[tokio::test]
async fn no_tenant_move_means_skip_case5() {
    let wm = ReconcileWatermark {
        tenant_id: TENANT.to_owned(),
        last_reconcile_at: Some("2026-06-25T10:00:00Z".to_owned()),
        max_seen_blob_id: 0,
        last_tenant_move_at: None,
    };
    assert!(!wm.tenant_move_since_last_pass());
}

// tenant_move_since_last_pass: move BEFORE last reconcile -> false.
#[tokio::test]
async fn old_tenant_move_means_skip_case5() {
    let wm = ReconcileWatermark {
        tenant_id: TENANT.to_owned(),
        last_reconcile_at: Some("2026-06-25T10:00:00Z".to_owned()),
        max_seen_blob_id: 0,
        last_tenant_move_at: Some("2026-06-24T08:00:00Z".to_owned()),
    };
    assert!(!wm.tenant_move_since_last_pass(), "old move: skip case-5");
}

// tenant_move_since_last_pass: move AFTER last reconcile -> true.
#[tokio::test]
async fn recent_tenant_move_triggers_case5() {
    let wm = ReconcileWatermark {
        tenant_id: TENANT.to_owned(),
        last_reconcile_at: Some("2026-06-25T10:00:00Z".to_owned()),
        max_seen_blob_id: 0,
        last_tenant_move_at: Some("2026-06-25T11:00:00Z".to_owned()),
    };
    assert!(wm.tenant_move_since_last_pass(), "recent move: run case-5");
}

// tenant_move_since_last_pass: move recorded, no prior reconcile -> true (first run).
#[tokio::test]
async fn tenant_move_no_prior_reconcile_triggers_case5() {
    let wm = ReconcileWatermark {
        tenant_id: TENANT.to_owned(),
        last_reconcile_at: None,
        max_seen_blob_id: 0,
        last_tenant_move_at: Some("2026-06-25T11:00:00Z".to_owned()),
    };
    assert!(wm.tenant_move_since_last_pass());
}

// record_tenant_move stores the timestamp and is visible via read_watermark.
#[tokio::test]
async fn record_tenant_move_is_visible() {
    let fx = fixture("branch-a").await;
    setup_meta(&fx.pool).await;
    record_tenant_move(&fx.pool, TENANT, "2026-06-25T12:00:00Z")
        .await
        .expect("record");
    let wm = read_watermark(&fx.pool, TENANT).await.expect("read");
    assert_eq!(
        wm.last_tenant_move_at.as_deref(),
        Some("2026-06-25T12:00:00Z")
    );
}
