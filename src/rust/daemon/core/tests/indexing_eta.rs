//! Tests for the shared `indexing_progress` module — the rate / ETA
//! math behind the `GetProjectStatus.eta_seconds` field, the MCP search
//! `indexing.eta_seconds`, and the Grafana
//! `memexd_indexing_eta_seconds_by_tenant` gauge.

use sqlx::SqlitePool;
use workspace_qdrant_core::indexing_progress::{
    estimate_eta_seconds, eta_for_gauge, rate_files_per_sec, UNKNOWN_ETA_SENTINEL,
};

/// Build an in-memory pool with the minimum schema we need to exercise
/// `rate_files_per_sec`. We avoid pulling the v37 migrations to keep
/// the test focused — the helper only reads `tracked_files.updated_at`
/// and `watch_folders.tenant_id / watch_id`.
async fn build_pool() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("connect in-memory sqlite");

    sqlx::query(
        r#"CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            path TEXT NOT NULL,
            collection TEXT NOT NULL DEFAULT 'projects'
        )"#,
    )
    .execute(&pool)
    .await
    .expect("create watch_folders");

    sqlx::query(
        r#"CREATE TABLE tracked_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"#,
    )
    .execute(&pool)
    .await
    .expect("create tracked_files");

    pool
}

/// Insert a tracked-file row with `updated_at = now - seconds_ago`.
/// Uses SQLite arithmetic so the comparison stays monotonic with the
/// daemon's `strftime('%s','now')` reads.
async fn seed_tracked_file(pool: &SqlitePool, watch_id: &str, path: &str, seconds_ago: i64) {
    sqlx::query(
        r#"INSERT INTO tracked_files (watch_folder_id, relative_path, updated_at)
           VALUES (?1, ?2, datetime('now', ?3 || ' seconds'))"#,
    )
    .bind(watch_id)
    .bind(path)
    .bind(-seconds_ago)
    .execute(pool)
    .await
    .expect("insert tracked_file");
}

async fn seed_watch_folder(pool: &SqlitePool, watch_id: &str, tenant_id: &str) {
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, tenant_id, path) VALUES (?1, ?2, '/dev/null')",
    )
    .bind(watch_id)
    .bind(tenant_id)
    .execute(pool)
    .await
    .expect("insert watch_folder");
}

#[tokio::test]
async fn rate_returns_none_for_unknown_tenant() {
    let pool = build_pool().await;
    let rate = rate_files_per_sec(&pool, "no_such_tenant").await;
    assert!(rate.is_none(), "unknown tenant must yield no rate signal");
}

#[tokio::test]
async fn rate_returns_none_below_cold_start_threshold() {
    let pool = build_pool().await;
    seed_watch_folder(&pool, "w1", "tenant_warm").await;
    // Only 30 seconds of activity — below MIN_RATE_WINDOW_SECONDS (60).
    for i in 0..5 {
        seed_tracked_file(&pool, "w1", &format!("f{i}.rs"), 30 - i).await;
    }
    let rate = rate_files_per_sec(&pool, "tenant_warm").await;
    assert!(
        rate.is_none(),
        "rate must be None during cold-start window; got {:?}",
        rate
    );
}

#[tokio::test]
async fn rate_reflects_recent_activity_above_threshold() {
    let pool = build_pool().await;
    seed_watch_folder(&pool, "w1", "tenant_active").await;
    // 120 files spread across the last 120 seconds → ~1.0 files/sec.
    for i in 0..120 {
        seed_tracked_file(&pool, "w1", &format!("f{i}.rs"), 120 - i).await;
    }
    let rate = rate_files_per_sec(&pool, "tenant_active")
        .await
        .expect("rate should be Some after 120s of activity");
    // Allow generous slack — SQLite's strftime resolution is whole
    // seconds, and the window calc rounds.
    assert!(
        rate > 0.5 && rate < 2.0,
        "expected ~1 file/sec, got {}",
        rate
    );
}

#[tokio::test]
async fn rate_excludes_old_activity_outside_window() {
    let pool = build_pool().await;
    seed_watch_folder(&pool, "w1", "tenant_old").await;
    // 50 files updated 1 hour ago — well outside the 300s window.
    for i in 0..50 {
        seed_tracked_file(&pool, "w1", &format!("f{i}.rs"), 3600 + i).await;
    }
    let rate = rate_files_per_sec(&pool, "tenant_old").await;
    assert!(
        rate.is_none(),
        "old activity must not produce a rate; got {:?}",
        rate
    );
}

#[tokio::test]
async fn estimate_eta_handles_drained_queue() {
    // No work remaining → ETA is exactly 0 regardless of rate.
    assert_eq!(estimate_eta_seconds(0, 0, Some(1.0)), Some(0));
    assert_eq!(estimate_eta_seconds(0, 0, None), Some(0));
}

#[tokio::test]
async fn estimate_eta_returns_none_without_rate() {
    assert!(estimate_eta_seconds(100, 0, None).is_none());
    assert!(estimate_eta_seconds(100, 0, Some(0.0)).is_none());
    assert!(estimate_eta_seconds(100, 0, Some(-1.0)).is_none());
}

#[tokio::test]
async fn estimate_eta_caps_at_24_hours() {
    // 1 million items at 0.001 files/sec → 1 billion seconds. Must cap.
    let eta = estimate_eta_seconds(1_000_000, 0, Some(0.001))
        .expect("Some when rate > 0 and remaining > 0");
    assert_eq!(eta, 24 * 60 * 60, "expected cap at 24h, got {}", eta);
}

#[tokio::test]
async fn estimate_eta_computes_expected_seconds() {
    // 600 items / 10 per sec = 60s.
    assert_eq!(estimate_eta_seconds(600, 0, Some(10.0)), Some(60));
    // Split between pending/in_progress shouldn't matter.
    assert_eq!(estimate_eta_seconds(300, 300, Some(10.0)), Some(60));
}

#[tokio::test]
async fn eta_for_gauge_uses_sentinel_for_unknown() {
    assert_eq!(eta_for_gauge(None), UNKNOWN_ETA_SENTINEL);
    assert_eq!(eta_for_gauge(Some(42)), 42);
    assert_eq!(eta_for_gauge(Some(0)), 0);
}
