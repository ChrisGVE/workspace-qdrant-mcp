//! Shared per-tenant indexing-progress helpers.
//!
//! The gRPC `GetProjectStatus` handler and the Prometheus exporter both
//! need the same rate / ETA math. This module owns:
//!   - the rolling-window rate query against `tracked_files.updated_at`
//!   - the ETA computation with cold-start guard and cap
//!
//! Keeping it here (rather than in the gRPC crate) lets the
//! `start_queue_depth_exporter` background task call it without a circular
//! crate dependency.

use sqlx::SqlitePool;

/// Rolling window for ETA rate computation. Five minutes balances
/// burst noise (LSP startup, rename storms) against responsiveness.
pub const RATE_WINDOW_SECONDS: i64 = 300;

/// Minimum window of activity required before publishing an ETA.
/// Below this we return `None` so callers show "warming up" instead
/// of a wildly skewed estimate.
pub const MIN_RATE_WINDOW_SECONDS: i64 = 60;

/// Hard cap on the ETA we publish — anything beyond 24h is more
/// confusing than useful.
pub const ETA_CAP_SECONDS: i64 = 24 * 60 * 60;

/// Compute the recent indexing rate (files / sec) for a tenant from
/// `tracked_files.updated_at`. Returns `None` when:
///   - the SQL query fails (callers degrade gracefully — ETA is best-effort)
///   - the observed window is shorter than [`MIN_RATE_WINDOW_SECONDS`]
///   - no rows updated in the last [`RATE_WINDOW_SECONDS`]
///
/// SQL relies on `strftime('%s', …)` to keep the arithmetic monotonic
/// with the daemon's wall clock and avoid parsing ISO timestamps in Rust.
pub async fn rate_files_per_sec(pool: &SqlitePool, tenant_id: &str) -> Option<f64> {
    let row = sqlx::query(
        r#"
        SELECT
            CAST(SUM(CASE
                WHEN strftime('%s','now') - strftime('%s', tf.updated_at) <= ?2
                THEN 1 ELSE 0 END) AS INTEGER) AS recent_count,
            CAST(strftime('%s','now') AS INTEGER)
                - CAST(strftime('%s', MIN(tf.updated_at)) AS INTEGER) AS window_seconds
        FROM tracked_files tf
        JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
        WHERE wf.tenant_id = ?1
          AND strftime('%s','now') - strftime('%s', tf.updated_at) <= ?2
        "#,
    )
    .bind(tenant_id)
    .bind(RATE_WINDOW_SECONDS)
    .fetch_one(pool)
    .await
    .ok()?;

    use sqlx::Row;
    let recent: i64 = row
        .try_get::<Option<i64>, _>("recent_count")
        .ok()?
        .unwrap_or(0);
    let window: i64 = row
        .try_get::<Option<i64>, _>("window_seconds")
        .ok()?
        .unwrap_or(0);

    if window < MIN_RATE_WINDOW_SECONDS || recent <= 0 {
        return None;
    }
    let effective_window = window.min(RATE_WINDOW_SECONDS).max(1) as f64;
    Some((recent as f64) / effective_window)
}

/// Estimate seconds remaining to drain `(pending + in_progress)` items
/// at the given rate. Returns `None` when the rate is unknown or zero
/// with work remaining. Returns `Some(0)` when nothing is in flight.
/// Capped at [`ETA_CAP_SECONDS`].
pub fn estimate_eta_seconds(pending: i64, in_progress: i64, rate: Option<f64>) -> Option<i64> {
    let remaining = pending + in_progress;
    if remaining <= 0 {
        return Some(0);
    }
    let rate = rate?;
    if rate <= 0.0 {
        return None;
    }
    let raw = (remaining as f64) / rate;
    let capped = raw.min(ETA_CAP_SECONDS as f64).max(0.0) as i64;
    Some(capped)
}

/// Sentinel value used in the Prometheus gauge to indicate "unknown".
/// Prometheus has no native null; PromQL can filter via `>= 0` to
/// hide rows where the daemon can't estimate yet.
pub const UNKNOWN_ETA_SENTINEL: i64 = -1;

/// Encode an `Option<i64>` ETA into the gauge sentinel form.
pub fn eta_for_gauge(eta_seconds: Option<i64>) -> i64 {
    eta_seconds.unwrap_or(UNKNOWN_ETA_SENTINEL)
}
