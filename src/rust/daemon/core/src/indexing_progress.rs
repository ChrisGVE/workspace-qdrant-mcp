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

    rate_from_window(recent, window)
}

/// Compute the **daemon-wide** indexing rate (files / sec) across every tenant,
/// from `tracked_files.updated_at`. Same windowing and cold-start guard as
/// [`rate_files_per_sec`], but without the per-tenant filter.
///
/// This is the fallback the ETA uses when a single tenant's own rate is
/// unavailable. The unified queue processes roughly one tenant at a time, so a
/// tenant that is *waiting its turn* writes no `tracked_files` rows and its
/// per-tenant window goes cold (`recent = 0`) — yielding a perpetual
/// "warming up". The global window stays warm as long as the daemon is making
/// progress *anywhere*, so callers can surface a realistic "time to process N
/// items at the daemon's current throughput" instead of nothing.
pub async fn global_rate_files_per_sec(pool: &SqlitePool) -> Option<f64> {
    let row = sqlx::query(
        r#"
        SELECT
            CAST(SUM(CASE
                WHEN strftime('%s','now') - strftime('%s', tf.updated_at) <= ?1
                THEN 1 ELSE 0 END) AS INTEGER) AS recent_count,
            CAST(strftime('%s','now') AS INTEGER)
                - CAST(strftime('%s', MIN(tf.updated_at)) AS INTEGER) AS window_seconds
        FROM tracked_files tf
        WHERE strftime('%s','now') - strftime('%s', tf.updated_at) <= ?1
        "#,
    )
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

    rate_from_window(recent, window)
}

/// Turn a `(recent_count, window_seconds)` pair into a files/sec rate, applying
/// the cold-start guard: `None` until at least [`MIN_RATE_WINDOW_SECONDS`] of
/// activity with a positive count, so callers show "warming up" rather than a
/// wildly skewed estimate. The window is clamped to [`RATE_WINDOW_SECONDS`].
fn rate_from_window(recent: i64, window: i64) -> Option<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_from_window_warming_up_below_min_window() {
        // Active but only 30s of history -> still warming up.
        assert_eq!(rate_from_window(100, 30), None);
    }

    #[test]
    fn rate_from_window_warming_up_when_no_recent_rows() {
        // A full window but nothing processed recently -> no rate.
        assert_eq!(rate_from_window(0, 300), None);
    }

    #[test]
    fn rate_from_window_computes_rate_once_warm() {
        // 150 files over a 300s window -> 0.5 files/sec.
        assert_eq!(rate_from_window(150, 300), Some(0.5));
    }

    #[test]
    fn rate_from_window_clamps_window_to_max() {
        // window longer than RATE_WINDOW_SECONDS is clamped to it.
        assert_eq!(rate_from_window(300, 600), Some(1.0));
    }

    #[test]
    fn eta_is_zero_when_nothing_in_flight() {
        assert_eq!(estimate_eta_seconds(0, 0, None), Some(0));
        assert_eq!(estimate_eta_seconds(0, 0, Some(5.0)), Some(0));
    }

    #[test]
    fn eta_warming_up_when_rate_unknown_with_work() {
        assert_eq!(estimate_eta_seconds(100, 0, None), None);
    }

    #[test]
    fn eta_computes_from_rate() {
        // 100 remaining at 5 files/sec -> 20s.
        assert_eq!(estimate_eta_seconds(80, 20, Some(5.0)), Some(20));
    }

    #[test]
    fn eta_capped_at_24h() {
        // 1 item at a glacial rate would exceed the cap -> clamped.
        assert_eq!(estimate_eta_seconds(1, 0, Some(0.0000001)), Some(ETA_CAP_SECONDS));
    }
}
