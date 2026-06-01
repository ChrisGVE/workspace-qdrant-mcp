//! State-DB maintenance metrics (PRD D5 DATA-N1 / PERF-N1).
//!
//! Exposes the timestamp of the last state-DB `VACUUM` (and integrity check) as
//! Prometheus gauges sourced from the single-row `db_maintenance` table.
//!
//! The gauge is **absent until the first VACUUM runs** — a NULL `last_vacuum_at`
//! leaves the series unregistered so a Grafana "Last Vacuum Age" panel renders
//! "No data" → "never vacuumed" instead of a misleading `0`. To achieve that the
//! gauge is registered into [`METRICS`] lazily, on first `set`, rather than at
//! construction.

use std::sync::atomic::{AtomicBool, Ordering};

use once_cell::sync::Lazy;
use prometheus::Gauge;
use sqlx::{Row, SqlitePool};
use tracing::debug;
use wqm_common::timestamps::now_utc;

use crate::monitoring::METRICS;

/// Lazily-registered state-DB maintenance gauges.
struct StateDbMetrics {
    last_vacuum: Gauge,
    last_vacuum_registered: AtomicBool,
    last_integrity_check: Gauge,
    last_integrity_check_registered: AtomicBool,
}

impl StateDbMetrics {
    fn new() -> Self {
        Self {
            last_vacuum: Gauge::new(
                "wqm_memexd_state_db_last_vacuum_timestamp_seconds",
                "Unix timestamp (seconds) of the last state-DB VACUUM",
            )
            .expect("metric can be created"),
            last_vacuum_registered: AtomicBool::new(false),
            last_integrity_check: Gauge::new(
                "wqm_memexd_state_db_last_integrity_check_timestamp_seconds",
                "Unix timestamp (seconds) of the last state-DB integrity check",
            )
            .expect("metric can be created"),
            last_integrity_check_registered: AtomicBool::new(false),
        }
    }

    fn set_vacuum(&self, epoch_seconds: f64) {
        if !self.last_vacuum_registered.swap(true, Ordering::Relaxed) {
            let _ = METRICS
                .registry
                .register(Box::new(self.last_vacuum.clone()));
        }
        self.last_vacuum.set(epoch_seconds);
    }

    fn set_integrity_check(&self, epoch_seconds: f64) {
        if !self
            .last_integrity_check_registered
            .swap(true, Ordering::Relaxed)
        {
            let _ = METRICS
                .registry
                .register(Box::new(self.last_integrity_check.clone()));
        }
        self.last_integrity_check.set(epoch_seconds);
    }
}

static STATE_DB_METRICS: Lazy<StateDbMetrics> = Lazy::new(StateDbMetrics::new);

/// Parse a stored ISO-8601 maintenance timestamp into Unix epoch seconds.
/// Returns `None` for a missing/NULL or unparseable value (→ metric stays
/// absent: "never run").
fn epoch_from_iso(value: Option<String>) -> Option<f64> {
    let s = value?;
    chrono::DateTime::parse_from_rfc3339(s.trim())
        .ok()
        .map(|dt| dt.timestamp() as f64)
}

/// Snapshot the state-DB maintenance gauges from the `db_maintenance` row.
/// No-op (and no series) when telemetry is off, the table is absent, or no
/// maintenance has ever run.
pub async fn snapshot_state_db_metrics(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    if !METRICS.is_enabled() {
        return Ok(());
    }
    let row = sqlx::query(
        "SELECT last_vacuum_at, last_integrity_check_at FROM db_maintenance WHERE id = 1",
    )
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else {
        return Ok(());
    };

    if let Some(epoch) = epoch_from_iso(row.get::<Option<String>, _>("last_vacuum_at")) {
        STATE_DB_METRICS.set_vacuum(epoch);
    }
    if let Some(epoch) = epoch_from_iso(row.get::<Option<String>, _>("last_integrity_check_at")) {
        STATE_DB_METRICS.set_integrity_check(epoch);
    }
    Ok(())
}

/// Record that a VACUUM just completed by stamping `last_vacuum_at` (UTC, ISO-8601).
pub async fn record_vacuum(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE db_maintenance SET last_vacuum_at = ?1 WHERE id = 1")
        .bind(now_utc())
        .execute(pool)
        .await?;
    Ok(())
}

/// Record that an integrity check just completed by stamping
/// `last_integrity_check_at` (UTC, ISO-8601).
pub async fn record_integrity_check(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE db_maintenance SET last_integrity_check_at = ?1 WHERE id = 1")
        .bind(now_utc())
        .execute(pool)
        .await?;
    Ok(())
}

/// Run `VACUUM` on the state DB and stamp `last_vacuum_at` (DATA-N1). VACUUM
/// cannot run inside a transaction, so it is issued directly on the pool.
pub async fn vacuum_state_db(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    debug!("Running state-DB VACUUM");
    sqlx::query("VACUUM").execute(pool).await?;
    record_vacuum(pool).await
}

/// Run `PRAGMA integrity_check` and, on success, stamp `last_integrity_check_at`
/// (PERF-N1 weekly cadence). Returns the integrity-check result string
/// (`"ok"` when healthy).
pub async fn integrity_check_state_db(pool: &SqlitePool) -> Result<String, sqlx::Error> {
    let result: String = sqlx::query_scalar("PRAGMA integrity_check")
        .fetch_one(pool)
        .await?;
    record_integrity_check(pool).await?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn migrated_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        pool
    }

    #[test]
    fn epoch_from_iso_handles_null_and_values() {
        assert_eq!(epoch_from_iso(None), None);
        assert_eq!(epoch_from_iso(Some("not-a-date".to_string())), None);
        let e = epoch_from_iso(Some("2026-06-01T00:00:00Z".to_string())).unwrap();
        // 2026-06-01T00:00:00Z = 1780272000 unix seconds.
        assert_eq!(e, 1_780_272_000.0);
    }

    #[tokio::test]
    async fn fresh_db_has_null_vacuum_then_record_sets_it() {
        let pool = migrated_pool().await;

        // Fresh: NULL → snapshot would not set the gauge.
        let vac: Option<String> =
            sqlx::query_scalar("SELECT last_vacuum_at FROM db_maintenance WHERE id = 1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(vac.is_none(), "fresh DB has no vacuum timestamp");

        // Record a vacuum → row now has a parseable ISO timestamp.
        record_vacuum(&pool).await.unwrap();
        let vac: Option<String> =
            sqlx::query_scalar("SELECT last_vacuum_at FROM db_maintenance WHERE id = 1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(
            epoch_from_iso(vac).is_some(),
            "recorded vacuum timestamp must parse to an epoch"
        );
    }

    #[tokio::test]
    async fn snapshot_sets_gauge_after_vacuum() {
        let pool = migrated_pool().await;
        record_vacuum(&pool).await.unwrap();
        snapshot_state_db_metrics(&pool).await.unwrap();
        assert!(
            STATE_DB_METRICS.last_vacuum.get() > 0.0,
            "gauge set to a positive unix timestamp after vacuum+snapshot"
        );
    }

    #[tokio::test]
    async fn vacuum_and_integrity_helpers_run() {
        let pool = migrated_pool().await;
        vacuum_state_db(&pool).await.unwrap();
        let res = integrity_check_state_db(&pool).await.unwrap();
        assert_eq!(res, "ok");

        let (vac, integ): (Option<String>, Option<String>) = sqlx::query_as(
            "SELECT last_vacuum_at, last_integrity_check_at FROM db_maintenance WHERE id = 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(epoch_from_iso(vac).is_some());
        assert!(epoch_from_iso(integ).is_some());
    }
}
