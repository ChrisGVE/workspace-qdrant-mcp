//! Phase-1 SQLite state-DB metrics (PRD D5).
//!
//! Snapshots the live state of the daemon's SQLite database as Prometheus
//! gauges, sourced from PRAGMAs and bounded per-table row counts. Emitted once
//! per metrics interval from a single read transaction on the state pool.
//!
//! Gauges (all under the `wqm_memexd_` namespace, A4):
//! - `wqm_memexd_state_db_schema_version`        — current migration version
//! - `wqm_memexd_state_db_size_bytes`            — `page_count * page_size`
//! - `wqm_memexd_state_db_wal_size_bytes`        — approx WAL file size
//! - `wqm_memexd_state_db_wal_frames_pending`    — un-checkpointed WAL frames
//! - `wqm_memexd_state_db_free_pages`            — `PRAGMA freelist_count`
//! - `wqm_memexd_state_db_integrity_ok`          — `PRAGMA quick_check` (0/1),
//!                                                 run at most weekly (PERF-N1)
//! - `wqm_memexd_state_db_table_rows{table}`     — row counts for the 14
//!                                                 canonical tables (bounded)
//!
//! The eighth Phase-1 gauge —
//! `wqm_memexd_state_db_last_vacuum_timestamp_seconds` — is owned by
//! [`super::state_db_metrics`] (it must stay absent until the first VACUUM) and
//! is therefore not redefined here.

use once_cell::sync::Lazy;
use prometheus::{IntGauge, IntGaugeVec, Opts};
use sqlx::{Row, SqlitePool};
use tracing::{debug, warn};

use crate::monitoring::{state_db_metrics, METRICS};

/// Canonical state-DB tables whose row counts are exported. Compile-time
/// constant → `table` label cardinality is bounded to exactly these 14 values.
pub const CANONICAL_TABLES: [&str; 14] = [
    "unified_queue",
    "dead_letter_queue",
    "tracked_files",
    "indexed_content",
    "qdrant_chunks",
    "graph_nodes",
    "graph_edges",
    "metrics_history",
    "code_lines",
    "watch_folders",
    "keywords",
    "search_events",
    "processing_timings",
    "resolution_events",
];

/// Minimum spacing between `PRAGMA quick_check` runs (PERF-N1: weekly cadence).
const INTEGRITY_CHECK_INTERVAL_SECS: i64 = 7 * 24 * 60 * 60;

/// WAL frame header size in bytes (prepended to each page in the WAL file).
const WAL_FRAME_HEADER_BYTES: i64 = 24;
/// WAL file header size in bytes.
const WAL_FILE_HEADER_BYTES: i64 = 32;

struct SqliteStateMetrics {
    schema_version: IntGauge,
    db_size_bytes: IntGauge,
    wal_size_bytes: IntGauge,
    wal_frames_pending: IntGauge,
    free_pages: IntGauge,
    integrity_ok: IntGauge,
    table_rows: IntGaugeVec,
}

impl SqliteStateMetrics {
    fn new() -> Self {
        let schema_version = IntGauge::new(
            "wqm_memexd_state_db_schema_version",
            "Current state-DB schema migration version",
        )
        .expect("metric can be created");
        let db_size_bytes = IntGauge::new(
            "wqm_memexd_state_db_size_bytes",
            "State-DB logical size in bytes (page_count * page_size)",
        )
        .expect("metric can be created");
        let wal_size_bytes = IntGauge::new(
            "wqm_memexd_state_db_wal_size_bytes",
            "Approximate state-DB WAL file size in bytes",
        )
        .expect("metric can be created");
        let wal_frames_pending = IntGauge::new(
            "wqm_memexd_state_db_wal_frames_pending",
            "WAL frames not yet checkpointed into the main state-DB file",
        )
        .expect("metric can be created");
        let free_pages = IntGauge::new(
            "wqm_memexd_state_db_free_pages",
            "Free pages in the state-DB freelist (PRAGMA freelist_count)",
        )
        .expect("metric can be created");
        let integrity_ok = IntGauge::new(
            "wqm_memexd_state_db_integrity_ok",
            "State-DB integrity: 1 = PRAGMA quick_check ok, 0 = corruption detected",
        )
        .expect("metric can be created");
        let table_rows = IntGaugeVec::new(
            Opts::new(
                "wqm_memexd_state_db_table_rows",
                "Row count per canonical state-DB table",
            ),
            &["table"],
        )
        .expect("metric can be created");

        let r = &METRICS.registry;
        let _ = r.register(Box::new(schema_version.clone()));
        let _ = r.register(Box::new(db_size_bytes.clone()));
        let _ = r.register(Box::new(wal_size_bytes.clone()));
        let _ = r.register(Box::new(wal_frames_pending.clone()));
        let _ = r.register(Box::new(free_pages.clone()));
        let _ = r.register(Box::new(integrity_ok.clone()));
        let _ = r.register(Box::new(table_rows.clone()));

        Self {
            schema_version,
            db_size_bytes,
            wal_size_bytes,
            wal_frames_pending,
            free_pages,
            integrity_ok,
            table_rows,
        }
    }
}

static SQLITE_STATE: Lazy<SqliteStateMetrics> = Lazy::new(SqliteStateMetrics::new);

/// Whether a `PRAGMA quick_check` is due, given the last-check timestamp and
/// the current instant. A missing/unparseable timestamp is always due.
fn is_integrity_due(last_check_at: Option<&str>, now: chrono::DateTime<chrono::Utc>) -> bool {
    match last_check_at {
        None => true,
        Some(s) => match chrono::DateTime::parse_from_rfc3339(s.trim()) {
            Ok(dt) => {
                (now - dt.with_timezone(&chrono::Utc)).num_seconds()
                    >= INTEGRITY_CHECK_INTERVAL_SECS
            }
            Err(_) => true,
        },
    }
}

/// Count rows in a canonical (compile-time-constant) table. The name is never
/// user-supplied, so interpolation is safe.
async fn count_table_rows(pool: &SqlitePool, table: &str) -> Result<i64, sqlx::Error> {
    let query = format!("SELECT COUNT(*) FROM {table}");
    sqlx::query_scalar(&query).fetch_one(pool).await
}

/// Run `PRAGMA quick_check` if it has not run in the last 7 days, updating the
/// `integrity_ok` gauge and stamping `last_integrity_check_at`. When not yet
/// due, the gauge retains its previous value (no check is performed).
async fn maybe_integrity_check(
    pool: &SqlitePool,
    m: &SqliteStateMetrics,
) -> Result<(), sqlx::Error> {
    let last: Option<String> = sqlx::query_scalar::<_, Option<String>>(
        "SELECT last_integrity_check_at FROM db_maintenance WHERE id = 1",
    )
    .fetch_optional(pool)
    .await?
    .flatten();

    if !is_integrity_due(last.as_deref(), chrono::Utc::now()) {
        return Ok(());
    }

    let result: String = sqlx::query_scalar("PRAGMA quick_check")
        .fetch_one(pool)
        .await?;
    m.integrity_ok.set(if result == "ok" { 1 } else { 0 });
    state_db_metrics::record_integrity_check(pool).await?;
    Ok(())
}

/// Snapshot the Phase-1 SQLite state gauges from a single bounded read
/// transaction. No-op when telemetry is disabled.
pub async fn snapshot_sqlite_state_metrics(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    if !METRICS.is_enabled() {
        return Ok(());
    }
    let m = &*SQLITE_STATE;

    if let Ok(version) =
        sqlx::query_scalar::<_, i64>("SELECT COALESCE(MAX(version), 0) FROM schema_version")
            .fetch_one(pool)
            .await
    {
        m.schema_version.set(version);
    }

    let page_size: i64 = sqlx::query_scalar("PRAGMA page_size")
        .fetch_one(pool)
        .await?;
    let page_count: i64 = sqlx::query_scalar("PRAGMA page_count")
        .fetch_one(pool)
        .await?;
    m.db_size_bytes.set(page_count.saturating_mul(page_size));

    let free_pages: i64 = sqlx::query_scalar("PRAGMA freelist_count")
        .fetch_one(pool)
        .await?;
    m.free_pages.set(free_pages);

    // PRAGMA wal_checkpoint(PASSIVE) → (busy, log_frames, checkpointed_frames).
    // PASSIVE never blocks; on a non-WAL DB it yields (0, -1, -1).
    if let Some(row) = sqlx::query("PRAGMA wal_checkpoint(PASSIVE)")
        .fetch_optional(pool)
        .await?
    {
        let log_frames: i64 = row.try_get(1).unwrap_or(0);
        let checkpointed: i64 = row.try_get(2).unwrap_or(0);
        let pending = (log_frames - checkpointed).max(0);
        m.wal_frames_pending.set(pending);
        m.wal_size_bytes.set(if log_frames > 0 {
            WAL_FILE_HEADER_BYTES + log_frames * (page_size + WAL_FRAME_HEADER_BYTES)
        } else {
            0
        });
    }

    for table in CANONICAL_TABLES {
        match count_table_rows(pool, table).await {
            Ok(n) => m.table_rows.with_label_values(&[table]).set(n),
            Err(e) => debug!("state-DB table_rows: skip {table} ({e})"),
        }
    }

    if let Err(e) = maybe_integrity_check(pool, m).await {
        warn!("state-DB integrity check failed: {e}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::{SchemaManager, CURRENT_SCHEMA_VERSION};
    use chrono::{Duration, Utc};
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
    fn canonical_table_set_is_bounded_to_14() {
        assert_eq!(CANONICAL_TABLES.len(), 14);
    }

    #[test]
    fn integrity_due_logic() {
        let now = Utc::now();
        // Never checked → due.
        assert!(is_integrity_due(None, now));
        // Unparseable → due.
        assert!(is_integrity_due(Some("garbage"), now));
        // Checked 8 days ago → due.
        let eight_days_ago = (now - Duration::days(8)).to_rfc3339();
        assert!(is_integrity_due(Some(&eight_days_ago), now));
        // Checked 6 days ago → not due.
        let six_days_ago = (now - Duration::days(6)).to_rfc3339();
        assert!(!is_integrity_due(Some(&six_days_ago), now));
    }

    #[tokio::test]
    async fn snapshot_populates_phase1_gauges() {
        let pool = migrated_pool().await;
        snapshot_sqlite_state_metrics(&pool).await.unwrap();
        let m = &*SQLITE_STATE;

        assert_eq!(
            m.schema_version.get(),
            CURRENT_SCHEMA_VERSION as i64,
            "schema_version gauge reflects the live migration version"
        );
        assert!(
            m.db_size_bytes.get() > 0,
            "db_size_bytes > 0 after migration"
        );
        assert!(m.free_pages.get() >= 0);
        // First snapshot: last_integrity_check_at is NULL → check runs → ok=1.
        assert_eq!(
            m.integrity_ok.get(),
            1,
            "fresh migrated DB passes quick_check"
        );
    }

    #[tokio::test]
    async fn table_rows_only_uses_canonical_labels() {
        let pool = migrated_pool().await;
        snapshot_sqlite_state_metrics(&pool).await.unwrap();

        let families = METRICS.registry.gather();
        let fam = families
            .iter()
            .find(|f| f.name() == "wqm_memexd_state_db_table_rows")
            .expect("table_rows metric family present");
        for metric in fam.get_metric() {
            let label = metric
                .get_label()
                .iter()
                .find(|l| l.name() == "table")
                .map(|l| l.value())
                .unwrap_or("");
            assert!(
                CANONICAL_TABLES.contains(&label),
                "unexpected table label: {label}"
            );
        }
    }

    #[tokio::test]
    async fn integrity_check_skipped_when_recent() {
        let pool = migrated_pool().await;
        // Stamp a recent check → snapshot must NOT re-run quick_check (which
        // would overwrite the timestamp). We assert the timestamp is unchanged.
        let recent = Utc::now().to_rfc3339();
        sqlx::query("UPDATE db_maintenance SET last_integrity_check_at = ?1 WHERE id = 1")
            .bind(&recent)
            .execute(&pool)
            .await
            .unwrap();

        snapshot_sqlite_state_metrics(&pool).await.unwrap();

        let after: Option<String> =
            sqlx::query_scalar("SELECT last_integrity_check_at FROM db_maintenance WHERE id = 1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(
            after.as_deref(),
            Some(recent.as_str()),
            "recent integrity check must not be re-run within the weekly window"
        );
    }
}
