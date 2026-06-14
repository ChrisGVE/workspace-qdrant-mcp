//! Migration v46: add `control_baseline` — metrics-switchboard slow-lane
//! persistence.
//!
//! The switchboard's control lane carries EWMA accumulators (the queue-health
//! verdict signal). The slow lane survives daemon restarts via one generic
//! table, `control_baseline`, keyed by `(metric_id, field, labels, lane)`. This
//! REPLACES the bespoke `queue_health_baseline` planned in #133 task 17 (now
//! never created) and serves ALL control metrics, not just queue-health.
//!
//! Additive `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` — both
//! natively idempotent, no `ALTER`, no FK rebuild. The DDL is wrapped in an
//! explicit `BEGIN IMMEDIATE`/`COMMIT` (v35+ convention) so table + index apply
//! atomically and a re-run after an up()-ran-but-unrecorded crash is a no-op.
//!
//! Only `lane = 'slow'` rows are persisted (the fast lane is rebuilt live).
//! Telemetry is NEVER written here — this table is control-lane only.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V46Migration;

const CREATE_CONTROL_BASELINE: &str = r#"
CREATE TABLE IF NOT EXISTS control_baseline (
    metric_id    TEXT    NOT NULL,
    field        TEXT    NOT NULL,
    labels       TEXT    NOT NULL,
    lane         TEXT    NOT NULL,
    value        REAL    NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (metric_id, field, labels, lane)
)
"#;

const CREATE_UPDATED_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_control_baseline_updated
    ON control_baseline (updated_at)
"#;

#[async_trait]
impl Migration for V46Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v46: add control_baseline table for switchboard slow-lane persistence");

        let mut conn = pool.acquire().await?;
        conn.execute("BEGIN IMMEDIATE").await?;
        conn.execute(CREATE_CONTROL_BASELINE.trim()).await?;
        conn.execute(CREATE_UPDATED_INDEX.trim()).await?;
        conn.execute("COMMIT").await?;

        debug!("Migration v46: control_baseline table and updated_at index ready");
        Ok(())
    }

    fn version(&self) -> i32 {
        46
    }

    fn description(&self) -> &'static str {
        "Add control_baseline table for metrics-switchboard slow-lane persistence"
    }
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

    async fn table_present(pool: &SqlitePool) -> bool {
        sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master \
             WHERE type='table' AND name='control_baseline')",
        )
        .fetch_one(pool)
        .await
        .unwrap()
    }

    async fn index_present(pool: &SqlitePool) -> bool {
        sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master \
             WHERE type='index' AND name='idx_control_baseline_updated')",
        )
        .fetch_one(pool)
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_v46_creates_table_and_index() {
        let pool = migrated_pool().await;
        assert!(
            table_present(&pool).await,
            "control_baseline should exist after v46"
        );
        assert!(
            index_present(&pool).await,
            "idx_control_baseline_updated should exist after v46"
        );
    }

    #[tokio::test]
    async fn test_v46_pk_enforced() {
        let pool = migrated_pool().await;
        sqlx::query(
            "INSERT INTO control_baseline (metric_id, field, labels, lane, value) \
             VALUES ('EmbedderLatency', 'embed_ms', '{\"model\":\"fastembed\"}', 'slow', 1.0)",
        )
        .execute(&pool)
        .await
        .unwrap();
        // Same PK → conflict (no ON CONFLICT clause here, so it must error).
        let dup = sqlx::query(
            "INSERT INTO control_baseline (metric_id, field, labels, lane, value) \
             VALUES ('EmbedderLatency', 'embed_ms', '{\"model\":\"fastembed\"}', 'slow', 2.0)",
        )
        .execute(&pool)
        .await;
        assert!(dup.is_err(), "duplicate PK must be rejected");
    }

    #[tokio::test]
    async fn test_v46_is_idempotent_on_rerun() {
        let pool = migrated_pool().await;
        let mgr = SchemaManager::new(pool.clone());
        mgr.run_migration(46)
            .await
            .expect("v46 re-run must be idempotent");
        assert!(table_present(&pool).await);
        assert!(index_present(&pool).await);
    }
}
