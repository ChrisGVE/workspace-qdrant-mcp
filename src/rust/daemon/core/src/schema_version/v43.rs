//! Migration v43: create db_maintenance table for vacuum/integrity timestamps
//! (PRD D5 DATA-N1 / PERF-N1).
//!
//! Single-row metadata table tracking when the state DB was last VACUUMed and
//! last integrity-checked. The snapshotter reads `last_vacuum_at` to populate
//! `wqm_memexd_state_db_last_vacuum_timestamp_seconds` (absent until the first
//! VACUUM runs — a NULL renders as "never vacuumed").

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V43Migration;

const CREATE_DB_MAINTENANCE_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS db_maintenance (
    id                      INTEGER PRIMARY KEY CHECK (id = 1),
    last_vacuum_at          TEXT,
    last_integrity_check_at TEXT
)
"#;

// Seed the single metadata row (NULL timestamps = never run).
const SEED_DB_MAINTENANCE_ROW: &str = "INSERT OR IGNORE INTO db_maintenance (id) VALUES (1)";

#[async_trait]
impl Migration for V43Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v43: create db_maintenance table");
        let mut conn = pool.acquire().await?;
        conn.execute(CREATE_DB_MAINTENANCE_TABLE).await?;
        conn.execute(SEED_DB_MAINTENANCE_ROW).await?;
        debug!("Migration v43: db_maintenance table created and seeded");
        Ok(())
    }

    fn version(&self) -> i32 {
        43
    }

    fn description(&self) -> &'static str {
        "Create db_maintenance table for vacuum/integrity timestamps"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
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

    #[tokio::test]
    async fn test_v43_creates_db_maintenance_table_with_seed_row() {
        let pool = setup_pool().await;
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='db_maintenance')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists, "db_maintenance table should exist after migration");

        // Single seeded row with NULL timestamps (never vacuumed/checked).
        let (count, vac): (i64, Option<String>) =
            sqlx::query_as("SELECT COUNT(*), MAX(last_vacuum_at) FROM db_maintenance WHERE id = 1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count, 1, "exactly one metadata row");
        assert!(vac.is_none(), "last_vacuum_at starts NULL");
    }

    #[tokio::test]
    async fn test_v43_single_row_constraint() {
        let pool = setup_pool().await;
        // The CHECK (id = 1) constraint forbids a second row.
        let res = sqlx::query("INSERT INTO db_maintenance (id) VALUES (2)")
            .execute(&pool)
            .await;
        assert!(res.is_err(), "id != 1 must be rejected by CHECK constraint");
    }
}
