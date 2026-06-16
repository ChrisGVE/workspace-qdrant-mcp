//! Migration v45: add `size_bytes` column + partial index to `unified_queue`
//! (#133 queue-health F1).
//!
//! Drain-time backlog estimation needs *pending bytes*, not item count
//! (`queue_depth` is a count only). This migration adds a nullable
//! `size_bytes INTEGER` column (NULL = unknown size, e.g. non-file items) and a
//! partial index over pending rows with a known size, so the poll-loop drain
//! aggregation stays cheap on large backlogs.
//!
//! Transaction ownership (v35+ convention): this migration wraps its DDL body in
//! an explicit `BEGIN IMMEDIATE` / `COMMIT` so the column + index are applied
//! atomically. A pure `ADD COLUMN` + `CREATE INDEX` needs no FK rebuild, so no
//! `ForeignKeysGuard` is required. `ADD COLUMN` is guarded by a column-existence
//! check so a re-run (e.g. after an up()-ran-but-unrecorded crash) is a no-op
//! rather than a "duplicate column name" failure.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V45Migration;

const ADD_SIZE_BYTES_COLUMN: &str = "ALTER TABLE unified_queue ADD COLUMN size_bytes INTEGER";

const CREATE_PENDING_SIZE_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_unified_queue_pending_size
    ON unified_queue(size_bytes)
    WHERE status = 'pending' AND size_bytes IS NOT NULL
"#;

#[async_trait]
impl Migration for V45Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v45: add size_bytes column + partial index to unified_queue");

        let mut conn = pool.acquire().await?;

        // Idempotency: ADD COLUMN is not natively idempotent (re-running raises
        // "duplicate column name"). Skip the ALTER if the column already exists.
        let column_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('unified_queue') \
             WHERE name = 'size_bytes')",
        )
        .fetch_one(&mut *conn)
        .await?;

        conn.execute("BEGIN IMMEDIATE").await?;
        if !column_exists {
            conn.execute(ADD_SIZE_BYTES_COLUMN).await?;
        }
        conn.execute(CREATE_PENDING_SIZE_INDEX.trim()).await?;
        conn.execute("COMMIT").await?;

        debug!("Migration v45: size_bytes column and partial index ready");
        Ok(())
    }

    fn version(&self) -> i32 {
        45
    }

    fn description(&self) -> &'static str {
        "Add size_bytes column + partial pending-size index to unified_queue"
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

    async fn column_present(pool: &SqlitePool) -> bool {
        sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('unified_queue') \
             WHERE name = 'size_bytes')",
        )
        .fetch_one(pool)
        .await
        .unwrap()
    }

    async fn index_present(pool: &SqlitePool) -> bool {
        sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master \
             WHERE type='index' AND name='idx_unified_queue_pending_size')",
        )
        .fetch_one(pool)
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_v45_adds_column_and_index() {
        let pool = migrated_pool().await;
        assert!(
            column_present(&pool).await,
            "size_bytes column should exist after v45"
        );
        assert!(
            index_present(&pool).await,
            "idx_unified_queue_pending_size should exist after v45"
        );
    }

    #[tokio::test]
    async fn test_v45_column_is_nullable_default_null() {
        let pool = migrated_pool().await;
        // Insert a minimal pending row without specifying size_bytes; it must be NULL.
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, item_type, op, tenant_id, collection, status, idempotency_key) \
             VALUES ('q-null', 'file', 'add', 't1', 'projects', 'pending', 'idem-null')",
        )
        .execute(&pool)
        .await
        .unwrap();
        let size: Option<i64> =
            sqlx::query_scalar("SELECT size_bytes FROM unified_queue WHERE queue_id = 'q-null'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(
            size.is_none(),
            "size_bytes defaults to NULL when unspecified"
        );
    }

    #[tokio::test]
    async fn test_v45_is_idempotent_on_rerun() {
        // Simulates the up()-ran-but-unrecorded crash window: running v45 again
        // when the column + index already exist must be a no-op, not an error.
        let pool = migrated_pool().await;
        let mgr = SchemaManager::new(pool.clone());
        mgr.run_migration(45)
            .await
            .expect("v45 re-run must be idempotent");
        assert!(column_present(&pool).await);
        assert!(index_present(&pool).await);
    }

    #[tokio::test]
    async fn test_v45_preserves_existing_rows_with_null_size() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let mgr = SchemaManager::new(pool.clone());
        // Migrate only up to v44 (the state before v45), insert a row, then apply v45.
        for v in 1..=44 {
            mgr.run_migration(v).await.unwrap();
        }
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, item_type, op, tenant_id, collection, status, idempotency_key) \
             VALUES ('pre-v45', 'file', 'add', 't1', 'projects', 'pending', 'idem-pre-v45')",
        )
        .execute(&pool)
        .await
        .unwrap();
        mgr.run_migration(45).await.unwrap();

        // Pre-existing row survives and carries NULL size_bytes (no backfill).
        let size: Option<i64> =
            sqlx::query_scalar("SELECT size_bytes FROM unified_queue WHERE queue_id = 'pre-v45'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(size.is_none(), "pre-v45 rows survive with NULL size_bytes");
    }
}
