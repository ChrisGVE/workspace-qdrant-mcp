//! Migration v44: Add file_type and embedding_engine columns to
//! processing_timings (Task 73, E3 — CLI per-dimension perf breakdown).

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V44Migration;

#[async_trait]
impl Migration for V44Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v44: Adding file_type and embedding_engine to processing_timings");

        for column in ["file_type", "embedding_engine"] {
            let exists: bool = sqlx::query_scalar(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('processing_timings') WHERE name = ?",
            )
            .bind(column)
            .fetch_one(pool)
            .await?;

            if !exists {
                debug!("Adding {column} column to processing_timings");
                // Column name is from a fixed allowlist above, not user input.
                sqlx::query(&format!(
                    "ALTER TABLE processing_timings ADD COLUMN {column} TEXT"
                ))
                .execute(pool)
                .await?;
            } else {
                debug!("{column} column already exists, skipping ALTER TABLE");
            }
        }

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_processing_timings_file_type \
             ON processing_timings (file_type)",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_processing_timings_embedding_engine \
             ON processing_timings (embedding_engine)",
        )
        .execute(pool)
        .await?;

        info!("Migration v44 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        44
    }

    fn description(&self) -> &'static str {
        "Add file_type and embedding_engine columns to processing_timings"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use sqlx::Row;

    /// Build a raw in-memory pool with a *pre-v44* processing_timings table
    /// (no file_type / embedding_engine columns) and one populated row, so the
    /// migration runs against existing data exactly as it would on upgrade.
    async fn pre_v44_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE processing_timings (\
                 id INTEGER PRIMARY KEY, \
                 queue_id TEXT, \
                 phase TEXT, \
                 duration_ms INTEGER, \
                 created_at TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO processing_timings (queue_id, phase, duration_ms, created_at) \
             VALUES ('q1', 'embed', 42, datetime('now'))",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    async fn column_exists(pool: &SqlitePool, column: &str) -> bool {
        sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('processing_timings') WHERE name = ?",
        )
        .bind(column)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_v44_adds_columns_to_populated_pre_v44_table() {
        let pool = pre_v44_pool().await;
        assert!(!column_exists(&pool, "file_type").await);
        assert!(!column_exists(&pool, "embedding_engine").await);

        V44Migration.up(&pool).await.unwrap();

        assert!(column_exists(&pool, "file_type").await);
        assert!(column_exists(&pool, "embedding_engine").await);

        // The pre-existing row keeps its data; the new columns are NULL, and the
        // perf query path's COALESCE(col, '') folds that NULL to '' (the
        // "(unknown)" bucket) rather than dropping the row or erroring.
        let row = sqlx::query(
            "SELECT duration_ms, file_type, embedding_engine, \
                    COALESCE(file_type, '') AS ft, COALESCE(embedding_engine, '') AS ee \
             FROM processing_timings WHERE queue_id = 'q1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.get::<i64, _>("duration_ms"), 42);
        assert!(row.get::<Option<String>, _>("file_type").is_none());
        assert!(row.get::<Option<String>, _>("embedding_engine").is_none());
        assert_eq!(row.get::<String, _>("ft"), "");
        assert_eq!(row.get::<String, _>("ee"), "");
    }

    #[tokio::test]
    async fn test_v44_is_idempotent() {
        let pool = pre_v44_pool().await;
        V44Migration.up(&pool).await.unwrap();
        // Second run must not error on the already-present columns/indexes.
        V44Migration.up(&pool).await.unwrap();
        assert!(column_exists(&pool, "file_type").await);
        assert!(column_exists(&pool, "embedding_engine").await);
    }
}
