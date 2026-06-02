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
