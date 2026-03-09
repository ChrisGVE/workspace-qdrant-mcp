//! Migration v29: Add language column to processing_timings table.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V29Migration;

#[async_trait]
impl Migration for V29Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v29: Adding language column to processing_timings");

        let has_language: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('processing_timings') WHERE name = 'language'",
        )
        .fetch_one(pool)
        .await?;

        if !has_language {
            debug!("Adding language column to processing_timings");
            sqlx::query("ALTER TABLE processing_timings ADD COLUMN language TEXT")
                .execute(pool)
                .await?;
        } else {
            debug!("language column already exists, skipping ALTER TABLE");
        }

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_processing_timings_language \
             ON processing_timings (language)",
        )
        .execute(pool)
        .await?;

        info!("Migration v29 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        29
    }
    fn description(&self) -> &'static str {
        "Add language column to processing_timings"
    }
}
