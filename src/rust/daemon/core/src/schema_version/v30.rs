//! Migration v30: Add last_corrected_n to corpus_statistics for IDF drift tracking.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V30Migration;

#[async_trait]
impl Migration for V30Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v30: Adding last_corrected_n to corpus_statistics");

        sqlx::query(
            "ALTER TABLE corpus_statistics \
             ADD COLUMN last_corrected_n INTEGER NOT NULL DEFAULT 0",
        )
        .execute(pool)
        .await?;

        info!("Migration v30 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        30
    }

    fn description(&self) -> &'static str {
        "Add last_corrected_n to corpus_statistics for IDF drift tracking"
    }
}
