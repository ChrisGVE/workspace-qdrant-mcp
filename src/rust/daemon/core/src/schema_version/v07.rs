//! Migration v7: Add is_archived column to watch_folders.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::SchemaError;
use super::migration::Migration;

pub struct V07Migration;

#[async_trait]
impl Migration for V07Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v7: Adding is_archived column to watch_folders");

        use crate::watch_folders_schema::MIGRATE_V7_ARCHIVE_SQL;

        let has_archived: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_archived'"
        )
        .fetch_one(pool).await?;

        if !has_archived {
            debug!("Running ALTER TABLE: {}", MIGRATE_V7_ARCHIVE_SQL);
            sqlx::query(MIGRATE_V7_ARCHIVE_SQL).execute(pool).await?;
        } else {
            debug!("is_archived column already exists, skipping ALTER TABLE");
        }

        info!("Migration v7 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 7 }
    fn description(&self) -> &'static str { "Add is_archived column to watch_folders" }
}
