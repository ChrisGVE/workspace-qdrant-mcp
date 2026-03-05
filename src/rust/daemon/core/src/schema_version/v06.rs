//! Migration v6: Add collection column to tracked_files.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V06Migration;

#[async_trait]
impl Migration for V06Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v6: Adding collection column to tracked_files");

        use crate::tracked_files_schema::MIGRATE_V6_SQL;

        let has_collection: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'collection'",
        )
        .fetch_one(pool)
        .await?;

        if !has_collection {
            debug!("Running ALTER TABLE: {}", MIGRATE_V6_SQL);
            sqlx::query(MIGRATE_V6_SQL).execute(pool).await?;
        } else {
            debug!("Collection column already exists, skipping ALTER TABLE");
        }

        info!("Migration v6 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        6
    }
    fn description(&self) -> &'static str {
        "Add collection column to tracked_files"
    }
}
