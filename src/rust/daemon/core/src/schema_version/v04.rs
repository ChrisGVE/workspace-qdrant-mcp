//! Migration v4: Add is_paused and pause_start_time columns to watch_folders.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V04Migration;

#[async_trait]
impl Migration for V04Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v4: Adding pause columns to watch_folders");

        use crate::watch_folders_schema::MIGRATE_V4_PAUSE_SQL;

        let has_is_paused: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_paused'",
        )
        .fetch_one(pool)
        .await?;

        if !has_is_paused {
            for alter_sql in MIGRATE_V4_PAUSE_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql).execute(pool).await?;
            }
        } else {
            debug!("Pause columns already exist, skipping ALTER TABLE");
        }

        info!("Migration v4 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        4
    }
    fn description(&self) -> &'static str {
        "Add pause columns to watch_folders"
    }
}
