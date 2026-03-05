//! Migration v27: Drop max_retries column from unified_queue (centralized in config).

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V27Migration;

#[async_trait]
impl Migration for V27Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v27: Drop max_retries column from unified_queue (centralized in config)");

        // SQLite 3.35.0+ supports ALTER TABLE DROP COLUMN.
        // For older SQLite, the column will remain but is unused.
        let result = sqlx::query("ALTER TABLE unified_queue DROP COLUMN max_retries")
            .execute(pool)
            .await;

        match result {
            Ok(_) => {
                info!("Migration v27 complete: max_retries column dropped");
            }
            Err(e) => {
                // Column may not exist (fresh schema) or SQLite too old
                debug!(
                    "Migration v27: max_retries column not dropped ({}), continuing",
                    e
                );
            }
        }

        Ok(())
    }

    fn version(&self) -> i32 {
        27
    }
    fn description(&self) -> &'static str {
        "Drop max_retries column from unified_queue"
    }
}
