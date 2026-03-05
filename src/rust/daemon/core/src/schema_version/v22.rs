//! Migration v22: Remove unused priority column from unified_queue.
//!
//! Priority is computed dynamically at dequeue time. The stored column was always 0.
//! Drop and recreate the table (pre-release, no data to preserve).

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V22Migration;

#[async_trait]
impl Migration for V22Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v22: Remove unused priority column from unified_queue");

        use crate::unified_queue_schema::{
            CREATE_QDRANT_STATUS_INDEX_SQL, CREATE_SEARCH_STATUS_INDEX_SQL,
            CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL,
        };

        sqlx::query("DROP TABLE IF EXISTS unified_queue")
            .execute(pool)
            .await?;

        sqlx::query(CREATE_UNIFIED_QUEUE_SQL).execute(pool).await?;

        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        // Recreate v20 indexes
        sqlx::query(CREATE_QDRANT_STATUS_INDEX_SQL)
            .execute(pool)
            .await?;
        sqlx::query(CREATE_SEARCH_STATUS_INDEX_SQL)
            .execute(pool)
            .await?;

        info!("Migration v22 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        22
    }
    fn description(&self) -> &'static str {
        "Remove unused priority column from unified_queue"
    }
}
