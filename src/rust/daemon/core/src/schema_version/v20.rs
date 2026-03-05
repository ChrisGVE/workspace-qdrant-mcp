//! Migration v20: Add qdrant_status, search_status, decision_json to unified_queue.
//!
//! Per-destination status tracking for dual-write queue processing.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V20Migration;

#[async_trait]
impl Migration for V20Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v20: Adding per-destination status columns to unified_queue");

        use crate::unified_queue_schema::{
            CREATE_QDRANT_STATUS_INDEX_SQL, CREATE_SEARCH_STATUS_INDEX_SQL,
            MIGRATE_V20_ADD_COLUMNS_SQL,
        };

        let has_qdrant_status: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('unified_queue') WHERE name = 'qdrant_status'"
        )
        .fetch_one(pool).await?;

        if !has_qdrant_status {
            for alter_sql in MIGRATE_V20_ADD_COLUMNS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql).execute(pool).await?;
            }
        } else {
            debug!("qdrant_status column already exists, skipping ALTER TABLE");
        }

        // Backfill: set done items to both destinations done
        let backfilled: u64 = sqlx::query(
            "UPDATE unified_queue SET qdrant_status = 'done', search_status = 'done'
             WHERE status = 'done' AND qdrant_status = 'pending'",
        )
        .execute(pool)
        .await?
        .rows_affected();

        if backfilled > 0 {
            info!(
                "Backfilled qdrant_status/search_status for {} done items",
                backfilled
            );
        }

        sqlx::query(CREATE_QDRANT_STATUS_INDEX_SQL)
            .execute(pool)
            .await?;
        sqlx::query(CREATE_SEARCH_STATUS_INDEX_SQL)
            .execute(pool)
            .await?;

        info!("Migration v20 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        20
    }
    fn description(&self) -> &'static str {
        "Add per-destination status columns to unified_queue"
    }
}
