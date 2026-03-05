//! Migration v9: Add 'url' to unified_queue item_type CHECK constraint.
//!
//! SQLite does not support ALTER TABLE to modify CHECK constraints,
//! so we recreate the table preserving all existing data.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::migration::Migration;
use super::SchemaError;

pub struct V09Migration;

#[async_trait]
impl Migration for V09Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v9: Adding 'url' item_type to unified_queue CHECK constraint");

        use crate::unified_queue_schema::{
            CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL,
        };

        sqlx::query("ALTER TABLE unified_queue RENAME TO unified_queue_old")
            .execute(pool)
            .await?;

        sqlx::query(CREATE_UNIFIED_QUEUE_SQL).execute(pool).await?;

        // Copy all existing data
        sqlx::query(
            "INSERT INTO unified_queue (
                queue_id, item_type, op, tenant_id, collection,
                status, created_at, updated_at, lease_until, worker_id,
                idempotency_key, payload_json, retry_count, max_retries,
                error_message, last_error_at, branch, metadata, file_path
            )
            SELECT
                queue_id, item_type, op, tenant_id, collection,
                status, created_at, updated_at, lease_until, worker_id,
                idempotency_key, payload_json, retry_count, max_retries,
                error_message, last_error_at, branch, metadata, file_path
            FROM unified_queue_old",
        )
        .execute(pool)
        .await?;

        sqlx::query("DROP TABLE unified_queue_old")
            .execute(pool)
            .await?;

        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v9 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        9
    }
    fn description(&self) -> &'static str {
        "Add url item_type to unified_queue"
    }
}
