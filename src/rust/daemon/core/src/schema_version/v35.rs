//! Migration v35: per-tenant `active_provider` column and `reembed` op value.
//!
//! Two changes (PRD §5.10):
//! 1. Add `active_provider TEXT DEFAULT 'openai_compatible'` to `watch_folders`
//!    so each tenant records the provider that produced its current vectors.
//! 2. Extend the `unified_queue.op` CHECK constraint to include `'reembed'`.
//!    SQLite cannot ALTER an existing CHECK; we rename → recreate → copy → drop.
//!
//! The recreated table DDL must mirror `CREATE_UNIFIED_QUEUE_SQL` in
//! `unified_queue_schema/sql.rs` — which is also updated to include `'reembed'`
//! so fresh installs and migrated databases agree.

use async_trait::async_trait;
use sqlx::{Executor, SqlitePool};
use tracing::{debug, info};

use super::migration::Migration;
use super::SchemaError;

pub struct V35Migration;

#[async_trait]
impl Migration for V35Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v35: adding watch_folders.active_provider and unified_queue.op='reembed'");

        let mut conn = pool.acquire().await?;

        // Part 1 — additive column.
        let watch_folders_sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='watch_folders'",
        )
        .fetch_one(&mut *conn)
        .await?;
        if !watch_folders_sql.contains("active_provider") {
            debug!("Adding active_provider column to watch_folders");
            conn.execute(
                "ALTER TABLE watch_folders ADD COLUMN active_provider TEXT \
                 DEFAULT 'openai_compatible'",
            )
            .await?;
        } else {
            debug!("active_provider column already present, skipping ALTER");
        }

        // Part 2 — rebuild unified_queue with the extended op CHECK.
        let queue_sql: String = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
        )
        .fetch_one(&mut *conn)
        .await?;
        if queue_sql.contains("'reembed'") {
            info!("Migration v35: unified_queue already accepts 'reembed', no rebuild needed");
            return Ok(());
        }

        // Rename → recreate → copy → drop. PRAGMA scope is per-connection.
        conn.execute("PRAGMA foreign_keys = OFF").await?;
        conn.execute("DROP TABLE IF EXISTS unified_queue_old")
            .await?;

        debug!("Renaming unified_queue → unified_queue_old");
        conn.execute("ALTER TABLE unified_queue RENAME TO unified_queue_old")
            .await?;

        debug!("Recreating unified_queue with extended op CHECK including 'reembed'");
        conn.execute(crate::CREATE_UNIFIED_QUEUE_SQL).await?;

        debug!("Copying rows from unified_queue_old");
        conn.execute(
            "INSERT INTO unified_queue (\
                 queue_id, item_type, op, tenant_id, collection, status, \
                 created_at, updated_at, lease_until, worker_id, idempotency_key, \
                 payload_json, retry_count, max_retries, error_message, \
                 last_error_at, branch, metadata, file_path, qdrant_status, \
                 search_status, decision_json\
             ) SELECT \
                 queue_id, item_type, op, tenant_id, collection, status, \
                 created_at, updated_at, lease_until, worker_id, idempotency_key, \
                 payload_json, retry_count, max_retries, error_message, \
                 last_error_at, branch, metadata, file_path, qdrant_status, \
                 search_status, decision_json \
             FROM unified_queue_old",
        )
        .await?;

        conn.execute("DROP TABLE unified_queue_old").await?;

        debug!("Recreating unified_queue indexes");
        for stmt in crate::CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            conn.execute(*stmt).await?;
        }

        conn.execute("PRAGMA foreign_keys = ON").await?;

        info!("Migration v35 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        35
    }

    fn description(&self) -> &'static str {
        "Add watch_folders.active_provider; extend unified_queue.op to include 'reembed'"
    }
}
