//! Migration v11: Queue taxonomy overhaul.
//!
//! Migrates item_type and op values in unified_queue to the new taxonomy:
//! - ItemType: content→text, project/library/delete_tenant→tenant, delete_document→doc
//! - QueueOperation: ingest→add
//! - Rename items: item_type='rename' becomes item_type='file', op='rename'
//!
//! SQLite CHECK constraints cannot be altered, so we recreate the table.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::info;

use super::SchemaError;
use super::migration::Migration;

pub struct V11Migration;

#[async_trait]
impl Migration for V11Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v11: Queue taxonomy overhaul (item_type + op values)");

        // Check if migration is needed by inspecting the table DDL
        let table_sql: Option<String> = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'"
        )
        .fetch_optional(pool).await?;

        let needs_migration = match &table_sql {
            Some(sql) => {
                sql.contains("'content'") || sql.contains("'ingest'") || sql.contains("'project'")
            }
            None => false,
        };

        if !needs_migration {
            info!("Migration v11: Table already has correct schema, skipping");
            return Ok(());
        }

        // Clean up any leftover temp table from a previous failed attempt
        sqlx::query("DROP TABLE IF EXISTS unified_queue_v11")
            .execute(pool).await?;

        sqlx::query(r#"
            CREATE TABLE unified_queue_v11 (
                queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
                item_type TEXT NOT NULL CHECK (item_type IN (
                    'text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'
                )),
                op TEXT NOT NULL CHECK (op IN ('add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset')),
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0 CHECK (priority >= 0 AND priority <= 10),
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                    'pending', 'in_progress', 'done', 'failed'
                )),
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                lease_until TEXT,
                worker_id TEXT,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL DEFAULT '{}',
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                error_message TEXT,
                last_error_at TEXT,
                branch TEXT DEFAULT 'main',
                metadata TEXT DEFAULT '{}',
                file_path TEXT UNIQUE
            )
        "#).execute(pool).await?;

        // Copy data with value transformations
        sqlx::query(r#"
            INSERT INTO unified_queue_v11 (
                queue_id, item_type, op, tenant_id, collection, status,
                created_at, updated_at, lease_until, worker_id, idempotency_key,
                payload_json, retry_count, max_retries, error_message, last_error_at,
                branch, metadata, file_path
            )
            SELECT
                queue_id,
                CASE item_type
                    WHEN 'content' THEN 'text'
                    WHEN 'project' THEN 'tenant'
                    WHEN 'library' THEN 'tenant'
                    WHEN 'delete_tenant' THEN 'tenant'
                    WHEN 'delete_document' THEN 'doc'
                    WHEN 'rename' THEN 'file'
                    ELSE item_type
                END,
                CASE
                    WHEN item_type = 'rename' THEN 'rename'
                    WHEN op = 'ingest' THEN 'add'
                    ELSE op
                END,
                tenant_id, collection, status,
                created_at, updated_at, lease_until, worker_id, idempotency_key,
                payload_json, retry_count, max_retries, error_message, last_error_at,
                branch, metadata, file_path
            FROM unified_queue
        "#).execute(pool).await?;

        sqlx::query("DROP TABLE unified_queue")
            .execute(pool).await?;
        sqlx::query("ALTER TABLE unified_queue_v11 RENAME TO unified_queue")
            .execute(pool).await?;

        // Recreate indexes
        use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_INDEXES_SQL;
        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        info!("Migration v11 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 11 }
    fn description(&self) -> &'static str { "Queue taxonomy overhaul" }
}
