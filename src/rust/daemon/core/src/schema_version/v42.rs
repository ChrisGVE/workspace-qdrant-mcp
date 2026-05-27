//! Migration v42: create dead_letter_queue table for permanently failed items.

use async_trait::async_trait;
use sqlx::{Executor, Row, SqlitePool};
use tracing::{debug, info, warn};

use super::migration::Migration;
use super::SchemaError;

pub struct V42Migration;

const CREATE_DLQ_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    dlq_id              TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    original_queue_id   TEXT NOT NULL,
    item_type           TEXT,
    op                  TEXT,
    tenant_id           TEXT,
    collection          TEXT,
    branch              TEXT,
    payload_json        TEXT DEFAULT '{}',
    file_path           TEXT,
    error_category      TEXT NOT NULL,
    error_message       TEXT NOT NULL,
    retry_count         INTEGER DEFAULT 0,
    resurrection_count  INTEGER DEFAULT 0,
    final_failure_at    TEXT NOT NULL,
    moved_to_dlq_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata            TEXT DEFAULT '{}'
)
"#;

const CREATE_DLQ_INDEXES: &str = r#"
CREATE INDEX IF NOT EXISTS idx_dlq_tenant ON dead_letter_queue(tenant_id);
CREATE INDEX IF NOT EXISTS idx_dlq_category ON dead_letter_queue(error_category);
CREATE INDEX IF NOT EXISTS idx_dlq_moved_at ON dead_letter_queue(moved_to_dlq_at);
CREATE INDEX IF NOT EXISTS idx_dlq_original_queue_id ON dead_letter_queue(original_queue_id)
"#;

#[async_trait]
impl Migration for V42Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v42: create dead_letter_queue table");

        let mut conn = pool.acquire().await?;

        conn.execute(CREATE_DLQ_TABLE).await?;

        for stmt in CREATE_DLQ_INDEXES.split(';') {
            let trimmed = stmt.trim();
            if !trimmed.is_empty() {
                conn.execute(trimmed).await?;
            }
        }

        debug!("Migration v42: DLQ table and indexes created");

        let swept = self.sweep_existing_failures(&mut conn).await?;
        if swept > 0 {
            info!(
                "Migration v42: swept {} permanently failed items into DLQ",
                swept
            );
        }

        info!("Migration v42 complete");
        Ok(())
    }

    fn version(&self) -> i32 {
        42
    }

    fn description(&self) -> &'static str {
        "Create dead_letter_queue table for permanently failed items"
    }
}

impl V42Migration {
    async fn sweep_existing_failures(
        &self,
        conn: &mut sqlx::pool::PoolConnection<sqlx::Sqlite>,
    ) -> Result<i64, SchemaError> {
        let has_queue: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='unified_queue')",
        )
        .fetch_one(&mut **conn)
        .await?;

        if !has_queue {
            return Ok(0);
        }

        let rows = sqlx::query(
            "SELECT queue_id, item_type, op, tenant_id, collection, branch, \
                    payload_json, file_path, error_message, retry_count, metadata, \
                    updated_at \
             FROM unified_queue \
             WHERE status = 'failed' \
               AND (error_message LIKE '[permanent_%' \
                    OR error_message LIKE '%exhausted%')",
        )
        .fetch_all(&mut **conn)
        .await?;

        let count = rows.len() as i64;
        if count == 0 {
            return Ok(0);
        }

        conn.execute("BEGIN IMMEDIATE").await?;

        for row in &rows {
            let queue_id: String = row.get("queue_id");
            let error_msg: String = row.get("error_message");
            let error_category = if error_msg.starts_with("[permanent_data]") {
                "permanent_data"
            } else if error_msg.starts_with("[permanent_gone]") {
                "permanent_gone"
            } else {
                "permanent_exhausted"
            };

            let metadata_str: Option<String> = row.get("metadata");
            let resurrection_count = metadata_str
                .as_deref()
                .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
                .and_then(|v| v.get("resurrection_count")?.as_i64())
                .unwrap_or(0);

            if let Err(e) = sqlx::query(
                "INSERT INTO dead_letter_queue \
                    (original_queue_id, item_type, op, tenant_id, collection, branch, \
                     payload_json, file_path, error_category, error_message, \
                     retry_count, resurrection_count, final_failure_at, metadata) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            )
            .bind(&queue_id)
            .bind(row.get::<Option<String>, _>("item_type"))
            .bind(row.get::<Option<String>, _>("op"))
            .bind(row.get::<Option<String>, _>("tenant_id"))
            .bind(row.get::<Option<String>, _>("collection"))
            .bind(row.get::<Option<String>, _>("branch"))
            .bind(row.get::<Option<String>, _>("payload_json"))
            .bind(row.get::<Option<String>, _>("file_path"))
            .bind(error_category)
            .bind(&error_msg)
            .bind(row.get::<i32, _>("retry_count"))
            .bind(resurrection_count)
            .bind(row.get::<String, _>("updated_at"))
            .bind(metadata_str.as_deref().unwrap_or("{}"))
            .execute(&mut **conn)
            .await
            {
                warn!("Migration v42: failed to sweep item {}: {}", queue_id, e);
                continue;
            }

            if let Err(e) = sqlx::query("DELETE FROM unified_queue WHERE queue_id = ?1")
                .bind(&queue_id)
                .execute(&mut **conn)
                .await
            {
                warn!(
                    "Migration v42: failed to delete swept item {}: {}",
                    queue_id, e
                );
            }
        }

        conn.execute("COMMIT").await?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        pool
    }

    #[tokio::test]
    async fn test_v42_creates_dlq_table() {
        let pool = setup_pool().await;
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='dead_letter_queue')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            exists,
            "dead_letter_queue table should exist after migration"
        );
    }

    #[tokio::test]
    async fn test_v42_dlq_indexes_exist() {
        let pool = setup_pool().await;
        let indexes: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='dead_letter_queue'",
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        assert!(indexes.contains(&"idx_dlq_tenant".to_string()));
        assert!(indexes.contains(&"idx_dlq_category".to_string()));
        assert!(indexes.contains(&"idx_dlq_moved_at".to_string()));
        assert!(indexes.contains(&"idx_dlq_original_queue_id".to_string()));
    }

    #[tokio::test]
    async fn test_v42_sweep_moves_permanent_failures() {
        let pool = setup_pool().await;

        sqlx::query(
            "INSERT INTO unified_queue \
                (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
                 branch, status, error_message, retry_count, payload_json) \
             VALUES ('q1', 'k1', 'doc', 'add', 't1', 'projects', \
                     'main', 'failed', '[permanent_data] bad format', 3, '{}')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let migration = V42Migration;
        let swept = {
            let mut conn = pool.acquire().await.unwrap();
            migration.sweep_existing_failures(&mut conn).await.unwrap()
        };
        assert_eq!(swept, 1);

        let dlq_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM dead_letter_queue")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(dlq_count, 1);

        let queue_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(queue_count, 0);
    }
}
