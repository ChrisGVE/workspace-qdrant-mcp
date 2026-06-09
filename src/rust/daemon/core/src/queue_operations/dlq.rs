//! Dead letter queue operations.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::{Executor, Row};
use tracing::info;
use wqm_common::timestamps::format_utc;

use super::{QueueError, QueueManager, QueueResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlqEntry {
    pub dlq_id: String,
    pub original_queue_id: String,
    pub item_type: Option<String>,
    pub op: Option<String>,
    pub tenant_id: Option<String>,
    pub collection: Option<String>,
    pub branch: Option<String>,
    pub file_path: Option<String>,
    pub error_category: String,
    pub error_message: String,
    pub retry_count: i32,
    pub resurrection_count: i32,
    pub final_failure_at: String,
    pub moved_to_dlq_at: String,
}

impl QueueManager {
    pub async fn move_to_dlq(&self, queue_id: &str) -> QueueResult<String> {
        let mut conn = self.pool.acquire().await?;
        conn.execute("BEGIN IMMEDIATE").await?;

        let row = sqlx::query(
            "SELECT queue_id, item_type, op, tenant_id, collection, branch, \
                    payload_json, file_path, error_message, retry_count, \
                    metadata, last_error_at, updated_at \
             FROM unified_queue WHERE queue_id = ?1",
        )
        .bind(queue_id)
        .fetch_optional(&mut *conn)
        .await?;

        let row = match row {
            Some(r) => r,
            None => {
                conn.execute("ROLLBACK").await.ok();
                return Err(QueueError::NotFound(queue_id.to_string()));
            }
        };

        let error_msg: String = row.get("error_message");
        let error_category = extract_error_category(&error_msg);
        let final_failure_at: String = row
            .try_get::<Option<String>, _>("last_error_at")
            .ok()
            .flatten()
            .or_else(|| {
                row.try_get::<Option<String>, _>("updated_at")
                    .ok()
                    .flatten()
            })
            .unwrap_or_else(|| format_utc(&Utc::now()));

        let metadata_str: Option<String> = row.get("metadata");
        let resurrection_count = metadata_str
            .as_deref()
            .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
            .and_then(|v| v.get("resurrection_count")?.as_i64())
            .unwrap_or(0);

        let dlq_id: String = sqlx::query_scalar(
            "INSERT INTO dead_letter_queue \
                (original_queue_id, item_type, op, tenant_id, collection, branch, \
                 payload_json, file_path, error_category, error_message, \
                 retry_count, resurrection_count, final_failure_at, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14) \
             RETURNING dlq_id",
        )
        .bind(queue_id)
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
        .bind(&final_failure_at)
        .bind(metadata_str.as_deref().unwrap_or("{}"))
        .fetch_one(&mut *conn)
        .await?;

        sqlx::query("DELETE FROM unified_queue WHERE queue_id = ?1")
            .bind(queue_id)
            .execute(&mut *conn)
            .await?;

        conn.execute("COMMIT").await?;

        info!(
            "Moved queue item {} to DLQ as {} (category={})",
            queue_id, dlq_id, error_category
        );
        Ok(dlq_id)
    }

    pub async fn replay_from_dlq(&self, dlq_id: &str, force: bool) -> QueueResult<String> {
        let mut conn = self.pool.acquire().await?;

        let row = sqlx::query("SELECT * FROM dead_letter_queue WHERE dlq_id = ?1")
            .bind(dlq_id)
            .fetch_optional(&mut *conn)
            .await?;

        let row = match row {
            Some(r) => r,
            None => return Err(QueueError::NotFound(dlq_id.to_string())),
        };

        let category: String = row.get("error_category");
        if category == "permanent_data" && !force {
            return Err(QueueError::InternalError(
                "Cannot replay permanent_data item without force=true".to_string(),
            ));
        }

        let new_queue_id = uuid::Uuid::new_v4().to_string();
        let replay_key = format!("replay-{}-{}", dlq_id, new_queue_id);
        let idempotency_key = {
            use sha2::{Digest, Sha256};
            let hash = Sha256::digest(replay_key.as_bytes());
            hash[..16]
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        };
        let now = format_utc(&Utc::now());
        let metadata =
            serde_json::json!({"replayed_from_dlq_id": dlq_id, "replayed_at": now}).to_string();

        conn.execute("BEGIN IMMEDIATE").await?;

        sqlx::query(
            "INSERT INTO unified_queue \
                (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
                 branch, payload_json, file_path, status, retry_count, metadata, \
                 created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'pending', 0, ?10, ?11, ?11)",
        )
        .bind(&new_queue_id)
        .bind(&idempotency_key)
        .bind(row.get::<Option<String>, _>("item_type"))
        .bind(row.get::<Option<String>, _>("op"))
        .bind(row.get::<Option<String>, _>("tenant_id"))
        .bind(row.get::<Option<String>, _>("collection"))
        .bind(row.get::<Option<String>, _>("branch"))
        .bind(row.get::<Option<String>, _>("payload_json"))
        .bind(row.get::<Option<String>, _>("file_path"))
        .bind(&metadata)
        .bind(&now)
        .execute(&mut *conn)
        .await?;

        sqlx::query("DELETE FROM dead_letter_queue WHERE dlq_id = ?1")
            .bind(dlq_id)
            .execute(&mut *conn)
            .await?;

        conn.execute("COMMIT").await?;

        info!(
            "Replayed DLQ item {} as new queue item {}",
            dlq_id, new_queue_id
        );
        Ok(new_queue_id)
    }

    pub async fn purge_dlq(
        &self,
        retention_days: u32,
        batch_size: usize,
    ) -> QueueResult<(i64, bool)> {
        let cutoff = Utc::now() - chrono::Duration::days(retention_days as i64);
        let cutoff_str = format_utc(&cutoff);

        let result = sqlx::query(
            "DELETE FROM dead_letter_queue WHERE rowid IN \
             (SELECT rowid FROM dead_letter_queue \
              WHERE moved_to_dlq_at < ?1 \
              ORDER BY moved_to_dlq_at ASC LIMIT ?2)",
        )
        .bind(&cutoff_str)
        .bind(batch_size as i64)
        .execute(&self.pool)
        .await?;

        let deleted = result.rows_affected() as i64;
        let has_more = deleted as usize >= batch_size;

        if deleted > 0 {
            info!(
                "Purged {} DLQ entries older than {} days (has_more={})",
                deleted, retention_days, has_more
            );
        }

        Ok((deleted, has_more))
    }

    pub async fn list_dlq(
        &self,
        tenant_id: Option<&str>,
        category: Option<&str>,
        limit: i32,
        offset: i32,
    ) -> QueueResult<(Vec<DlqEntry>, i64)> {
        let mut where_clauses = Vec::new();
        if tenant_id.is_some() {
            where_clauses.push("tenant_id = ?1");
        }
        if category.is_some() {
            where_clauses.push("error_category = ?2");
        }

        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let count_sql = format!("SELECT COUNT(*) FROM dead_letter_queue {}", where_clause);
        let list_sql = format!(
            "SELECT dlq_id, original_queue_id, item_type, op, tenant_id, collection, \
                    branch, file_path, error_category, error_message, \
                    retry_count, resurrection_count, final_failure_at, moved_to_dlq_at \
             FROM dead_letter_queue {} \
             ORDER BY moved_to_dlq_at DESC LIMIT ?3 OFFSET ?4",
            where_clause
        );

        let total: i64 = sqlx::query_scalar(&count_sql)
            .bind(tenant_id.unwrap_or(""))
            .bind(category.unwrap_or(""))
            .fetch_one(&self.pool)
            .await?;

        let rows = sqlx::query(&list_sql)
            .bind(tenant_id.unwrap_or(""))
            .bind(category.unwrap_or(""))
            .bind(limit)
            .bind(offset)
            .fetch_all(&self.pool)
            .await?;

        let entries = rows
            .iter()
            .map(|row| DlqEntry {
                dlq_id: row.get("dlq_id"),
                original_queue_id: row.get("original_queue_id"),
                item_type: row.get("item_type"),
                op: row.get("op"),
                tenant_id: row.get("tenant_id"),
                collection: row.get("collection"),
                branch: row.get("branch"),
                file_path: row.get("file_path"),
                error_category: row.get("error_category"),
                error_message: row.get("error_message"),
                retry_count: row.get("retry_count"),
                resurrection_count: row.get("resurrection_count"),
                final_failure_at: row.get("final_failure_at"),
                moved_to_dlq_at: row.get("moved_to_dlq_at"),
            })
            .collect();

        Ok((entries, total))
    }

    pub async fn get_dlq_entry(&self, dlq_id: &str) -> QueueResult<DlqEntry> {
        let row = sqlx::query(
            "SELECT dlq_id, original_queue_id, item_type, op, tenant_id, collection, \
                    branch, file_path, error_category, error_message, \
                    retry_count, resurrection_count, final_failure_at, moved_to_dlq_at \
             FROM dead_letter_queue WHERE dlq_id = ?1",
        )
        .bind(dlq_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(DlqEntry {
                dlq_id: row.get("dlq_id"),
                original_queue_id: row.get("original_queue_id"),
                item_type: row.get("item_type"),
                op: row.get("op"),
                tenant_id: row.get("tenant_id"),
                collection: row.get("collection"),
                branch: row.get("branch"),
                file_path: row.get("file_path"),
                error_category: row.get("error_category"),
                error_message: row.get("error_message"),
                retry_count: row.get("retry_count"),
                resurrection_count: row.get("resurrection_count"),
                final_failure_at: row.get("final_failure_at"),
                moved_to_dlq_at: row.get("moved_to_dlq_at"),
            }),
            None => Err(QueueError::NotFound(dlq_id.to_string())),
        }
    }
}

fn extract_error_category(error_msg: &str) -> &str {
    if let Some(rest) = error_msg.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            return &rest[..end];
        }
    }
    if error_msg.contains("exhausted") {
        return "permanent_exhausted";
    }
    "unknown"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup() -> QueueManager {
        let pool = SqlitePoolOptions::new()
            .max_connections(2)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        QueueManager::new(pool)
    }

    async fn insert_failed_item(qm: &QueueManager, queue_id: &str, error_msg: &str) {
        sqlx::query(
            "INSERT INTO unified_queue \
                (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
                 branch, status, error_message, retry_count, payload_json) \
             VALUES (?1, ?2, 'doc', 'add', 'test-tenant', 'projects', \
                     'main', 'failed', ?3, 3, '{}')",
        )
        .bind(queue_id)
        .bind(format!("key-{}", queue_id))
        .bind(error_msg)
        .execute(qm.pool())
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_move_to_dlq() {
        let qm = setup().await;
        insert_failed_item(&qm, "q-move-1", "[permanent_data] bad format").await;

        let dlq_id = qm.move_to_dlq("q-move-1").await.unwrap();
        assert!(!dlq_id.is_empty());

        let queue_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'q-move-1'")
                .fetch_one(qm.pool())
                .await
                .unwrap();
        assert_eq!(queue_count, 0);

        let entry = qm.get_dlq_entry(&dlq_id).await.unwrap();
        assert_eq!(entry.error_category, "permanent_data");
        assert_eq!(entry.original_queue_id, "q-move-1");
    }

    #[tokio::test]
    async fn test_move_to_dlq_not_found() {
        let qm = setup().await;
        let result = qm.move_to_dlq("nonexistent").await;
        assert!(matches!(result, Err(QueueError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_replay_from_dlq() {
        let qm = setup().await;
        insert_failed_item(&qm, "q-replay-1", "[transient_infrastructure] timeout").await;
        let dlq_id = qm.move_to_dlq("q-replay-1").await.unwrap();

        let new_queue_id = qm.replay_from_dlq(&dlq_id, false).await.unwrap();
        assert!(!new_queue_id.is_empty());

        let dlq_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM dead_letter_queue WHERE dlq_id = ?1")
                .bind(&dlq_id)
                .fetch_one(qm.pool())
                .await
                .unwrap();
        assert_eq!(dlq_count, 0);

        let status: String =
            sqlx::query_scalar("SELECT status FROM unified_queue WHERE queue_id = ?1")
                .bind(&new_queue_id)
                .fetch_one(qm.pool())
                .await
                .unwrap();
        assert_eq!(status, "pending");
    }

    #[tokio::test]
    async fn test_replay_permanent_data_requires_force() {
        let qm = setup().await;
        insert_failed_item(&qm, "q-perm-1", "[permanent_data] bad json").await;
        let dlq_id = qm.move_to_dlq("q-perm-1").await.unwrap();

        let result = qm.replay_from_dlq(&dlq_id, false).await;
        assert!(result.is_err());

        let result = qm.replay_from_dlq(&dlq_id, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_purge_dlq() {
        let qm = setup().await;
        insert_failed_item(&qm, "q-purge-1", "[permanent_data] old").await;
        qm.move_to_dlq("q-purge-1").await.unwrap();

        sqlx::query("UPDATE dead_letter_queue SET moved_to_dlq_at = '2020-01-01T00:00:00.000Z'")
            .execute(qm.pool())
            .await
            .unwrap();

        let (deleted, _has_more) = qm.purge_dlq(30, 500).await.unwrap();
        assert_eq!(deleted, 1);
    }

    #[tokio::test]
    async fn test_purge_dlq_zero_retention_purges_recent() {
        // #119: retention_days=0 → cutoff=now → purge every entry, including
        // ones moved to the DLQ moments ago. The 30-day default that a bare
        // `purge_dlq` used to substitute lives in the CLI now, so 0 reaching
        // here is an explicit "purge all" and must not leave recent entries.
        let qm = setup().await;
        insert_failed_item(&qm, "q-recent-1", "[permanent_data] fresh").await;
        qm.move_to_dlq("q-recent-1").await.unwrap();
        // moved_to_dlq_at is "now" (not back-dated) — a 30-day cutoff would skip it.

        let (deleted, _has_more) = qm.purge_dlq(0, 500).await.unwrap();
        assert_eq!(deleted, 1, "retention 0 must purge a just-added DLQ entry");

        let (entries, total) = qm.list_dlq(None, None, 10, 0).await.unwrap();
        assert_eq!(total, 0);
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn test_list_dlq() {
        let qm = setup().await;
        insert_failed_item(&qm, "q-list-1", "[permanent_data] err1").await;
        insert_failed_item(&qm, "q-list-2", "[permanent_gone] err2").await;
        qm.move_to_dlq("q-list-1").await.unwrap();
        qm.move_to_dlq("q-list-2").await.unwrap();

        let (entries, total) = qm.list_dlq(None, None, 10, 0).await.unwrap();
        assert_eq!(total, 2);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_extract_error_category() {
        assert_eq!(
            extract_error_category("[permanent_data] bad"),
            "permanent_data"
        );
        assert_eq!(
            extract_error_category("[permanent_gone] deleted"),
            "permanent_gone"
        );
        assert_eq!(
            extract_error_category("[transient_infrastructure] timeout"),
            "transient_infrastructure"
        );
        assert_eq!(
            extract_error_category("exhausted after 5 retries"),
            "permanent_exhausted"
        );
        assert_eq!(extract_error_category("random error"), "unknown");
    }
}
