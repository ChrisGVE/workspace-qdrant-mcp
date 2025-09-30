//! Queue Operations Module
//!
//! Provides Rust interface to the ingestion queue system with full compatibility
//! with Python queue client operations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, error, info, warn};

/// Queue operation errors
#[derive(Error, Debug)]
pub enum QueueError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid priority: {0}")]
    InvalidPriority(i32),

    #[error("Invalid operation type: {0}")]
    InvalidOperation(String),

    #[error("Queue item not found: {0}")]
    NotFound(String),
}

/// Result type for queue operations
pub type QueueResult<T> = Result<T, QueueError>;

/// Queue operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueueOperation {
    Ingest,
    Update,
    Delete,
}

impl QueueOperation {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueOperation::Ingest => "ingest",
            QueueOperation::Update => "update",
            QueueOperation::Delete => "delete",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, QueueError> {
        match s {
            "ingest" => Ok(QueueOperation::Ingest),
            "update" => Ok(QueueOperation::Update),
            "delete" => Ok(QueueOperation::Delete),
            _ => Err(QueueError::InvalidOperation(s.to_string())),
        }
    }
}

/// Queue item representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
    pub file_absolute_path: String,
    pub collection_name: String,
    pub tenant_id: String,
    pub branch: String,
    pub operation: QueueOperation,
    pub priority: i32,
    pub queued_timestamp: DateTime<Utc>,
    pub retry_count: i32,
    pub retry_from: Option<String>,
    pub error_message_id: Option<i64>,
}

impl QueueItem {
    /// Validate priority is in range 0-10
    pub fn validate_priority(&self) -> QueueResult<()> {
        if self.priority < 0 || self.priority > 10 {
            return Err(QueueError::InvalidPriority(self.priority));
        }
        Ok(())
    }
}

/// Collection type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CollectionType {
    NonWatched,
    WatchedDynamic,
    WatchedCumulative,
    Project,
}

impl CollectionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            CollectionType::NonWatched => "non-watched",
            CollectionType::WatchedDynamic => "watched-dynamic",
            CollectionType::WatchedCumulative => "watched-cumulative",
            CollectionType::Project => "project",
        }
    }
}

/// Collection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub collection_name: String,
    pub collection_type: CollectionType,
    pub created_timestamp: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub configuration: HashMap<String, serde_json::Value>,
    pub tenant_id: String,
    pub branch: String,
}

/// Error message record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub id: Option<i64>,
    pub error_type: String,
    pub error_message: String,
    pub error_details: Option<HashMap<String, serde_json::Value>>,
    pub occurred_timestamp: DateTime<Utc>,
    pub file_path: Option<String>,
    pub collection_name: Option<String>,
    pub retry_count: i32,
}

/// Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub total_items: i64,
    pub urgent_items: i64,
    pub high_items: i64,
    pub normal_items: i64,
    pub low_items: i64,
    pub retry_items: i64,
    pub error_items: i64,
    pub unique_collections: i64,
    pub oldest_item: Option<DateTime<Utc>>,
    pub newest_item: Option<DateTime<Utc>>,
}

/// Queue manager for Rust daemon operations
pub struct QueueManager {
    pool: SqlitePool,
}

impl QueueManager {
    /// Create a new queue manager with existing connection pool
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Enqueue a file for processing
    pub async fn enqueue_file(
        &self,
        file_path: &str,
        collection: &str,
        tenant_id: &str,
        branch: &str,
        operation: QueueOperation,
        priority: i32,
        retry_from: Option<&str>,
    ) -> QueueResult<String> {
        // Validate priority
        if priority < 0 || priority > 10 {
            return Err(QueueError::InvalidPriority(priority));
        }

        let query = r#"
            INSERT INTO ingestion_queue (
                file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, retry_from
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#;

        sqlx::query(query)
            .bind(file_path)
            .bind(collection)
            .bind(tenant_id)
            .bind(branch)
            .bind(operation.as_str())
            .bind(priority)
            .bind(retry_from)
            .execute(&self.pool)
            .await?;

        debug!(
            "Enqueued file: {} (collection={}, priority={})",
            file_path, collection, priority
        );

        Ok(file_path.to_string())
    }

    /// Dequeue a batch of items for processing
    pub async fn dequeue_batch(
        &self,
        batch_size: i32,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> QueueResult<Vec<QueueItem>> {
        let mut query = String::from(
            r#"
            SELECT
                file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, queued_timestamp, retry_count,
                retry_from, error_message_id
            FROM ingestion_queue
            "#,
        );

        let mut conditions = Vec::new();

        if tenant_id.is_some() {
            conditions.push("tenant_id = ?");
        }

        if branch.is_some() {
            conditions.push("branch = ?");
        }

        if !conditions.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&conditions.join(" AND "));
        }

        query.push_str(" ORDER BY priority DESC, queued_timestamp ASC LIMIT ?");

        let mut query_builder = sqlx::query(&query);

        if let Some(tid) = tenant_id {
            query_builder = query_builder.bind(tid);
        }

        if let Some(br) = branch {
            query_builder = query_builder.bind(br);
        }

        query_builder = query_builder.bind(batch_size);

        let rows = query_builder.fetch_all(&self.pool).await?;

        let mut items = Vec::new();

        for row in rows {
            let operation_str: String = row.try_get("operation")?;
            let operation = QueueOperation::from_str(&operation_str)?;

            items.push(QueueItem {
                file_absolute_path: row.try_get("file_absolute_path")?,
                collection_name: row.try_get("collection_name")?,
                tenant_id: row.try_get("tenant_id")?,
                branch: row.try_get("branch")?,
                operation,
                priority: row.try_get("priority")?,
                queued_timestamp: row.try_get("queued_timestamp")?,
                retry_count: row.try_get("retry_count")?,
                retry_from: row.try_get("retry_from")?,
                error_message_id: row.try_get("error_message_id")?,
            });
        }

        debug!("Dequeued {} items from queue", items.len());

        Ok(items)
    }

    /// Update priority for a queued file
    pub async fn update_priority(&self, file_path: &str, new_priority: i32) -> QueueResult<bool> {
        // Validate priority
        if new_priority < 0 || new_priority > 10 {
            return Err(QueueError::InvalidPriority(new_priority));
        }

        let query = r#"
            UPDATE ingestion_queue
            SET priority = ?1
            WHERE file_absolute_path = ?2
        "#;

        let result = sqlx::query(query)
            .bind(new_priority)
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        let updated = result.rows_affected() > 0;

        if updated {
            debug!("Updated priority for {}: {}", file_path, new_priority);
        } else {
            warn!("File not found in queue: {}", file_path);
        }

        Ok(updated)
    }

    /// Mark a file as completed and remove from queue
    pub async fn mark_complete(&self, file_path: &str) -> QueueResult<bool> {
        let query = "DELETE FROM ingestion_queue WHERE file_absolute_path = ?1";

        let result = sqlx::query(query)
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() > 0;

        if deleted {
            debug!("Marked complete and removed from queue: {}", file_path);
        } else {
            warn!("File not found in queue: {}", file_path);
        }

        Ok(deleted)
    }

    /// Mark a file as having an error
    pub async fn mark_error(
        &self,
        file_path: &str,
        error_type: &str,
        error_message: &str,
        error_details: Option<&HashMap<String, serde_json::Value>>,
        max_retries: i32,
    ) -> QueueResult<(bool, i64)> {
        // Start transaction
        let mut tx = self.pool.begin().await?;

        // Insert error message
        let error_details_json = if let Some(details) = error_details {
            Some(serde_json::to_string(details)?)
        } else {
            None
        };

        let error_query = r#"
            INSERT INTO messages (
                error_type, error_message, error_details,
                file_path, collection_name
            ) VALUES (?1, ?2, ?3, ?4, (
                SELECT collection_name FROM ingestion_queue
                WHERE file_absolute_path = ?5
            ))
        "#;

        let result = sqlx::query(error_query)
            .bind(error_type)
            .bind(error_message)
            .bind(error_details_json)
            .bind(file_path)
            .bind(file_path)
            .execute(&mut *tx)
            .await?;

        let error_message_id = result.last_insert_rowid();

        // Get current retry count
        let row = sqlx::query("SELECT retry_count FROM ingestion_queue WHERE file_absolute_path = ?1")
            .bind(file_path)
            .fetch_optional(&mut *tx)
            .await?;

        if let Some(row) = row {
            let current_retry_count: i32 = row.try_get("retry_count")?;
            let new_retry_count = current_retry_count + 1;

            if new_retry_count >= max_retries {
                // Max retries reached, remove from queue
                sqlx::query("DELETE FROM ingestion_queue WHERE file_absolute_path = ?1")
                    .bind(file_path)
                    .execute(&mut *tx)
                    .await?;

                warn!(
                    "Max retries ({}) reached for {}, removing from queue",
                    max_retries, file_path
                );

                tx.commit().await?;
                Ok((false, error_message_id))
            } else {
                // Update retry count and link error
                sqlx::query(
                    r#"
                    UPDATE ingestion_queue
                    SET retry_count = ?1, error_message_id = ?2
                    WHERE file_absolute_path = ?3
                    "#,
                )
                .bind(new_retry_count)
                .bind(error_message_id)
                .bind(file_path)
                .execute(&mut *tx)
                .await?;

                debug!(
                    "Updated error for {}: retry {}/{}",
                    file_path, new_retry_count, max_retries
                );

                tx.commit().await?;
                Ok((true, error_message_id))
            }
        } else {
            warn!("File not found in queue: {}", file_path);
            tx.commit().await?;
            Ok((false, error_message_id))
        }
    }

    /// Get queue statistics
    pub async fn get_queue_stats(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> QueueResult<QueueStats> {
        let mut query = String::from(
            r#"
            SELECT
                COUNT(*) as total_items,
                SUM(CASE WHEN priority >= 8 THEN 1 ELSE 0 END) as urgent_items,
                SUM(CASE WHEN priority >= 5 AND priority < 8 THEN 1 ELSE 0 END) as high_items,
                SUM(CASE WHEN priority >= 3 AND priority < 5 THEN 1 ELSE 0 END) as normal_items,
                SUM(CASE WHEN priority < 3 THEN 1 ELSE 0 END) as low_items,
                SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as retry_items,
                SUM(CASE WHEN error_message_id IS NOT NULL THEN 1 ELSE 0 END) as error_items,
                COUNT(DISTINCT collection_name) as unique_collections,
                MIN(queued_timestamp) as oldest_item,
                MAX(queued_timestamp) as newest_item
            FROM ingestion_queue
            "#,
        );

        let mut conditions = Vec::new();

        if tenant_id.is_some() {
            conditions.push("tenant_id = ?");
        }

        if branch.is_some() {
            conditions.push("branch = ?");
        }

        if !conditions.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&conditions.join(" AND "));
        }

        let mut query_builder = sqlx::query(&query);

        if let Some(tid) = tenant_id {
            query_builder = query_builder.bind(tid);
        }

        if let Some(br) = branch {
            query_builder = query_builder.bind(br);
        }

        let row = query_builder.fetch_one(&self.pool).await?;

        Ok(QueueStats {
            total_items: row.try_get("total_items")?,
            urgent_items: row.try_get("urgent_items")?,
            high_items: row.try_get("high_items")?,
            normal_items: row.try_get("normal_items")?,
            low_items: row.try_get("low_items")?,
            retry_items: row.try_get("retry_items")?,
            error_items: row.try_get("error_items")?,
            unique_collections: row.try_get("unique_collections")?,
            oldest_item: row.try_get("oldest_item")?,
            newest_item: row.try_get("newest_item")?,
        })
    }

    /// Register or update collection metadata
    pub async fn register_collection(
        &self,
        collection_name: &str,
        collection_type: CollectionType,
        configuration: &HashMap<String, serde_json::Value>,
        tenant_id: &str,
        branch: &str,
    ) -> QueueResult<()> {
        let config_json = serde_json::to_string(configuration)?;

        let query = r#"
            INSERT INTO collection_metadata (
                collection_name, collection_type, configuration, tenant_id, branch
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(collection_name) DO UPDATE SET
                collection_type = excluded.collection_type,
                configuration = excluded.configuration,
                last_updated = CURRENT_TIMESTAMP
        "#;

        sqlx::query(query)
            .bind(collection_name)
            .bind(collection_type.as_str())
            .bind(config_json)
            .bind(tenant_id)
            .bind(branch)
            .execute(&self.pool)
            .await?;

        debug!(
            "Registered collection: {} (type={})",
            collection_name,
            collection_type.as_str()
        );

        Ok(())
    }

    /// Get collection metadata
    pub async fn get_collection_info(
        &self,
        collection_name: &str,
    ) -> QueueResult<Option<CollectionMetadata>> {
        let query = r#"
            SELECT
                collection_name, collection_type, created_timestamp,
                last_updated, configuration, tenant_id, branch
            FROM collection_metadata
            WHERE collection_name = ?1
        "#;

        let row = sqlx::query(query)
            .bind(collection_name)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            let collection_type_str: String = row.try_get("collection_type")?;
            let collection_type = match collection_type_str.as_str() {
                "non-watched" => CollectionType::NonWatched,
                "watched-dynamic" => CollectionType::WatchedDynamic,
                "watched-cumulative" => CollectionType::WatchedCumulative,
                "project" => CollectionType::Project,
                _ => CollectionType::NonWatched,
            };

            let config_json: String = row.try_get("configuration")?;
            let configuration: HashMap<String, serde_json::Value> =
                serde_json::from_str(&config_json)?;

            Ok(Some(CollectionMetadata {
                collection_name: row.try_get("collection_name")?,
                collection_type,
                created_timestamp: row.try_get("created_timestamp")?,
                last_updated: row.try_get("last_updated")?,
                configuration,
                tenant_id: row.try_get("tenant_id")?,
                branch: row.try_get("branch")?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Clear items from the queue
    pub async fn clear_queue(
        &self,
        collection: Option<&str>,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> QueueResult<u64> {
        let mut query = String::from("DELETE FROM ingestion_queue");
        let mut conditions = Vec::new();

        if collection.is_some() {
            conditions.push("collection_name = ?");
        }

        if tenant_id.is_some() {
            conditions.push("tenant_id = ?");
        }

        if branch.is_some() {
            conditions.push("branch = ?");
        }

        if !conditions.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&conditions.join(" AND "));
        }

        let mut query_builder = sqlx::query(&query);

        if let Some(col) = collection {
            query_builder = query_builder.bind(col);
        }

        if let Some(tid) = tenant_id {
            query_builder = query_builder.bind(tid);
        }

        if let Some(br) = branch {
            query_builder = query_builder.bind(br);
        }

        let result = query_builder.execute(&self.pool).await?;

        let deleted_count = result.rows_affected();
        info!("Cleared {} items from queue", deleted_count);

        Ok(deleted_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_config::QueueConnectionConfig;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_queue.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schema
        sqlx::query(include_str!("../../../../../python/common/core/queue_schema.sql"))
            .execute(&pool)
            .await
            .unwrap();

        let manager = QueueManager::new(pool);

        // Enqueue a file
        manager
            .enqueue_file(
                "/test/file.txt",
                "test-collection",
                "default",
                "main",
                QueueOperation::Ingest,
                5,
                None,
            )
            .await
            .unwrap();

        // Dequeue
        let items = manager.dequeue_batch(10, None, None).await.unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].file_absolute_path, "/test/file.txt");
        assert_eq!(items[0].priority, 5);
    }

    #[tokio::test]
    async fn test_priority_validation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_priority.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);

        // Invalid priority should fail
        let result = manager
            .enqueue_file(
                "/test/file.txt",
                "test-collection",
                "default",
                "main",
                QueueOperation::Ingest,
                11, // Invalid: > 10
                None,
            )
            .await;

        assert!(result.is_err());
    }
}
