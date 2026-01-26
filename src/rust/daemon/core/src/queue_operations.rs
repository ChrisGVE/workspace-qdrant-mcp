//! Queue Operations Module
//!
//! Provides Rust interface to the ingestion queue system with full compatibility
//! with Python queue client operations.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, error, info, warn};

use crate::unified_queue_schema::{
    ItemType, QueueOperation as UnifiedOp, QueueStatus,
    UnifiedQueueItem, UnifiedQueueStats,
    generate_unified_idempotency_key,
};

// Import MissingTool from queue_processor module
use crate::queue_types::MissingTool;
use crate::metrics::METRICS;

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

/// Classify collection type based on naming patterns
/// Matches Python CollectionTypeClassifier logic
fn classify_collection_type(collection_name: &str) -> Option<String> {
    // Global collections
    const GLOBAL_COLLECTIONS: &[&str] = &[
        "algorithms",
        "codebase",
        "context",
        "documents",
        "knowledge",
        "memory",
        "projects",
        "workspace",
    ];

    if collection_name.is_empty() {
        return Some("unknown".to_string());
    }

    // System collections: __ prefix
    if collection_name.starts_with("__") {
        return Some("system".to_string());
    }

    // Library collections: _ prefix (but not __)
    if collection_name.starts_with('_') && !collection_name.starts_with("__") {
        return Some("library".to_string());
    }

    // Global collections: predefined names
    if GLOBAL_COLLECTIONS.contains(&collection_name) {
        return Some("global".to_string());
    }

    // Project collections: {project}-{suffix} pattern
    if collection_name.contains('-') {
        // Simple pattern match: contains dash
        return Some("project".to_string());
    }

    Some("unknown".to_string())
}

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
    pub collection_type: Option<String>,  // Collection type: system, library, project, global
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

/// Missing metadata queue item representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingMetadataItem {
    pub queue_id: String,
    pub file_absolute_path: String,
    pub collection_name: String,
    pub tenant_id: String,
    pub branch: String,
    pub operation: QueueOperation,
    pub priority: i32,
    pub missing_tools: Vec<MissingTool>,
    pub queued_timestamp: DateTime<Utc>,
    pub retry_count: i32,
    pub last_check_timestamp: Option<DateTime<Utc>>,
    pub metadata: Option<String>,
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

/// Queue load level for adaptive throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueLoadLevel {
    /// Normal load - no throttling needed
    Normal,
    /// High load - moderate throttling recommended
    High,
    /// Critical load - aggressive throttling required
    Critical,
}

impl QueueLoadLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueLoadLevel::Normal => "normal",
            QueueLoadLevel::High => "high",
            QueueLoadLevel::Critical => "critical",
        }
    }
}

/// Queue throttling summary for adaptive rate control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueThrottlingSummary {
    /// Total items across all collections
    pub total_depth: i64,
    /// Per-collection queue depths
    pub by_collection: HashMap<String, i64>,
    /// Current load level
    pub load_level: QueueLoadLevel,
    /// Suggested polling interval multiplier (1.0-4.0)
    pub throttle_factor: f64,
    /// Threshold for high load
    pub high_threshold: i64,
    /// Threshold for critical load
    pub critical_threshold: i64,
}

#[derive(Clone)]
/// Queue manager for Rust daemon operations
pub struct QueueManager {
    pool: SqlitePool,
}

impl QueueManager {
    /// Create a new queue manager with existing connection pool
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Get reference to the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Initialize the missing_metadata_queue table
    pub async fn init_missing_metadata_queue(&self) -> QueueResult<()> {
        let schema = include_str!("../../../../python/common/core/missing_metadata_queue_schema.sql");
        sqlx::query(schema).execute(&self.pool).await?;
        debug!("Missing metadata queue table initialized");
        Ok(())
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
        if !(0..=10).contains(&priority) {
            return Err(QueueError::InvalidPriority(priority));
        }

        // Detect collection type from collection name pattern
        let collection_type = classify_collection_type(collection);

        let query = r#"
            INSERT INTO ingestion_queue (
                file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, retry_from, collection_type
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#;

        sqlx::query(query)
            .bind(file_path)
            .bind(collection)
            .bind(tenant_id)
            .bind(branch)
            .bind(operation.as_str())
            .bind(priority)
            .bind(retry_from)
            .bind(collection_type.as_deref())
            .execute(&self.pool)
            .await?;

        debug!(
            "Enqueued file: {} (collection={}, priority={}, type={:?})",
            file_path, collection, priority, collection_type
        );

        // Update queue depth metric (Task 412.7)
        // Note: We use the priority and collection for metric labels
        let priority_str = match priority {
            1 => "high",
            3 => "normal",
            5 => "low",
            _ => "other",
        };
        // Get actual queue depth after insert
        if let Ok(depth) = self.get_queue_depth(Some(tenant_id), None).await {
            METRICS.set_queue_depth(priority_str, collection, depth);
        }

        Ok(file_path.to_string())
    }

    /// Dequeue a batch of items for processing
    ///
    /// Filters out items with future retry_from timestamps to implement exponential backoff.
    pub async fn dequeue_batch(
        &self,
        batch_size: i32,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> QueueResult<Vec<QueueItem>> {
        let now = Utc::now().to_rfc3339();

        let mut query = String::from(
            r#"
            SELECT
                file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, queued_timestamp, retry_count,
                retry_from, error_message_id, collection_type
            FROM ingestion_queue
            "#,
        );

        let mut conditions = Vec::new();

        // Add retry_from filter to skip items scheduled for future retry
        conditions.push("(retry_from IS NULL OR retry_from <= ?)");

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

        // Bind the current timestamp for retry_from comparison
        query_builder = query_builder.bind(&now);

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
                collection_type: row.try_get("collection_type").ok(),  // May be None for legacy items
            });
        }

        debug!("Dequeued {} items from queue", items.len());

        Ok(items)
    }

    /// Update retry_from timestamp for exponential backoff
    ///
    /// Sets when an item should be retried after a processing failure.
    pub async fn update_retry_from(
        &self,
        file_path: &str,
        retry_from: DateTime<Utc>,
        retry_count: i32,
    ) -> QueueResult<bool> {
        let query = r#"
            UPDATE ingestion_queue
            SET retry_from = ?1,
                retry_count = ?2
            WHERE file_absolute_path = ?3
        "#;

        let result = sqlx::query(query)
            .bind(retry_from.to_rfc3339())
            .bind(retry_count)
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        let updated = result.rows_affected() > 0;

        if updated {
            debug!(
                "Updated retry_from for {}: {} (retry_count={})",
                file_path,
                retry_from.to_rfc3339(),
                retry_count
            );
        } else {
            warn!("File not found in queue: {}", file_path);
        }

        Ok(updated)
    }

    /// Mark a file as failed (max retries exceeded)
    pub async fn mark_failed(
        &self,
        file_path: &str,
        error_message: &str,
    ) -> QueueResult<bool> {
        // Start transaction
        let mut tx = self.pool.begin().await?;

        // Insert error message
        let error_query = r#"
            INSERT INTO messages (
                error_type, error_message, file_path, collection_name
            ) VALUES ('MAX_RETRIES_EXCEEDED', ?1, ?2, (
                SELECT collection_name FROM ingestion_queue
                WHERE file_absolute_path = ?3
            ))
        "#;

        sqlx::query(error_query)
            .bind(error_message)
            .bind(file_path)
            .bind(file_path)
            .execute(&mut *tx)
            .await?;

        // Remove from queue
        let delete_query = "DELETE FROM ingestion_queue WHERE file_absolute_path = ?1";
        let result = sqlx::query(delete_query)
            .bind(file_path)
            .execute(&mut *tx)
            .await?;

        let deleted = result.rows_affected() > 0;

        if deleted {
            // Record failure metric (Task 412.7)
            METRICS.queue_item_processed("normal", "failure", 0.0);
            METRICS.ingestion_error("max_retries_exceeded");
            warn!("Marked as failed and removed from queue: {}", file_path);
        } else {
            warn!("File not found in queue when marking failed: {}", file_path);
        }

        tx.commit().await?;
        Ok(deleted)
    }

    /// Update priority for a queued file
    pub async fn update_priority(&self, file_path: &str, new_priority: i32) -> QueueResult<bool> {
        // Validate priority
        if !(0..=10).contains(&new_priority) {
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
    ///
    /// Records queue_item_processed metric with "success" status.
    /// For accurate processing time metrics, use `mark_complete_with_duration` instead.
    pub async fn mark_complete(&self, file_path: &str) -> QueueResult<bool> {
        let query = "DELETE FROM ingestion_queue WHERE file_absolute_path = ?1";

        let result = sqlx::query(query)
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() > 0;

        if deleted {
            // Record successful processing metric (Task 412.7)
            // Note: Processing time is 0.0 because we don't track it here
            // Use mark_complete_with_duration for accurate timing
            METRICS.queue_item_processed("normal", "success", 0.0);
            debug!("Marked complete and removed from queue: {}", file_path);
        } else {
            warn!("File not found in queue: {}", file_path);
        }

        Ok(deleted)
    }

    /// Mark a file as completed with processing duration
    ///
    /// Records queue_item_processed metric with actual processing time.
    pub async fn mark_complete_with_duration(
        &self,
        file_path: &str,
        priority: i32,
        processing_time_secs: f64,
    ) -> QueueResult<bool> {
        let query = "DELETE FROM ingestion_queue WHERE file_absolute_path = ?1";

        let result = sqlx::query(query)
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() > 0;

        if deleted {
            // Record successful processing metric (Task 412.7)
            let priority_str = match priority {
                1 => "high",
                3 => "normal",
                5 => "low",
                _ => "other",
            };
            METRICS.queue_item_processed(priority_str, "success", processing_time_secs);
            debug!(
                "Marked complete and removed from queue: {} (processing_time={:.3}s)",
                file_path, processing_time_secs
            );
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

                // Record failure metrics (Task 412.7)
                METRICS.queue_item_processed("normal", "failure", 0.0);
                METRICS.ingestion_error(error_type);

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

                // Record error metric - item will be retried (Task 412.7)
                METRICS.ingestion_error(error_type);

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

    /// Move item to missing_metadata_queue
    pub async fn move_to_missing_metadata_queue(
        &self,
        item: &QueueItem,
        missing_tools: &[MissingTool],
    ) -> QueueResult<String> {
        // Generate queue_id using hash of file path, tenant_id, and branch
        let queue_id = format!(
            "{:x}",
            md5::compute(format!(
                "{}||{}||{}||{}",
                item.file_absolute_path,
                item.tenant_id,
                item.branch,
                Utc::now().timestamp_millis()
            ))
        );

        // Serialize missing_tools to JSON
        let missing_tools_json = serde_json::to_string(missing_tools)?;

        // Build metadata JSON if needed (placeholder for now)
        let metadata_json = None::<String>;

        // Insert into missing_metadata_queue (using INSERT OR REPLACE for UNIQUE constraint)
        let query = r#"
            INSERT OR REPLACE INTO missing_metadata_queue (
                queue_id, file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, missing_tools, queued_timestamp, retry_count, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
        "#;

        sqlx::query(query)
            .bind(&queue_id)
            .bind(&item.file_absolute_path)
            .bind(&item.collection_name)
            .bind(&item.tenant_id)
            .bind(&item.branch)
            .bind(item.operation.as_str())
            .bind(item.priority)
            .bind(&missing_tools_json)
            .bind(item.queued_timestamp.to_rfc3339())
            .bind(item.retry_count)
            .bind(metadata_json)
            .execute(&self.pool)
            .await?;

        // Remove from main ingestion_queue
        self.mark_complete(&item.file_absolute_path).await?;

        info!(
            "Moved to missing_metadata_queue: {} (missing tools: {})",
            item.file_absolute_path,
            missing_tools
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        Ok(queue_id)
    }

    /// Get items from missing_metadata_queue
    pub async fn get_missing_metadata_items(
        &self,
        limit: i32,
    ) -> QueueResult<Vec<MissingMetadataItem>> {
        let query = r#"
            SELECT
                queue_id, file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, missing_tools, queued_timestamp,
                retry_count, last_check_timestamp, metadata
            FROM missing_metadata_queue
            ORDER BY priority DESC, queued_timestamp ASC
            LIMIT ?1
        "#;

        let rows = sqlx::query(query)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        let mut items = Vec::new();

        for row in rows {
            let operation_str: String = row.try_get("operation")?;
            let operation = QueueOperation::from_str(&operation_str)?;

            let missing_tools_json: String = row.try_get("missing_tools")?;
            let missing_tools: Vec<MissingTool> = serde_json::from_str(&missing_tools_json)?;

            let queued_timestamp_str: String = row.try_get("queued_timestamp")?;
            let queued_timestamp = DateTime::parse_from_rfc3339(&queued_timestamp_str)
                .map_err(|e| {
                    QueueError::Database(sqlx::Error::Decode(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to parse timestamp: {}", e),
                    ))))
                })?
                .with_timezone(&Utc);

            let last_check_timestamp: Option<String> = row.try_get("last_check_timestamp")?;
            let last_check_timestamp = last_check_timestamp
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            items.push(MissingMetadataItem {
                queue_id: row.try_get("queue_id")?,
                file_absolute_path: row.try_get("file_absolute_path")?,
                collection_name: row.try_get("collection_name")?,
                tenant_id: row.try_get("tenant_id")?,
                branch: row.try_get("branch")?,
                operation,
                priority: row.try_get("priority")?,
                missing_tools,
                queued_timestamp,
                retry_count: row.try_get("retry_count")?,
                last_check_timestamp,
                metadata: row.try_get("metadata")?,
            });
        }

        debug!(
            "Retrieved {} items from missing_metadata_queue",
            items.len()
        );

        Ok(items)
    }

    /// Retry missing metadata item by moving back to ingestion_queue
    pub async fn retry_missing_metadata_item(&self, queue_id: &str) -> QueueResult<bool> {
        // Start transaction
        let mut tx = self.pool.begin().await?;

        // Get the item from missing_metadata_queue
        let query = r#"
            SELECT
                file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, retry_count
            FROM missing_metadata_queue
            WHERE queue_id = ?1
        "#;

        let row = sqlx::query(query)
            .bind(queue_id)
            .fetch_optional(&mut *tx)
            .await?;

        if let Some(row) = row {
            let file_path: String = row.try_get("file_absolute_path")?;
            let collection: String = row.try_get("collection_name")?;
            let tenant_id: String = row.try_get("tenant_id")?;
            let branch: String = row.try_get("branch")?;
            let operation_str: String = row.try_get("operation")?;
            let operation = QueueOperation::from_str(&operation_str)?;
            let priority: i32 = row.try_get("priority")?;
            let retry_count: i32 = row.try_get("retry_count")?;

            // Insert back into ingestion_queue (with incremented retry_count)
            let insert_query = r#"
                INSERT OR REPLACE INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id, branch,
                    operation, priority, retry_count
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#;

            sqlx::query(insert_query)
                .bind(&file_path)
                .bind(&collection)
                .bind(&tenant_id)
                .bind(&branch)
                .bind(operation.as_str())
                .bind(priority)
                .bind(retry_count + 1)
                .execute(&mut *tx)
                .await?;

            // Update last_check_timestamp in missing_metadata_queue
            let update_query = r#"
                UPDATE missing_metadata_queue
                SET last_check_timestamp = ?1, retry_count = retry_count + 1
                WHERE queue_id = ?2
            "#;

            sqlx::query(update_query)
                .bind(Utc::now().to_rfc3339())
                .bind(queue_id)
                .execute(&mut *tx)
                .await?;

            // Don't delete from missing_metadata_queue yet - let processing decide if successful
            tx.commit().await?;

            info!("Retrying item from missing_metadata_queue: {}", file_path);
            Ok(true)
        } else {
            warn!("Queue item not found in missing_metadata_queue: {}", queue_id);
            tx.commit().await?;
            Ok(false)
        }
    }

    /// Remove item from missing_metadata_queue
    pub async fn remove_from_missing_metadata_queue(&self, queue_id: &str) -> QueueResult<bool> {
        let query = "DELETE FROM missing_metadata_queue WHERE queue_id = ?1";

        let result = sqlx::query(query)
            .bind(queue_id)
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() > 0;

        if deleted {
            debug!("Removed from missing_metadata_queue: {}", queue_id);
        } else {
            warn!("Queue item not found in missing_metadata_queue: {}", queue_id);
        }

        Ok(deleted)
    }

    /// Get missing metadata queue depth
    pub async fn get_missing_metadata_queue_depth(&self) -> QueueResult<i64> {
        let query = "SELECT COUNT(*) FROM missing_metadata_queue";
        let count: i64 = sqlx::query_scalar(query).fetch_one(&self.pool).await?;
        Ok(count)
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

    /// Enqueue multiple files as a batch with priority calculation
    pub async fn enqueue_batch(
        &self,
        items: Vec<QueueItem>,
        max_queue_depth: Option<i64>,
        overflow_strategy: &str,
    ) -> QueueResult<(i64, Vec<String>)> {
        if items.is_empty() {
            return Ok((0, Vec::new()));
        }

        // Validate all items first
        for item in &items {
            if !(0..=10).contains(&item.priority) {
                return Err(QueueError::InvalidPriority(item.priority));
            }
        }

        // Check queue depth if limit specified
        if let Some(max_depth) = max_queue_depth {
            let current_depth = self.get_queue_depth(None, None).await?;

            if current_depth + items.len() as i64 > max_depth {
                if overflow_strategy == "reject" {
                    return Err(QueueError::InvalidOperation(format!(
                        "Queue depth limit ({}) would be exceeded. Current: {}, Adding: {}",
                        max_depth,
                        current_depth,
                        items.len()
                    )));
                } else if overflow_strategy == "replace_lowest" {
                    // Remove lowest priority items to make space
                    let items_to_remove = (current_depth + items.len() as i64) - max_depth;

                    let delete_query = format!(
                        "DELETE FROM ingestion_queue WHERE file_absolute_path IN \
                         (SELECT file_absolute_path FROM ingestion_queue \
                          ORDER BY priority ASC, queued_timestamp DESC LIMIT {})",
                        items_to_remove
                    );

                    sqlx::query(&delete_query).execute(&self.pool).await?;

                    info!(
                        "Removed {} lowest priority items due to queue depth limit",
                        items_to_remove
                    );
                }
            }
        }

        // Batch insert
        let mut successful = 0i64;
        let mut failed = Vec::new();

        for item in items {
            // Use existing collection_type or detect it
            let collection_type = item.collection_type.clone()
                .or_else(|| classify_collection_type(&item.collection_name));

            let result = sqlx::query(
                r#"
                INSERT OR REPLACE INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id, branch,
                    operation, priority, queued_timestamp, collection_type
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, CURRENT_TIMESTAMP, ?7)
                "#,
            )
            .bind(&item.file_absolute_path)
            .bind(&item.collection_name)
            .bind(&item.tenant_id)
            .bind(&item.branch)
            .bind(item.operation.as_str())
            .bind(item.priority)
            .bind(collection_type.as_deref())
            .execute(&self.pool)
            .await;

            match result {
                Ok(_) => successful += 1,
                Err(e) => {
                    error!("Failed to enqueue {}: {}", item.file_absolute_path, e);
                    failed.push(item.file_absolute_path);
                }
            }
        }

        info!(
            "Batch enqueue completed: {} successful, {} failed",
            successful,
            failed.len()
        );

        Ok((successful, failed))
    }

    /// Purge completed items based on retention policy
    pub async fn purge_completed_items(
        &self,
        retention_hours: i32,
        _tenant_id: Option<&str>,
        _branch: Option<&str>,
    ) -> QueueResult<u64> {
        let query = format!(
            "DELETE FROM messages WHERE created_timestamp < datetime('now', '-{} hours')",
            retention_hours
        );

        let result = sqlx::query(&query).execute(&self.pool).await?;

        let purged_count = result.rows_affected();
        info!(
            "Purged {} messages older than {} hours",
            purged_count, retention_hours
        );

        Ok(purged_count)
    }

    /// Get current queue depth (item count)
    pub async fn get_queue_depth(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> QueueResult<i64> {
        let mut query = String::from("SELECT COUNT(*) FROM ingestion_queue");
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

        let mut query_builder = sqlx::query_scalar::<_, i64>(&query);

        if let Some(tid) = tenant_id {
            query_builder = query_builder.bind(tid);
        }

        if let Some(br) = branch {
            query_builder = query_builder.bind(br);
        }

        let count = query_builder.fetch_one(&self.pool).await?;

        Ok(count)
    }

    /// Get queue depth for a specific collection.
    ///
    /// Used for coordination between file watchers and queue processor
    /// to implement adaptive throttling when queue is overloaded.
    ///
    /// # Arguments
    /// * `collection` - Collection name to filter by
    ///
    /// # Returns
    /// Number of items in queue for the specified collection
    pub async fn get_queue_depth_by_collection(&self, collection: &str) -> QueueResult<i64> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ingestion_queue WHERE collection_name = ?",
        )
        .bind(collection)
        .fetch_one(&self.pool)
        .await?;

        Ok(count)
    }

    /// Get queue depth grouped by collection.
    ///
    /// Useful for monitoring and load balancing across collections.
    ///
    /// # Returns
    /// HashMap mapping collection names to their queue depths
    pub async fn get_queue_depth_all_collections(&self) -> QueueResult<HashMap<String, i64>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT collection_name, COUNT(*) as depth FROM ingestion_queue GROUP BY collection_name",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().collect())
    }

    /// Get queue summary for adaptive throttling decisions.
    ///
    /// Provides actionable information for file watchers to adjust
    /// their polling frequency based on queue load.
    ///
    /// # Arguments
    /// * `high_threshold` - Queue depth considered high load (default: 1000)
    /// * `critical_threshold` - Queue depth considered critical (default: 5000)
    ///
    /// # Returns
    /// QueueThrottlingSummary with load level and throttle factor
    pub async fn get_queue_summary_for_throttling(
        &self,
        high_threshold: i64,
        critical_threshold: i64,
    ) -> QueueResult<QueueThrottlingSummary> {
        let total_depth = self.get_queue_depth(None, None).await?;
        let by_collection = self.get_queue_depth_all_collections().await?;

        let (load_level, throttle_factor) = if total_depth >= critical_threshold {
            (QueueLoadLevel::Critical, 4.0)
        } else if total_depth >= high_threshold {
            (QueueLoadLevel::High, 2.0)
        } else {
            (QueueLoadLevel::Normal, 1.0)
        };

        Ok(QueueThrottlingSummary {
            total_depth,
            by_collection,
            load_level,
            throttle_factor,
            high_threshold,
            critical_threshold,
        })
    }

    // ========================================================================
    // Unified Queue Operations (Task 37.21-37.29)
    // ========================================================================

    /// Initialize the unified_queue table schema
    ///
    /// Creates the table and indexes if they don't exist.
    pub async fn init_unified_queue(&self) -> QueueResult<()> {
        use crate::unified_queue_schema::{CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL};

        // Create the table
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&self.pool)
            .await?;

        // Create all indexes
        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }

        debug!("Unified queue table initialized");
        Ok(())
    }

    /// Enqueue an item to the unified queue with idempotency support
    ///
    /// Returns (queue_id, is_new) where is_new indicates if this was a new insertion
    /// or if an existing item with the same idempotency key was found.
    ///
    /// # Arguments
    /// * `item_type` - Type of queue item (content, file, folder, etc.)
    /// * `op` - Operation to perform (ingest, update, delete, scan)
    /// * `tenant_id` - Project/tenant identifier
    /// * `collection` - Target Qdrant collection
    /// * `payload_json` - JSON payload with operation-specific data
    /// * `priority` - Processing priority (0-10, higher = more urgent)
    /// * `branch` - Git branch (default: main)
    /// * `metadata` - Optional additional metadata as JSON
    pub async fn enqueue_unified(
        &self,
        item_type: ItemType,
        op: UnifiedOp,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        priority: i32,
        branch: Option<&str>,
        metadata: Option<&str>,
    ) -> QueueResult<(String, bool)> {
        // Validate priority
        if !(0..=10).contains(&priority) {
            return Err(QueueError::InvalidPriority(priority));
        }

        // Generate idempotency key
        let idempotency_key = generate_unified_idempotency_key(
            item_type,
            op,
            tenant_id,
            collection,
            payload_json,
        ).map_err(|e| QueueError::InvalidOperation(e.to_string()))?;

        let branch = branch.unwrap_or("main");
        let metadata = metadata.unwrap_or("{}");

        // Use INSERT OR IGNORE to handle race conditions, then check what happened
        let insert_query = r#"
            INSERT OR IGNORE INTO unified_queue (
                item_type, op, tenant_id, collection, priority,
                branch, payload_json, metadata, idempotency_key
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#;

        let result = sqlx::query(insert_query)
            .bind(item_type.to_string())
            .bind(op.to_string())
            .bind(tenant_id)
            .bind(collection)
            .bind(priority)
            .bind(branch)
            .bind(payload_json)
            .bind(metadata)
            .bind(&idempotency_key)
            .execute(&self.pool)
            .await?;

        let is_new = result.rows_affected() > 0;

        // Get the queue_id (either newly inserted or existing)
        let queue_id: String = sqlx::query_scalar(
            "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1"
        )
            .bind(&idempotency_key)
            .fetch_one(&self.pool)
            .await?;

        if is_new {
            debug!(
                "Enqueued unified item: {} (type={}, op={}, collection={})",
                queue_id, item_type, op, collection
            );
        } else {
            // Update timestamp to show we tried to enqueue again
            sqlx::query(
                "UPDATE unified_queue SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE idempotency_key = ?1"
            )
                .bind(&idempotency_key)
                .execute(&self.pool)
                .await?;

            debug!(
                "Unified item already exists: {} (idempotency_key={})",
                queue_id, idempotency_key
            );
        }

        Ok((queue_id, is_new))
    }

    /// Dequeue a batch of items from the unified queue with lease-based locking
    ///
    /// Acquires a lease on the items to prevent concurrent processing.
    /// Items with expired leases are also considered for dequeuing.
    ///
    /// # Arguments
    /// * `batch_size` - Maximum number of items to dequeue
    /// * `worker_id` - Identifier for this worker (for lease tracking)
    /// * `lease_duration_secs` - How long to hold the lease (default: 300 seconds)
    /// * `tenant_id` - Optional filter by tenant
    /// * `item_type` - Optional filter by item type
    pub async fn dequeue_unified(
        &self,
        batch_size: i32,
        worker_id: &str,
        lease_duration_secs: Option<i64>,
        tenant_id: Option<&str>,
        item_type: Option<ItemType>,
    ) -> QueueResult<Vec<UnifiedQueueItem>> {
        let lease_duration = lease_duration_secs.unwrap_or(300);
        let lease_until = Utc::now() + ChronoDuration::seconds(lease_duration);
        let lease_until_str = lease_until.to_rfc3339();
        let now_str = Utc::now().to_rfc3339();

        // First, select the queue_ids to process using a simpler approach
        let queue_ids: Vec<String> = match (tenant_id, item_type) {
            (Some(tid), Some(itype)) => {
                sqlx::query_scalar::<_, String>(
                    r#"
                    SELECT queue_id FROM unified_queue
                    WHERE (status = 'pending' OR (status = 'in_progress' AND lease_until < ?1))
                    AND tenant_id = ?2
                    AND item_type = ?3
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?4
                    "#,
                )
                    .bind(&now_str)
                    .bind(tid)
                    .bind(itype.to_string())
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (Some(tid), None) => {
                sqlx::query_scalar::<_, String>(
                    r#"
                    SELECT queue_id FROM unified_queue
                    WHERE (status = 'pending' OR (status = 'in_progress' AND lease_until < ?1))
                    AND tenant_id = ?2
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?3
                    "#,
                )
                    .bind(&now_str)
                    .bind(tid)
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (None, Some(itype)) => {
                sqlx::query_scalar::<_, String>(
                    r#"
                    SELECT queue_id FROM unified_queue
                    WHERE (status = 'pending' OR (status = 'in_progress' AND lease_until < ?1))
                    AND item_type = ?2
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?3
                    "#,
                )
                    .bind(&now_str)
                    .bind(itype.to_string())
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (None, None) => {
                sqlx::query_scalar::<_, String>(
                    r#"
                    SELECT queue_id FROM unified_queue
                    WHERE (status = 'pending' OR (status = 'in_progress' AND lease_until < ?1))
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?2
                    "#,
                )
                    .bind(&now_str)
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
        };

        if queue_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Update the selected items to in_progress
        let placeholders: Vec<String> = (1..=queue_ids.len())
            .map(|i| format!("?{}", i + 2))
            .collect();
        let update_query = format!(
            r#"
            UPDATE unified_queue
            SET status = 'in_progress',
                worker_id = ?1,
                lease_until = ?2,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE queue_id IN ({})
            "#,
            placeholders.join(", ")
        );

        let mut update_builder = sqlx::query(&update_query)
            .bind(worker_id)
            .bind(&lease_until_str);

        for queue_id in &queue_ids {
            update_builder = update_builder.bind(queue_id);
        }

        update_builder.execute(&self.pool).await?;

        // Fetch the updated items
        let fetch_placeholders: Vec<String> = (1..=queue_ids.len())
            .map(|i| format!("?{}", i))
            .collect();
        let fetch_query = format!(
            "SELECT * FROM unified_queue WHERE queue_id IN ({})",
            fetch_placeholders.join(", ")
        );

        let mut fetch_builder = sqlx::query(&fetch_query);
        for queue_id in &queue_ids {
            fetch_builder = fetch_builder.bind(queue_id);
        }

        let rows = fetch_builder.fetch_all(&self.pool).await?;

        let mut items = Vec::new();
        for row in rows {
            let item_type_str: String = row.try_get("item_type")?;
            let op_str: String = row.try_get("op")?;
            let status_str: String = row.try_get("status")?;

            items.push(UnifiedQueueItem {
                queue_id: row.try_get("queue_id")?,
                idempotency_key: row.try_get("idempotency_key")?,
                item_type: ItemType::from_str(&item_type_str)
                    .ok_or_else(|| QueueError::InvalidOperation(item_type_str.clone()))?,
                op: UnifiedOp::from_str(&op_str)
                    .ok_or_else(|| QueueError::InvalidOperation(op_str.clone()))?,
                tenant_id: row.try_get("tenant_id")?,
                collection: row.try_get("collection")?,
                priority: row.try_get("priority")?,
                status: QueueStatus::from_str(&status_str)
                    .ok_or_else(|| QueueError::InvalidOperation(status_str.clone()))?,
                branch: row.try_get("branch")?,
                payload_json: row.try_get("payload_json")?,
                metadata: row.try_get("metadata")?,
                created_at: row.try_get("created_at")?,
                updated_at: row.try_get("updated_at")?,
                lease_until: row.try_get("lease_until")?,
                worker_id: row.try_get("worker_id")?,
                retry_count: row.try_get("retry_count")?,
                max_retries: row.try_get("max_retries")?,
                error_message: row.try_get("error_message")?,
                last_error_at: row.try_get("last_error_at")?,
            });
        }

        debug!(
            "Dequeued {} unified items for worker {}",
            items.len(),
            worker_id
        );

        Ok(items)
    }

    /// Mark a unified queue item as successfully completed
    ///
    /// Sets status to 'done' and clears the lease.
    pub async fn mark_unified_done(&self, queue_id: &str) -> QueueResult<bool> {
        let query = r#"
            UPDATE unified_queue
            SET status = 'done',
                lease_until = NULL,
                worker_id = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE queue_id = ?1 AND status = 'in_progress'
        "#;

        let result = sqlx::query(query)
            .bind(queue_id)
            .execute(&self.pool)
            .await?;

        let updated = result.rows_affected() > 0;

        if updated {
            debug!("Marked unified item as done: {}", queue_id);
            METRICS.queue_item_processed("unified", "success", 0.0);
        } else {
            warn!("Failed to mark unified item as done: {} (not in_progress)", queue_id);
        }

        Ok(updated)
    }

    /// Mark a unified queue item as failed
    ///
    /// If retries remain, increments retry_count and resets to pending.
    /// If max retries exceeded, sets status to 'failed'.
    ///
    /// Returns true if the item will be retried, false if permanently failed.
    pub async fn mark_unified_failed(
        &self,
        queue_id: &str,
        error_message: &str,
    ) -> QueueResult<bool> {
        // Get current retry state
        let row = sqlx::query(
            "SELECT retry_count, max_retries FROM unified_queue WHERE queue_id = ?1"
        )
            .bind(queue_id)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            let retry_count: i32 = row.try_get("retry_count")?;
            let max_retries: i32 = row.try_get("max_retries")?;
            let new_retry_count = retry_count + 1;

            if new_retry_count < max_retries {
                // Can retry - reset to pending with incremented retry count
                let query = r#"
                    UPDATE unified_queue
                    SET status = 'pending',
                        retry_count = ?1,
                        error_message = ?2,
                        last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        lease_until = NULL,
                        worker_id = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE queue_id = ?3
                "#;

                sqlx::query(query)
                    .bind(new_retry_count)
                    .bind(error_message)
                    .bind(queue_id)
                    .execute(&self.pool)
                    .await?;

                info!(
                    "Unified item {} failed, will retry ({}/{}): {}",
                    queue_id, new_retry_count, max_retries, error_message
                );

                Ok(true)
            } else {
                // Max retries exceeded - mark as permanently failed
                let query = r#"
                    UPDATE unified_queue
                    SET status = 'failed',
                        retry_count = ?1,
                        error_message = ?2,
                        last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        lease_until = NULL,
                        worker_id = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE queue_id = ?3
                "#;

                sqlx::query(query)
                    .bind(new_retry_count)
                    .bind(error_message)
                    .bind(queue_id)
                    .execute(&self.pool)
                    .await?;

                warn!(
                    "Unified item {} permanently failed after {} retries: {}",
                    queue_id, new_retry_count, error_message
                );

                METRICS.queue_item_processed("unified", "failure", 0.0);
                METRICS.ingestion_error("max_retries_exceeded");

                Ok(false)
            }
        } else {
            warn!("Unified queue item not found: {}", queue_id);
            Err(QueueError::NotFound(queue_id.to_string()))
        }
    }

    /// Recover stale leases from crashed workers
    ///
    /// Finds items with status 'in_progress' and expired leases,
    /// resets them to 'pending' for reprocessing.
    ///
    /// Should be called at daemon startup and periodically.
    ///
    /// Returns the number of recovered items.
    pub async fn recover_stale_unified_leases(&self) -> QueueResult<u64> {
        let now_str = Utc::now().to_rfc3339();

        let query = r#"
            UPDATE unified_queue
            SET status = 'pending',
                lease_until = NULL,
                worker_id = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE status = 'in_progress' AND lease_until < ?1
        "#;

        let result = sqlx::query(query)
            .bind(&now_str)
            .execute(&self.pool)
            .await?;

        let recovered = result.rows_affected();

        if recovered > 0 {
            info!("Recovered {} stale unified queue leases", recovered);
        } else {
            debug!("No stale unified queue leases to recover");
        }

        Ok(recovered)
    }

    /// Get statistics for the unified queue
    pub async fn get_unified_queue_stats(&self) -> QueueResult<UnifiedQueueStats> {
        let now_str = Utc::now().to_rfc3339();

        // Get counts by status
        let status_query = r#"
            SELECT
                COUNT(*) as total_items,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_items,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress_items,
                SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as done_items,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_items,
                SUM(CASE WHEN status = 'in_progress' AND lease_until < ?1 THEN 1 ELSE 0 END) as stale_leases,
                MIN(CASE WHEN status = 'pending' THEN created_at END) as oldest_pending,
                MAX(created_at) as newest_item
            FROM unified_queue
        "#;

        let row = sqlx::query(status_query)
            .bind(&now_str)
            .fetch_one(&self.pool)
            .await?;

        // Get counts by item_type
        let type_rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT item_type, COUNT(*) FROM unified_queue GROUP BY item_type"
        )
            .fetch_all(&self.pool)
            .await?;

        // Get counts by operation
        let op_rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT op, COUNT(*) FROM unified_queue GROUP BY op"
        )
            .fetch_all(&self.pool)
            .await?;

        Ok(UnifiedQueueStats {
            total_items: row.try_get("total_items")?,
            pending_items: row.try_get("pending_items")?,
            in_progress_items: row.try_get("in_progress_items")?,
            done_items: row.try_get("done_items")?,
            failed_items: row.try_get("failed_items")?,
            stale_leases: row.try_get("stale_leases")?,
            oldest_pending: row.try_get("oldest_pending")?,
            newest_item: row.try_get("newest_item")?,
            by_item_type: type_rows.into_iter().collect(),
            by_operation: op_rows.into_iter().collect(),
        })
    }

    /// Get the depth of the unified queue (pending items only)
    pub async fn get_unified_queue_depth(
        &self,
        item_type: Option<ItemType>,
        tenant_id: Option<&str>,
    ) -> QueueResult<i64> {
        let count: i64 = match (item_type, tenant_id) {
            (Some(itype), Some(tid)) => {
                sqlx::query_scalar(
                    "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND item_type = ?1 AND tenant_id = ?2"
                )
                    .bind(itype.to_string())
                    .bind(tid)
                    .fetch_one(&self.pool)
                    .await?
            }
            (Some(itype), None) => {
                sqlx::query_scalar(
                    "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND item_type = ?1"
                )
                    .bind(itype.to_string())
                    .fetch_one(&self.pool)
                    .await?
            }
            (None, Some(tid)) => {
                sqlx::query_scalar(
                    "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND tenant_id = ?1"
                )
                    .bind(tid)
                    .fetch_one(&self.pool)
                    .await?
            }
            (None, None) => {
                sqlx::query_scalar(
                    "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending'"
                )
                    .fetch_one(&self.pool)
                    .await?
            }
        };
        Ok(count)
    }

    /// Clean up completed items older than the specified retention period
    ///
    /// Removes items with status 'done' that were completed before the cutoff.
    ///
    /// # Arguments
    /// * `retention_hours` - How many hours to keep completed items (default: 24)
    ///
    /// Returns the number of items cleaned up.
    pub async fn cleanup_completed_unified_items(
        &self,
        retention_hours: Option<i64>,
    ) -> QueueResult<u64> {
        let hours = retention_hours.unwrap_or(24);

        let query = format!(
            "DELETE FROM unified_queue WHERE status = 'done' AND updated_at < datetime('now', '-{} hours')",
            hours
        );

        let result = sqlx::query(&query).execute(&self.pool).await?;

        let deleted = result.rows_affected();

        if deleted > 0 {
            info!("Cleaned up {} completed unified queue items", deleted);
        } else {
            debug!("No completed unified queue items to clean up");
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_config::QueueConnectionConfig;
    use tempfile::tempdir;

    async fn apply_sql_script(pool: &SqlitePool, script: &str) -> Result<(), sqlx::Error> {
        let mut conn = pool.acquire().await?;
        let mut statement = String::new();
        let mut in_trigger = false;

        for line in script.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("--") {
                continue;
            }

            if trimmed.to_uppercase().starts_with("CREATE TRIGGER") {
                in_trigger = true;
            }

            statement.push_str(line);
            statement.push('\n');

            if in_trigger {
                if trimmed.eq_ignore_ascii_case("END;") || trimmed.eq_ignore_ascii_case("END") {
                    in_trigger = false;
                    let stmt = statement.trim();
                    if !stmt.is_empty() {
                        sqlx::query(stmt).execute(&mut *conn).await?;
                    }
                    statement.clear();
                }
                continue;
            }

            if trimmed.ends_with(';') {
                let stmt = statement.trim();
                if !stmt.is_empty() {
                    sqlx::query(stmt).execute(&mut *conn).await?;
                }
                statement.clear();
            }
        }

        let remainder = statement.trim();
        if !remainder.is_empty() {
            sqlx::query(remainder).execute(&mut *conn).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_queue.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schema
        apply_sql_script(
            &pool,
            include_str!("../../../../python/common/core/queue_schema.sql"),
        )
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

    #[tokio::test]
    async fn test_update_retry_from() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_retry_from.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize base schema, then apply migration
        apply_sql_script(
            &pool,
            include_str!("../../../../python/common/core/queue_schema.sql"),
        )
        .await
        .unwrap();

        apply_sql_script(
            &pool,
            include_str!("../../../../python/common/core/schema/queue_retry_timestamp_migration.sql"),
        )
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

        // Update retry_from to future timestamp
        let future_time = Utc::now() + chrono::Duration::hours(1);
        let updated = manager
            .update_retry_from("/test/file.txt", future_time, 1)
            .await
            .unwrap();
        assert!(updated);

        // Dequeue should skip the item (retry_from is in the future)
        let items = manager.dequeue_batch(10, None, None).await.unwrap();
        assert_eq!(items.len(), 0);

        // Update retry_from to past timestamp
        let past_time = Utc::now() - chrono::Duration::hours(1);
        manager
            .update_retry_from("/test/file.txt", past_time, 1)
            .await
            .unwrap();

        // Dequeue should return the item now
        let items = manager.dequeue_batch(10, None, None).await.unwrap();
        assert_eq!(items.len(), 1);
    }

    #[tokio::test]
    async fn test_mark_failed() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_mark_failed.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schema
        apply_sql_script(
            &pool,
            include_str!("../../../../python/common/core/queue_schema.sql"),
        )
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

        // Mark as failed
        let removed = manager
            .mark_failed("/test/file.txt", "Max retries exceeded")
            .await
            .unwrap();
        assert!(removed);

        // Verify removed from queue
        let items = manager.dequeue_batch(10, None, None).await.unwrap();
        assert_eq!(items.len(), 0);
    }

    #[tokio::test]
    async fn test_missing_metadata_queue() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_missing_metadata.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schemas
        apply_sql_script(
            &pool,
            include_str!("../../../../python/common/core/queue_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool);
        manager.init_missing_metadata_queue().await.unwrap();

        // Create a queue item
        manager
            .enqueue_file(
                "/test/file.rs",
                "test-collection",
                "default",
                "main",
                QueueOperation::Ingest,
                5,
                None,
            )
            .await
            .unwrap();

        // Dequeue it
        let items = manager.dequeue_batch(10, None, None).await.unwrap();
        assert_eq!(items.len(), 1);
        let item = &items[0];

        // Create missing tools
        let missing_tools = vec![MissingTool::LspServer {
            language: "rust".to_string(),
        }];

        // Move to missing_metadata_queue
        let queue_id = manager
            .move_to_missing_metadata_queue(item, &missing_tools)
            .await
            .unwrap();

        assert!(!queue_id.is_empty());

        // Verify item removed from main queue
        let main_items = manager.dequeue_batch(10, None, None).await.unwrap();
        assert_eq!(main_items.len(), 0);

        // Verify item in missing_metadata_queue
        let missing_items = manager.get_missing_metadata_items(10).await.unwrap();
        assert_eq!(missing_items.len(), 1);
        assert_eq!(missing_items[0].file_absolute_path, "/test/file.rs");
        assert_eq!(missing_items[0].missing_tools.len(), 1);

        // Test queue depth
        let depth = manager.get_missing_metadata_queue_depth().await.unwrap();
        assert_eq!(depth, 1);

        // Test removal
        let removed = manager
            .remove_from_missing_metadata_queue(&queue_id)
            .await
            .unwrap();
        assert!(removed);

        let depth_after = manager.get_missing_metadata_queue_depth().await.unwrap();
        assert_eq!(depth_after, 0);
    }

    // ========================================================================
    // Unified Queue Tests (Task 37.21-37.29)
    // ========================================================================

    use crate::unified_queue_schema::{
        ItemType, QueueOperation as UnifiedOp, QueueStatus,
    };

    #[tokio::test]
    async fn test_unified_queue_enqueue_dequeue() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_queue.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue an item
        let (queue_id, is_new) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                5,
                Some("main"),
                None,
            )
            .await
            .unwrap();

        assert!(is_new);
        assert!(!queue_id.is_empty());

        // Enqueue same item again (idempotent)
        let (queue_id2, is_new2) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                5,
                Some("main"),
                None,
            )
            .await
            .unwrap();

        assert_eq!(queue_id, queue_id2);
        assert!(!is_new2); // Should be duplicate

        // Dequeue
        let items = manager
            .dequeue_unified(10, "worker-1", Some(300), None, None)
            .await
            .unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].queue_id, queue_id);
        assert_eq!(items[0].status, QueueStatus::InProgress);
        assert_eq!(items[0].worker_id, Some("worker-1".to_string()));
    }

    #[tokio::test]
    async fn test_unified_queue_mark_done() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_done.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue and dequeue
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::Content,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"content":"test"}"#,
                5,
                None,
                None,
            )
            .await
            .unwrap();

        let items = manager
            .dequeue_unified(10, "worker-1", None, None, None)
            .await
            .unwrap();
        assert_eq!(items.len(), 1);

        // Mark as done
        let marked = manager.mark_unified_done(&queue_id).await.unwrap();
        assert!(marked);

        // Verify status changed
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.done_items, 1);
        assert_eq!(stats.in_progress_items, 0);
    }

    #[tokio::test]
    async fn test_unified_queue_mark_failed_retry() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_failed.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                5,
                None,
                None,
            )
            .await
            .unwrap();

        // Dequeue
        manager
            .dequeue_unified(10, "worker-1", None, None, None)
            .await
            .unwrap();

        // First failure - should retry
        let will_retry = manager
            .mark_unified_failed(&queue_id, "Test error 1")
            .await
            .unwrap();
        assert!(will_retry);

        // Check it's back to pending
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.pending_items, 1);

        // Dequeue again and fail until max retries
        for i in 2..=3 {
            manager
                .dequeue_unified(10, "worker-1", None, None, None)
                .await
                .unwrap();
            let will_retry = manager
                .mark_unified_failed(&queue_id, &format!("Test error {}", i))
                .await
                .unwrap();

            if i < 3 {
                assert!(will_retry);
            } else {
                assert!(!will_retry); // Max retries exceeded
            }
        }

        // Verify permanently failed
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.failed_items, 1);
        assert_eq!(stats.pending_items, 0);
    }

    #[tokio::test]
    async fn test_unified_queue_recover_stale_leases() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_stale.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                5,
                None,
                None,
            )
            .await
            .unwrap();

        // Dequeue with very short lease (1 second)
        manager
            .dequeue_unified(10, "worker-1", Some(1), None, None)
            .await
            .unwrap();

        // Wait for lease to expire
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Recover stale leases
        let recovered = manager.recover_stale_unified_leases().await.unwrap();
        assert_eq!(recovered, 1);

        // Verify it's back to pending
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.pending_items, 1);
        assert_eq!(stats.in_progress_items, 0);
    }

    #[tokio::test]
    async fn test_unified_queue_stats() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_stats.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue items of different types
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file1.rs"}"#,
                5,
                None,
                None,
            )
            .await
            .unwrap();

        manager
            .enqueue_unified(
                ItemType::Content,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"content":"test content"}"#,
                8,
                None,
                None,
            )
            .await
            .unwrap();

        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Delete,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file2.rs"}"#,
                3,
                None,
                None,
            )
            .await
            .unwrap();

        let stats = manager.get_unified_queue_stats().await.unwrap();

        assert_eq!(stats.total_items, 3);
        assert_eq!(stats.pending_items, 3);
        assert_eq!(stats.by_item_type.get("file"), Some(&2));
        assert_eq!(stats.by_item_type.get("content"), Some(&1));
        assert_eq!(stats.by_operation.get("ingest"), Some(&2));
        assert_eq!(stats.by_operation.get("delete"), Some(&1));
    }

    #[tokio::test]
    async fn test_unified_queue_cleanup() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_cleanup.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue and complete an item
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::Content,
                UnifiedOp::Ingest,
                "test-tenant",
                "test-collection",
                r#"{"content":"test"}"#,
                5,
                None,
                None,
            )
            .await
            .unwrap();

        manager
            .dequeue_unified(10, "worker-1", None, None, None)
            .await
            .unwrap();
        manager.mark_unified_done(&queue_id).await.unwrap();

        // With 0 hours retention, it should be cleaned up immediately
        // But since it was just completed, it won't be older than now
        // So we test with a longer retention to verify no cleanup happens
        let cleaned = manager.cleanup_completed_unified_items(Some(24)).await.unwrap();
        assert_eq!(cleaned, 0); // Item is too recent

        // Verify item still exists
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.done_items, 1);
    }

    #[tokio::test]
    async fn test_unified_queue_depth() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_depth.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue items
        for i in 0..5 {
            manager
                .enqueue_unified(
                    ItemType::File,
                    UnifiedOp::Ingest,
                    "test-tenant",
                    "test-collection",
                    &format!(r#"{{"file_path":"/test/file{}.rs"}}"#, i),
                    5,
                    None,
                    None,
                )
                .await
                .unwrap();
        }

        // Check depth
        let depth = manager.get_unified_queue_depth(None, None).await.unwrap();
        assert_eq!(depth, 5);

        // Check depth filtered by type
        let depth_file = manager
            .get_unified_queue_depth(Some(ItemType::File), None)
            .await
            .unwrap();
        assert_eq!(depth_file, 5);

        let depth_content = manager
            .get_unified_queue_depth(Some(ItemType::Content), None)
            .await
            .unwrap();
        assert_eq!(depth_content, 0);
    }
}
