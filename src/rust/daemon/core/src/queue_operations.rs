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
use wqm_common::timestamps;
use wqm_common::constants::{COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_MEMORY};

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

    // Task 46: Strict validation errors
    #[error("tenant_id is required and cannot be empty or whitespace")]
    EmptyTenantId,

    #[error("collection is required and cannot be empty or whitespace")]
    EmptyCollection,

    #[error("Invalid payload JSON: {0}")]
    InvalidPayloadJson(String),

    #[error("Missing required field '{field}' in payload for item_type '{item_type}'")]
    MissingPayloadField { item_type: String, field: String },

    #[error("Internal queue error: {0}")]
    InternalError(String),
}

/// Result type for queue operations
pub type QueueResult<T> = Result<T, QueueError>;

/// Missing metadata queue item representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingMetadataItem {
    pub queue_id: String,
    pub file_absolute_path: String,
    pub collection_name: String,
    pub tenant_id: String,
    pub branch: String,
    pub operation: UnifiedOp,
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

    /// Validate payload contains required fields for the given item type (Task 46)
    ///
    /// Returns Ok(()) if valid, or an error describing the missing field.
    fn validate_payload_for_type(item_type: ItemType, op: UnifiedOp, payload: &serde_json::Value) -> QueueResult<()> {
        match item_type {
            ItemType::File => {
                if !payload.get("file_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: file item missing 'file_path' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "file".to_string(),
                        field: "file_path".to_string(),
                    });
                }
                if op == UnifiedOp::Rename {
                    if !payload.get("old_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                        error!("Queue validation failed: file rename missing 'old_path' in payload");
                        return Err(QueueError::MissingPayloadField {
                            item_type: "file".to_string(),
                            field: "old_path".to_string(),
                        });
                    }
                }
            }
            ItemType::Folder => {
                if !payload.get("folder_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: folder item missing 'folder_path' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "folder".to_string(),
                        field: "folder_path".to_string(),
                    });
                }
                if op == UnifiedOp::Rename {
                    if !payload.get("old_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                        error!("Queue validation failed: folder rename missing 'old_path' in payload");
                        return Err(QueueError::MissingPayloadField {
                            item_type: "folder".to_string(),
                            field: "old_path".to_string(),
                        });
                    }
                }
            }
            ItemType::Tenant => {
                // Tenant validation depends on collection context
                // Projects need project_root, libraries need library_name
                // For delete ops, tenant_id is sufficient (already in queue item)
                // For rename ops, need old_tenant_id
                if op == UnifiedOp::Rename {
                    if !payload.get("old_tenant_id").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                        error!("Queue validation failed: tenant rename missing 'old_tenant_id'");
                        return Err(QueueError::MissingPayloadField {
                            item_type: "tenant".to_string(),
                            field: "old_tenant_id".to_string(),
                        });
                    }
                }
                // For add/scan ops on projects, project_root is needed
                // For add ops on libraries, library_name is needed
                // These are validated contextually in the processor
            }
            ItemType::Doc => {
                if !payload.get("document_id").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: doc item missing 'document_id' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "doc".to_string(),
                        field: "document_id".to_string(),
                    });
                }
            }
            ItemType::Text => {
                // Text items must have a 'content' field (can be empty string for some operations)
                if !payload.get("content").map_or(false, |v| v.is_string()) {
                    error!("Queue validation failed: text item missing 'content' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "text".to_string(),
                        field: "content".to_string(),
                    });
                }
            }
            ItemType::Website => {
                if !payload.get("url").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: website item missing 'url' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "website".to_string(),
                        field: "url".to_string(),
                    });
                }
            }
            ItemType::Collection => {
                if !payload.get("collection_name").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: collection item missing 'collection_name' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "collection".to_string(),
                        field: "collection_name".to_string(),
                    });
                }
            }
            ItemType::Url => {
                if !payload.get("url").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: url item missing 'url' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "url".to_string(),
                        field: "url".to_string(),
                    });
                }
            }
        }
        Ok(())
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
            let operation = UnifiedOp::from_str(&operation_str)
                .unwrap_or(UnifiedOp::Add);

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

    /// Retry missing metadata item by enqueueing to unified_queue
    ///
    /// Note: Updated per Task 26 to use unified_queue instead of legacy ingestion_queue.
    pub async fn retry_missing_metadata_item(&self, queue_id: &str) -> QueueResult<bool> {
        use crate::unified_queue_schema::{ItemType, QueueOperation as UnifiedOp, FilePayload};

        // Start transaction
        let mut tx = self.pool.begin().await?;

        // Get the item from missing_metadata_queue
        let query = r#"
            SELECT
                file_absolute_path, collection_name, tenant_id, branch,
                operation
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

            // Map legacy operation to unified operation
            let unified_op = match operation_str.as_str() {
                "ingest" | "add" => UnifiedOp::Add,
                "update" => UnifiedOp::Update,
                "delete" => UnifiedOp::Delete,
                _ => UnifiedOp::Add, // Default to add
            };

            // Create file payload
            let payload = FilePayload {
                file_path: file_path.clone(),
                file_type: None,
                file_hash: None,
                size_bytes: None,
                old_path: None,
            };
            let payload_json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());

            // Commit transaction before enqueueing (enqueue_unified uses its own transaction)
            tx.commit().await?;

            // Enqueue to unified_queue (priority is dynamic, always pass 0)
            match self.enqueue_unified(
                ItemType::File,
                unified_op,
                &tenant_id,
                &collection,
                &payload_json,
                0,
                Some(&branch),
                None, // No custom idempotency key - let it generate
            ).await {
                Ok((new_queue_id, _is_new)) => {
                    // Update last_check_timestamp in missing_metadata_queue
                    let update_query = r#"
                        UPDATE missing_metadata_queue
                        SET last_check_timestamp = ?1, retry_count = retry_count + 1
                        WHERE queue_id = ?2
                    "#;

                    sqlx::query(update_query)
                        .bind(timestamps::now_utc())
                        .bind(queue_id)
                        .execute(&self.pool)
                        .await?;

                    info!(
                        "Retrying item from missing_metadata_queue: {} -> unified_queue {}",
                        file_path, new_queue_id
                    );
                    Ok(true)
                }
                Err(e) => {
                    error!("Failed to enqueue retry item to unified_queue: {}", e);
                    Err(e)
                }
            }
        } else {
            warn!("Queue item not found in missing_metadata_queue: {}", queue_id);
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
    /// * `priority` - UNUSED: always pass 0. Priority is computed at dequeue time via CASE/JOIN.
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
        // Task 46: Strict validation
        // Validate tenant_id (cannot be empty or whitespace-only)
        let tenant_id = tenant_id.trim();
        if tenant_id.is_empty() {
            error!("Queue validation failed: tenant_id is empty or whitespace-only");
            return Err(QueueError::EmptyTenantId);
        }

        // Validate collection (cannot be empty or whitespace-only)
        let collection = collection.trim();
        if collection.is_empty() {
            error!("Queue validation failed: collection is empty or whitespace-only");
            return Err(QueueError::EmptyCollection);
        }

        // Validate priority
        if !(0..=10).contains(&priority) {
            return Err(QueueError::InvalidPriority(priority));
        }

        // Validate payload_json is valid JSON
        let payload: serde_json::Value = serde_json::from_str(payload_json)
            .map_err(|e| {
                error!("Queue validation failed: invalid payload JSON - {}", e);
                QueueError::InvalidPayloadJson(e.to_string())
            })?;

        // Type-specific payload validation
        Self::validate_payload_for_type(item_type, op, &payload)?;

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

        // Task 22: Extract file_path for per-file deduplication
        // Only set for item_type='file', NULL for other types
        let file_path: Option<String> = if item_type == ItemType::File {
            payload.get("file_path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

        // Use INSERT OR IGNORE to handle race conditions, then check what happened
        // The UNIQUE constraint on file_path will also prevent duplicate file entries
        let insert_query = r#"
            INSERT OR IGNORE INTO unified_queue (
                item_type, op, tenant_id, collection, priority,
                branch, payload_json, metadata, idempotency_key, file_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
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
            .bind(&file_path)
            .execute(&self.pool)
            .await?;

        let is_new = result.rows_affected() > 0;

        // Get the queue_id (either newly inserted or existing)
        // When INSERT OR IGNORE is ignored, it could be due to either:
        //   1. idempotency_key UNIQUE constraint (same exact operation)
        //   2. file_path UNIQUE constraint (same file, different operation/metadata)
        // We try idempotency_key first, then fall back to file_path for case 2.
        let queue_id: String = if is_new {
            sqlx::query_scalar(
                "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1"
            )
                .bind(&idempotency_key)
                .fetch_one(&self.pool)
                .await?
        } else {
            // INSERT was ignored — try idempotency_key first
            match sqlx::query_scalar::<_, String>(
                "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1"
            )
                .bind(&idempotency_key)
                .fetch_optional(&self.pool)
                .await?
            {
                Some(id) => id,
                None => {
                    // Ignored due to file_path UNIQUE constraint (different idempotency_key)
                    if let Some(ref fp) = file_path {
                        sqlx::query_scalar(
                            "SELECT queue_id FROM unified_queue WHERE file_path = ?1"
                        )
                            .bind(fp)
                            .fetch_one(&self.pool)
                            .await?
                    } else {
                        return Err(QueueError::InternalError(
                            "INSERT OR IGNORE returned 0 rows but no matching idempotency_key or file_path found".to_string()
                        ));
                    }
                }
            }
        };

        if is_new {
            debug!(
                "Enqueued unified item: {} (type={}, op={}, collection={})",
                queue_id, item_type, op, collection
            );
        } else {
            // Update timestamp to show we tried to enqueue again
            // Use queue_id (always valid) instead of idempotency_key (may not match)
            sqlx::query(
                "UPDATE unified_queue SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE queue_id = ?1"
            )
                .bind(&queue_id)
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
    /// * `priority_descending` - If true, high priority first (DESC) with FIFO tiebreaker;
    ///   if false, low priority first (ASC) with LIFO tiebreaker.
    ///   Used for anti-starvation alternation. Defaults to true if None.
    pub async fn dequeue_unified(
        &self,
        batch_size: i32,
        worker_id: &str,
        lease_duration_secs: Option<i64>,
        tenant_id: Option<&str>,
        item_type: Option<ItemType>,
        priority_descending: Option<bool>,
    ) -> QueueResult<Vec<UnifiedQueueItem>> {
        let lease_duration = lease_duration_secs.unwrap_or(300);
        let lease_until = Utc::now() + ChronoDuration::seconds(lease_duration);
        let lease_until_str = timestamps::format_utc(&lease_until);
        let now_str = timestamps::now_utc();

        // Task 21: Priority direction for anti-starvation alternation
        // When true (default): high priority first (DESC) - active projects prioritized
        // When false: low priority first (ASC) - inactive projects get a turn
        let is_descending = priority_descending.unwrap_or(true);
        let priority_order = if is_descending { "DESC" } else { "ASC" };

        // Task 9: FIFO/LIFO alternation for idle processing
        // DESC phase (active first): FIFO tiebreaker (created_at ASC) — clear old backlog
        // ASC phase (anti-starvation): LIFO tiebreaker (created_at DESC) — recent changes first
        // When all projects are idle, the priority CASE is equal for all items,
        // so this effectively alternates between FIFO and LIFO batches.
        let created_at_order = if is_descending { "ASC" } else { "DESC" };

        // First, select the queue_ids to process with calculated priority (Task 20)
        // Priority is computed at query time via JOIN with watch_folders:
        // - memory collection: 1 (high)
        // - libraries collection: 0 (low)
        // - projects collection with is_active=1: 1 (high)
        // - projects collection with is_active=0: 0 (low)
        let queue_ids: Vec<String> = match (tenant_id, item_type) {
            (Some(tid), Some(itype)) => {
                // Task 21: Dynamic priority ordering for anti-starvation
                let query = format!(
                    r#"
                    SELECT q.queue_id
                    FROM unified_queue q
                    LEFT JOIN watch_folders w
                        ON q.tenant_id = w.tenant_id
                        AND q.collection = '{coll_projects}'
                        AND w.parent_watch_id IS NULL
                    WHERE (
                        (q.status = 'pending' AND (q.lease_until IS NULL OR q.lease_until < ?1))
                        OR (q.status = 'in_progress' AND q.lease_until < ?1)
                    )
                    AND q.tenant_id = ?2
                    AND q.item_type = ?3
                    ORDER BY
                        CASE
                            WHEN q.collection = '{coll_memory}' THEN 1
                            WHEN q.collection = '{coll_libraries}' THEN 0
                            WHEN w.is_active = 1 THEN 1
                            ELSE 0
                        END {priority_order},
                        q.created_at {created_at_order}
                    LIMIT ?4
                    "#,
                    coll_projects = COLLECTION_PROJECTS,
                    coll_libraries = COLLECTION_LIBRARIES,
                    coll_memory = COLLECTION_MEMORY,
                    priority_order = priority_order,
                    created_at_order = created_at_order,
                );
                sqlx::query_scalar::<_, String>(&query)
                    .bind(&now_str)
                    .bind(tid)
                    .bind(itype.to_string())
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (Some(tid), None) => {
                // Task 21: Dynamic priority ordering for anti-starvation
                let query = format!(
                    r#"
                    SELECT q.queue_id
                    FROM unified_queue q
                    LEFT JOIN watch_folders w
                        ON q.tenant_id = w.tenant_id
                        AND q.collection = '{coll_projects}'
                        AND w.parent_watch_id IS NULL
                    WHERE (
                        (q.status = 'pending' AND (q.lease_until IS NULL OR q.lease_until < ?1))
                        OR (q.status = 'in_progress' AND q.lease_until < ?1)
                    )
                    AND q.tenant_id = ?2
                    ORDER BY
                        CASE
                            WHEN q.collection = '{coll_memory}' THEN 1
                            WHEN q.collection = '{coll_libraries}' THEN 0
                            WHEN w.is_active = 1 THEN 1
                            ELSE 0
                        END {priority_order},
                        q.created_at {created_at_order}
                    LIMIT ?3
                    "#,
                    coll_projects = COLLECTION_PROJECTS,
                    coll_libraries = COLLECTION_LIBRARIES,
                    coll_memory = COLLECTION_MEMORY,
                    priority_order = priority_order,
                    created_at_order = created_at_order,
                );
                sqlx::query_scalar::<_, String>(&query)
                    .bind(&now_str)
                    .bind(tid)
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (None, Some(itype)) => {
                // Task 21: Dynamic priority ordering for anti-starvation
                let query = format!(
                    r#"
                    SELECT q.queue_id
                    FROM unified_queue q
                    LEFT JOIN watch_folders w
                        ON q.tenant_id = w.tenant_id
                        AND q.collection = '{coll_projects}'
                        AND w.parent_watch_id IS NULL
                    WHERE (
                        (q.status = 'pending' AND (q.lease_until IS NULL OR q.lease_until < ?1))
                        OR (q.status = 'in_progress' AND q.lease_until < ?1)
                    )
                    AND q.item_type = ?2
                    ORDER BY
                        CASE
                            WHEN q.collection = '{coll_memory}' THEN 1
                            WHEN q.collection = '{coll_libraries}' THEN 0
                            WHEN w.is_active = 1 THEN 1
                            ELSE 0
                        END {priority_order},
                        q.created_at {created_at_order}
                    LIMIT ?3
                    "#,
                    coll_projects = COLLECTION_PROJECTS,
                    coll_libraries = COLLECTION_LIBRARIES,
                    coll_memory = COLLECTION_MEMORY,
                    priority_order = priority_order,
                    created_at_order = created_at_order,
                );
                sqlx::query_scalar::<_, String>(&query)
                    .bind(&now_str)
                    .bind(itype.to_string())
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (None, None) => {
                // Task 21: Dynamic priority ordering for anti-starvation
                let query = format!(
                    r#"
                    SELECT q.queue_id
                    FROM unified_queue q
                    LEFT JOIN watch_folders w
                        ON q.tenant_id = w.tenant_id
                        AND q.collection = '{coll_projects}'
                        AND w.parent_watch_id IS NULL
                    WHERE (
                        (q.status = 'pending' AND (q.lease_until IS NULL OR q.lease_until < ?1))
                        OR (q.status = 'in_progress' AND q.lease_until < ?1)
                    )
                    ORDER BY
                        CASE
                            WHEN q.collection = '{coll_memory}' THEN 1
                            WHEN q.collection = '{coll_libraries}' THEN 0
                            WHEN w.is_active = 1 THEN 1
                            ELSE 0
                        END {priority_order},
                        q.created_at {created_at_order}
                    LIMIT ?2
                    "#,
                    coll_projects = COLLECTION_PROJECTS,
                    coll_libraries = COLLECTION_LIBRARIES,
                    coll_memory = COLLECTION_MEMORY,
                    priority_order = priority_order,
                    created_at_order = created_at_order,
                );
                sqlx::query_scalar::<_, String>(&query)
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

        let mut items = Vec::with_capacity(rows.len());
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
                file_path: row.try_get("file_path")?, // Task 22
            });
        }

        // Preserve the ordering from the initial SELECT (which has ORDER BY).
        // The fetch query uses WHERE IN (...) which returns rows in arbitrary order.
        {
            let id_positions: std::collections::HashMap<&str, usize> = queue_ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.as_str(), i))
                .collect();
            items.sort_by_key(|item| *id_positions.get(item.queue_id.as_str()).unwrap_or(&usize::MAX));
        }

        debug!(
            "Dequeued {} unified items for worker {}",
            items.len(),
            worker_id
        );

        Ok(items)
    }

    /// Delete a unified queue item after successful processing
    ///
    /// Per WORKSPACE_QDRANT_MCP.md spec line 813:
    /// "On success: DELETE items from queue"
    ///
    /// This is the correct method for handling successfully processed items.
    /// Use this instead of mark_unified_done.
    pub async fn delete_unified_item(&self, queue_id: &str) -> QueueResult<bool> {
        let result = sqlx::query("DELETE FROM unified_queue WHERE queue_id = ?1")
            .bind(queue_id)
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() > 0;

        if deleted {
            debug!("Deleted unified item after successful processing: {}", queue_id);
            METRICS.queue_item_processed("unified", "deleted", 0.0);
        } else {
            warn!("Failed to delete unified item: {} (not found)", queue_id);
        }

        Ok(deleted)
    }


    /// Mark a unified queue item as failed
    ///
    /// If `permanent` is true, skips retry logic and marks as failed immediately.
    /// Otherwise, if retries remain, increments retry_count, sets exponential
    /// backoff delay via `lease_until`, and resets to pending.
    /// If max retries exceeded, sets status to 'failed'.
    ///
    /// Backoff schedule: 60s * 2^retry_count, capped at 1 hour, with 10% jitter.
    ///
    /// Returns true if the item will be retried, false if permanently failed.
    pub async fn mark_unified_failed(
        &self,
        queue_id: &str,
        error_message: &str,
        permanent: bool,
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

            if !permanent && new_retry_count < max_retries {
                // Can retry - reset to pending with incremented retry count
                // Apply exponential backoff: 60s * 2^retry_count, capped at 3600s
                let base_delay_secs = 60.0_f64;
                let delay_secs = (base_delay_secs * 2.0_f64.powi(retry_count)).min(3600.0);
                // Add 10% jitter to prevent thundering herd
                let jitter = delay_secs * 0.1 * rand::random::<f64>();
                let total_delay = delay_secs + jitter;
                let retry_after = Utc::now() + ChronoDuration::seconds(total_delay as i64);
                let retry_after_str = timestamps::format_utc(&retry_after);

                let query = r#"
                    UPDATE unified_queue
                    SET status = 'pending',
                        retry_count = ?1,
                        error_message = ?2,
                        last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        lease_until = ?3,
                        worker_id = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE queue_id = ?4
                "#;

                sqlx::query(query)
                    .bind(new_retry_count)
                    .bind(error_message)
                    .bind(&retry_after_str)
                    .bind(queue_id)
                    .execute(&self.pool)
                    .await?;

                info!(
                    "Unified item {} failed, will retry ({}/{}) after {:.0}s backoff: {}",
                    queue_id, new_retry_count, max_retries, total_delay, error_message
                );

                Ok(true)
            } else {
                // Permanent error or max retries exceeded - mark as permanently failed
                let reason = if permanent { "permanent error" } else { "max retries exceeded" };

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
                    "Unified item {} permanently failed ({}, attempt {}/{}): {}",
                    queue_id, reason, new_retry_count, max_retries, error_message
                );

                METRICS.queue_item_processed("unified", "failure", 0.0);
                METRICS.ingestion_error(reason);

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
        let now_str = timestamps::now_utc();

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
        let now_str = timestamps::now_utc();

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

    /// Get the depth of the unified queue per collection (pending items only)
    ///
    /// Returns a HashMap mapping collection names to their pending item counts.
    /// Used for queue depth monitoring and throttling decisions.
    pub async fn get_unified_queue_depth_all_collections(&self) -> QueueResult<HashMap<String, i64>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT collection, COUNT(*) as depth FROM unified_queue WHERE status = 'pending' GROUP BY collection",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().collect())
    }

    /// Get the oldest pending item in the unified queue
    ///
    /// Used by the fairness scheduler to check for stale items that need
    /// priority processing (starvation guard).
    ///
    /// Returns the oldest pending item without acquiring a lease.
    pub async fn get_oldest_pending_unified_item(&self) -> QueueResult<Option<UnifiedQueueItem>> {
        let query = r#"
            SELECT * FROM unified_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 1
        "#;

        let row = sqlx::query(query).fetch_optional(&self.pool).await?;

        match row {
            Some(row) => {
                let item_type_str: String = row.try_get("item_type")?;
                let op_str: String = row.try_get("op")?;
                let status_str: String = row.try_get("status")?;

                Ok(Some(UnifiedQueueItem {
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
                    file_path: row.try_get("file_path")?, // Task 22
                }))
            }
            None => Ok(None),
        }
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

    /// Enqueue cascade rename operations for all specified collections.
    ///
    /// Creates a rename queue item for each collection that needs its
    /// tenant_id payloads updated in Qdrant. Called after SQLite state
    /// (watch_folders, tracked_files) has already been updated.
    pub async fn enqueue_cascade_rename(
        &self,
        old_tenant_id: &str,
        new_tenant_id: &str,
        collections: &[&str],
        reason: &str,
    ) -> QueueResult<Vec<String>> {
        use crate::unified_queue_schema::ProjectPayload;

        let mut queue_ids = Vec::new();

        for collection in collections {
            let payload = ProjectPayload {
                project_root: String::new(), // Not needed for rename
                git_remote: None,
                project_type: None,
                old_tenant_id: Some(old_tenant_id.to_string()),
                is_active: None,
            };

            let payload_json = serde_json::to_string(&payload)
                .map_err(|e| QueueError::InvalidPayloadJson(e.to_string()))?;

            let metadata = serde_json::json!({
                "reason": reason,
            }).to_string();

            let (queue_id, _is_new) = self.enqueue_unified(
                ItemType::Tenant,
                UnifiedOp::Rename,
                new_tenant_id,
                collection,
                &payload_json,
                0,
                None,
                Some(&metadata),
            ).await?;

            queue_ids.push(queue_id);
        }

        info!(
            "Enqueued {} cascade rename items: {} -> {} (reason: {})",
            queue_ids.len(), old_tenant_id, new_tenant_id, reason
        );

        Ok(queue_ids)
    }

    /// Enqueue a library document for ingestion.
    ///
    /// Convenience wrapper around `enqueue_unified` that accepts a
    /// `LibraryDocumentPayload` and routes to the libraries collection.
    pub async fn enqueue_library_document(
        &self,
        payload: &wqm_common::payloads::LibraryDocumentPayload,
        op: UnifiedOp,
        branch: Option<&str>,
    ) -> QueueResult<(String, bool)> {
        let payload_json = serde_json::to_string(payload)?;
        self.enqueue_unified(
            ItemType::File,
            op,
            &payload.library_name,
            COLLECTION_LIBRARIES,
            &payload_json,
            0, // priority computed at dequeue
            branch,
            None,
        ).await
    }

    /// Validate a library document payload has required fields.
    ///
    /// Called during library document processing to ensure the payload
    /// contains the document family taxonomy fields.
    pub fn validate_library_document_payload(
        payload: &serde_json::Value,
    ) -> QueueResult<()> {
        let required_fields = [
            ("document_path", "Library document missing 'document_path'"),
            ("library_name", "Library document missing 'library_name'"),
            ("document_type", "Library document missing 'document_type'"),
            ("source_format", "Library document missing 'source_format'"),
            ("doc_id", "Library document missing 'doc_id'"),
        ];

        for (field, msg) in &required_fields {
            if !payload.get(*field).map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                error!("Queue validation failed: {}", msg);
                return Err(QueueError::MissingPayloadField {
                    item_type: "file".to_string(),
                    field: field.to_string(),
                });
            }
        }

        // Validate document_type is one of the known families
        if let Some(doc_type) = payload.get("document_type").and_then(|v| v.as_str()) {
            if doc_type != "page_based" && doc_type != "stream_based" {
                error!("Queue validation failed: invalid document_type '{}', must be 'page_based' or 'stream_based'", doc_type);
                return Err(QueueError::InvalidOperation(
                    format!("Invalid document_type: '{}', must be 'page_based' or 'stream_based'", doc_type),
                ));
            }
        }

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_config::QueueConnectionConfig;
    use std::sync::Arc;
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

    // NOTE: Legacy queue tests removed per Task 21
    // The following tests for ingestion_queue have been removed:
    // - test_enqueue_dequeue
    // - test_priority_validation
    // - test_update_retry_from
    // - test_mark_failed
    // - test_missing_metadata_queue
    //
    // See unified queue tests below for spec-compliant queue testing.

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

        // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue an item
        let (queue_id, is_new) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
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
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                Some("main"),
                None,
            )
            .await
            .unwrap();

        assert_eq!(queue_id, queue_id2);
        assert!(!is_new2); // Should be duplicate

        // Dequeue
        let items = manager
            .dequeue_unified(10, "worker-1", Some(300), None, None, None)
            .await
            .unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].queue_id, queue_id);
        assert_eq!(items[0].status, QueueStatus::InProgress);
        assert_eq!(items[0].worker_id, Some("worker-1".to_string()));
    }

    #[tokio::test]
    async fn test_unified_queue_delete_item() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_delete.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue and dequeue
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::Text,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"content":"test"}"#,
                0,
                None,
                None,
            )
            .await
            .unwrap();

        let items = manager
            .dequeue_unified(10, "worker-1", None, None, None, None)
            .await
            .unwrap();
        assert_eq!(items.len(), 1);

        // Delete item after successful processing (per spec)
        let deleted = manager.delete_unified_item(&queue_id).await.unwrap();
        assert!(deleted);

        // Verify item is completely gone
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.done_items, 0);
        assert_eq!(stats.in_progress_items, 0);
        assert_eq!(stats.pending_items, 0);
        assert_eq!(stats.failed_items, 0);
    }


    #[tokio::test]
    async fn test_unified_queue_mark_failed_retry() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_failed.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();
        let test_pool = pool.clone(); // Keep reference for test-only backoff reset

        // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await
            .unwrap();

        // Dequeue
        manager
            .dequeue_unified(10, "worker-1", None, None, None, None)
            .await
            .unwrap();

        // First failure (transient) - should retry with backoff
        let will_retry = manager
            .mark_unified_failed(&queue_id, "Test error 1", false)
            .await
            .unwrap();
        assert!(will_retry);

        // Check it's back to pending (with backoff lease_until)
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.pending_items, 1);

        // Item has backoff, so dequeue won't return it. Reset lease_until for test.
        sqlx::query("UPDATE unified_queue SET lease_until = NULL WHERE queue_id = ?1")
            .bind(&queue_id)
            .execute(&test_pool)
            .await
            .unwrap();

        // Dequeue again and fail until max retries
        for i in 2..=3 {
            manager
                .dequeue_unified(10, "worker-1", None, None, None, None)
                .await
                .unwrap();
            let will_retry = manager
                .mark_unified_failed(&queue_id, &format!("Test error {}", i), false)
                .await
                .unwrap();

            if i < 3 {
                assert!(will_retry);
                // Clear backoff for next test iteration
                sqlx::query("UPDATE unified_queue SET lease_until = NULL WHERE queue_id = ?1")
                    .bind(&queue_id)
                    .execute(&test_pool)
                    .await
                    .unwrap();
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
    async fn test_unified_queue_mark_failed_permanent() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_permanent.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schemas
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await
            .unwrap();

        // Dequeue
        manager
            .dequeue_unified(10, "worker-1", None, None, None, None)
            .await
            .unwrap();

        // Permanent failure - should NOT retry even though retries remain
        let will_retry = manager
            .mark_unified_failed(&queue_id, "File not found: /test/file.rs", true)
            .await
            .unwrap();
        assert!(!will_retry, "Permanent errors should not retry");

        // Verify immediately failed (not pending)
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.failed_items, 1);
        assert_eq!(stats.pending_items, 0);
    }

    #[tokio::test]
    async fn test_unified_queue_backoff_prevents_immediate_dequeue() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_backoff.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schemas
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await
            .unwrap();

        // Dequeue
        manager
            .dequeue_unified(10, "worker-1", None, None, None, None)
            .await
            .unwrap();

        // Transient failure with backoff
        let will_retry = manager
            .mark_unified_failed(&queue_id, "Connection refused", false)
            .await
            .unwrap();
        assert!(will_retry);

        // Try to dequeue immediately - should get nothing (item is in backoff)
        let items = manager
            .dequeue_unified(10, "worker-2", None, None, None, None)
            .await
            .unwrap();
        assert!(items.is_empty(), "Item should not be dequeued during backoff");

        // Verify item is still pending (not lost)
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.pending_items, 1);
        assert_eq!(stats.failed_items, 0);
    }

    #[tokio::test]
    async fn test_unified_queue_recover_stale_leases() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_stale.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Enqueue
        manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await
            .unwrap();

        // Dequeue with very short lease (1 second)
        manager
            .dequeue_unified(10, "worker-1", Some(1), None, None, None)
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
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file1.rs"}"#,
                0,
                None,
                None,
            )
            .await
            .unwrap();

        manager
            .enqueue_unified(
                ItemType::Text,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"content":"test content"}"#,
                0,
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
                0,
                None,
                None,
            )
            .await
            .unwrap();

        let stats = manager.get_unified_queue_stats().await.unwrap();

        assert_eq!(stats.total_items, 3);
        assert_eq!(stats.pending_items, 3);
        assert_eq!(stats.by_item_type.get("file"), Some(&2));
        assert_eq!(stats.by_item_type.get("text"), Some(&1));
        assert_eq!(stats.by_operation.get("add"), Some(&2));
        assert_eq!(stats.by_operation.get("delete"), Some(&1));
    }

    #[tokio::test]
    async fn test_unified_queue_cleanup() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_unified_cleanup.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = QueueManager::new(pool.clone());
        manager.init_unified_queue().await.unwrap();

        // Enqueue and dequeue an item
        let (queue_id, _) = manager
            .enqueue_unified(
                ItemType::Text,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"content":"test"}"#,
                0,
                None,
                None,
            )
            .await
            .unwrap();

        manager
            .dequeue_unified(10, "worker-1", None, None, None, None)
            .await
            .unwrap();

        // Set status to 'done' directly via SQL to test cleanup of done items
        sqlx::query("UPDATE unified_queue SET status = 'done', lease_until = NULL, worker_id = NULL WHERE queue_id = ?1")
            .bind(&queue_id)
            .execute(&pool)
            .await
            .unwrap();

        // With 24 hours retention, recently completed items should NOT be cleaned up
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
                    UnifiedOp::Add,
                    "test-tenant",
                    "test-collection",
                    &format!(r#"{{"file_path":"/test/file{}.rs"}}"#, i),
                    0,
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
            .get_unified_queue_depth(Some(ItemType::Text), None)
            .await
            .unwrap();
        assert_eq!(depth_content, 0);
    }

    // Concurrent Idempotency Tests (Task 45)
    // ========================================================================

    #[tokio::test]
    async fn test_concurrent_enqueue_idempotency() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_concurrent_idempotency.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = Arc::new(QueueManager::new(pool));
        manager.init_unified_queue().await.unwrap();

        // Spawn 10 concurrent enqueue operations for the same item
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let mgr = Arc::clone(&manager);
                tokio::spawn(async move {
                    mgr.enqueue_unified(
                        ItemType::File,
                        UnifiedOp::Add,
                        "test-tenant",
                        "test-collection",
                        r#"{"file_path":"/test/concurrent_file.rs"}"#,
                        0,
                        Some("main"),
                        Some(&format!(r#"{{"worker":{}}}"#, i)),
                    )
                    .await
                })
            })
            .collect();

        // Wait for all to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // All should succeed (no errors from UNIQUE constraint violation)
        assert!(results.iter().all(|r| r.is_ok()));

        // All should return the same queue_id
        let queue_ids: Vec<_> = results
            .into_iter()
            .map(|r| r.unwrap().0)
            .collect();

        let first_id = &queue_ids[0];
        assert!(queue_ids.iter().all(|id| id == first_id),
            "All concurrent enqueues should return the same queue_id");

        // Only one row should exist in the database
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.total_items, 1,
            "Only one item should exist despite concurrent enqueues");

        // At most one should report is_new=true (the one that won the race)
        // Note: We can't easily test this since we don't capture is_new in the join
    }

    #[tokio::test]
    async fn test_concurrent_enqueue_different_items() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_concurrent_different.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = Arc::new(QueueManager::new(pool));
        manager.init_unified_queue().await.unwrap();

        // Spawn 10 concurrent enqueue operations for different items
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let mgr = Arc::clone(&manager);
                tokio::spawn(async move {
                    mgr.enqueue_unified(
                        ItemType::File,
                        UnifiedOp::Add,
                        "test-tenant",
                        "test-collection",
                        &format!(r#"{{"file_path":"/test/file_{}.rs"}}"#, i),
                        0,
                        Some("main"),
                        None,
                    )
                    .await
                })
            })
            .collect();

        // Wait for all to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // All should succeed
        assert!(results.iter().all(|r| r.is_ok()));

        // All should be new items
        let new_flags: Vec<_> = results
            .into_iter()
            .map(|r| r.unwrap().1)
            .collect();
        assert!(new_flags.iter().all(|&is_new| is_new),
            "All different items should be marked as new");

        // All 10 items should exist
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.total_items, 10,
            "All 10 different items should exist");
    }

    #[tokio::test]
    async fn test_concurrent_enqueue_mixed_operations() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_concurrent_mixed.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = Arc::new(QueueManager::new(pool));
        manager.init_unified_queue().await.unwrap();

        // Enqueue the same content with different operations (each should be unique)
        // Note: Uses ItemType::Text (not File) to avoid per-file UNIQUE constraint (Task 22)
        let ops = vec![
            (UnifiedOp::Add, "ingest"),
            (UnifiedOp::Update, "update"),
            (UnifiedOp::Delete, "delete"),
        ];

        let handles: Vec<_> = ops
            .into_iter()
            .map(|(op, _name)| {
                let mgr = Arc::clone(&manager);
                tokio::spawn(async move {
                    mgr.enqueue_unified(
                        ItemType::Text,
                        op,
                        "test-tenant",
                        "test-collection",
                        r#"{"content":"test content","source_type":"test"}"#,
                        0,
                        Some("main"),
                        None,
                    )
                    .await
                })
            })
            .collect();

        // Wait for all to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // All should succeed
        assert!(results.iter().all(|r| r.is_ok()));

        // All should be new (different operations = different idempotency keys)
        let new_flags: Vec<_> = results
            .into_iter()
            .map(|r| r.unwrap().1)
            .collect();
        assert!(new_flags.iter().all(|&is_new| is_new),
            "Different operations should create different items");

        // All 3 items should exist
        let stats = manager.get_unified_queue_stats().await.unwrap();
        assert_eq!(stats.total_items, 3,
            "3 items with different operations should exist");
    }

    #[tokio::test]
    async fn test_idempotency_across_workers() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_idempotency_workers.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schemas (watch_folders required for JOIN in dequeue_unified)
        apply_sql_script(
            &pool,
            include_str!("schema/watch_folders_schema.sql"),
        )
        .await
        .unwrap();

        let manager = Arc::new(QueueManager::new(pool));
        manager.init_unified_queue().await.unwrap();

        // First enqueue
        let (queue_id1, is_new1) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/worker_test.rs"}"#,
                0,
                Some("main"),
                None,
            )
            .await
            .unwrap();

        assert!(is_new1);

        // Dequeue with worker-1
        let items = manager
            .dequeue_unified(10, "worker-1", Some(300), None, None, None)
            .await
            .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].worker_id, Some("worker-1".to_string()));

        // Try to enqueue same item again (while worker-1 is processing)
        let (queue_id2, is_new2) = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/worker_test.rs"}"#,
                0,
                Some("main"),
                None,
            )
            .await
            .unwrap();

        // Should return same queue_id, not new
        assert_eq!(queue_id1, queue_id2);
        assert!(!is_new2);

        // Item should still be in_progress with worker-1's lease
        let items_after = manager
            .dequeue_unified(10, "worker-2", Some(300), None, None, None)
            .await
            .unwrap();
        assert_eq!(items_after.len(), 0, "No items should be available - lease still held by worker-1");
    }

    // Queue Validation Tests (Task 46)
    // ========================================================================

    #[tokio::test]
    async fn test_validation_empty_tenant_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_tenant.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Empty tenant_id should fail
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QueueError::EmptyTenantId));
    }

    #[tokio::test]
    async fn test_validation_whitespace_tenant_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_ws_tenant.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Whitespace-only tenant_id should fail
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "   ",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QueueError::EmptyTenantId));
    }

    #[tokio::test]
    async fn test_validation_empty_collection() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_collection.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Empty collection should fail
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QueueError::EmptyCollection));
    }

    #[tokio::test]
    async fn test_validation_invalid_json_payload() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_json.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Invalid JSON should fail
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                "not valid json",
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QueueError::InvalidPayloadJson(_)));
    }

    #[tokio::test]
    async fn test_validation_file_missing_file_path() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_file_path.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // File item without file_path should fail
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"other_field":"value"}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QueueError::MissingPayloadField { item_type, field }
            if item_type == "file" && field == "file_path"
        ));
    }

    #[tokio::test]
    async fn test_validation_content_missing_content() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_content.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Content item without content field should fail
        let result = manager
            .enqueue_unified(
                ItemType::Text,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"source_type":"mcp"}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QueueError::MissingPayloadField { item_type, field }
            if item_type == "text" && field == "content"
        ));
    }

    #[tokio::test]
    async fn test_validation_delete_document_missing_document_id() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_delete_doc.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // DeleteDocument without document_id should fail
        let result = manager
            .enqueue_unified(
                ItemType::Doc,
                UnifiedOp::Delete,
                "test-tenant",
                "test-collection",
                r#"{"point_ids":["abc"]}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QueueError::MissingPayloadField { item_type, field }
            if item_type == "doc" && field == "document_id"
        ));
    }

    #[tokio::test]
    async fn test_validation_file_rename_missing_old_path() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_rename.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // File rename without old_path should fail
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Rename,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/new.rs"}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QueueError::MissingPayloadField { item_type, field }
            if item_type == "file" && field == "old_path"
        ));
    }

    #[tokio::test]
    async fn test_validation_valid_items_pass() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_valid.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // Valid file item should succeed
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/file.rs"}"#,
                0,
                None,
                None,
            )
            .await;
        assert!(result.is_ok());

        // Valid content item should succeed
        let result = manager
            .enqueue_unified(
                ItemType::Text,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"content":"test content","source_type":"mcp"}"#,
                0,
                None,
                None,
            )
            .await;
        assert!(result.is_ok());

        // Valid file rename item should succeed
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Rename,
                "test-tenant",
                "test-collection",
                r#"{"file_path":"/test/new.rs","old_path":"/test/old.rs"}"#,
                0,
                None,
                None,
            )
            .await;
        assert!(result.is_ok());

        // Valid doc delete item should succeed
        let result = manager
            .enqueue_unified(
                ItemType::Doc,
                UnifiedOp::Delete,
                "test-tenant",
                "test-collection",
                r#"{"document_id":"doc-123"}"#,
                0,
                None,
                None,
            )
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validation_empty_string_in_required_field() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_validation_empty_field.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        let manager = QueueManager::new(pool);
        manager.init_unified_queue().await.unwrap();

        // File with empty file_path should fail
        let result = manager
            .enqueue_unified(
                ItemType::File,
                UnifiedOp::Add,
                "test-tenant",
                "test-collection",
                r#"{"file_path":""}"#,
                0,
                None,
                None,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QueueError::MissingPayloadField { item_type, field }
            if item_type == "file" && field == "file_path"
        ));
    }

    /// Test FIFO ordering: priority_descending=true → created_at ASC (oldest first)
    #[tokio::test]
    async fn test_dequeue_fifo_ordering() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_fifo.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        apply_sql_script(&pool, include_str!("schema/watch_folders_schema.sql"))
            .await.unwrap();

        let manager = QueueManager::new(pool.clone());
        manager.init_unified_queue().await.unwrap();

        // Create an inactive project watch_folder
        sqlx::query(
            r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
               created_at, updated_at)
               VALUES ('w1', '/test', 'projects', 'tenant-a', 0,
               '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
        ).execute(&pool).await.unwrap();

        // Enqueue 3 items with staggered timestamps (old → new)
        for i in 1..=3 {
            let ts = format!("2026-01-0{}T00:00:00.000Z", i);
            sqlx::query(
                r#"INSERT INTO unified_queue
                   (queue_id, item_type, op, tenant_id, collection, priority, status,
                    branch, idempotency_key, payload_json, created_at, updated_at)
                   VALUES (?1, 'file', 'add', 'tenant-a', 'projects', 0, 'pending',
                    'main', ?2, ?3, ?4, ?4)"#,
            )
            .bind(format!("fifo-q{}", i))
            .bind(format!("key-fifo-{}", i))
            .bind(format!(r#"{{"file_path":"/test/file{}.rs"}}"#, i))
            .bind(&ts)
            .execute(&pool).await.unwrap();
        }

        // DESC direction → FIFO (oldest first)
        let items = manager
            .dequeue_unified(3, "test-worker", Some(300), None, None, Some(true))
            .await.unwrap();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].queue_id, "fifo-q1"); // oldest
        assert_eq!(items[1].queue_id, "fifo-q2");
        assert_eq!(items[2].queue_id, "fifo-q3"); // newest
    }

    /// Test LIFO ordering: priority_descending=false → created_at DESC (newest first)
    #[tokio::test]
    async fn test_dequeue_lifo_ordering() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_lifo.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        apply_sql_script(&pool, include_str!("schema/watch_folders_schema.sql"))
            .await.unwrap();

        let manager = QueueManager::new(pool.clone());
        manager.init_unified_queue().await.unwrap();

        // Create an inactive project watch_folder
        sqlx::query(
            r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active,
               created_at, updated_at)
               VALUES ('w1', '/test', 'projects', 'tenant-a', 0,
               '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z')"#,
        ).execute(&pool).await.unwrap();

        // Enqueue 3 items with staggered timestamps (old → new)
        for i in 1..=3 {
            let ts = format!("2026-01-0{}T00:00:00.000Z", i);
            sqlx::query(
                r#"INSERT INTO unified_queue
                   (queue_id, item_type, op, tenant_id, collection, priority, status,
                    branch, idempotency_key, payload_json, created_at, updated_at)
                   VALUES (?1, 'file', 'add', 'tenant-a', 'projects', 0, 'pending',
                    'main', ?2, ?3, ?4, ?4)"#,
            )
            .bind(format!("lifo-q{}", i))
            .bind(format!("key-lifo-{}", i))
            .bind(format!(r#"{{"file_path":"/test/lifo{}.rs"}}"#, i))
            .bind(&ts)
            .execute(&pool).await.unwrap();
        }

        // ASC direction → LIFO (newest first)
        let items = manager
            .dequeue_unified(3, "test-worker", Some(300), None, None, Some(false))
            .await.unwrap();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].queue_id, "lifo-q3"); // newest
        assert_eq!(items[1].queue_id, "lifo-q2");
        assert_eq!(items[2].queue_id, "lifo-q1"); // oldest
    }

    // ===== Library document payload validation =====

    #[test]
    fn test_validate_library_document_payload_valid() {
        let payload = serde_json::json!({
            "document_path": "/docs/report.pdf",
            "library_name": "internal-docs",
            "document_type": "page_based",
            "source_format": "pdf",
            "doc_id": "550e8400-e29b-41d4-a716-446655440000",
        });
        assert!(QueueManager::validate_library_document_payload(&payload).is_ok());
    }

    #[test]
    fn test_validate_library_document_payload_stream_based() {
        let payload = serde_json::json!({
            "document_path": "/books/novel.epub",
            "library_name": "ebooks",
            "document_type": "stream_based",
            "source_format": "epub",
            "doc_id": "uuid-here",
        });
        assert!(QueueManager::validate_library_document_payload(&payload).is_ok());
    }

    #[test]
    fn test_validate_library_document_payload_missing_field() {
        let payload = serde_json::json!({
            "document_path": "/docs/report.pdf",
            "library_name": "internal-docs",
            // missing document_type, source_format, doc_id
        });
        let result = QueueManager::validate_library_document_payload(&payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_library_document_payload_invalid_document_type() {
        let payload = serde_json::json!({
            "document_path": "/docs/report.pdf",
            "library_name": "internal-docs",
            "document_type": "unknown_type",
            "source_format": "pdf",
            "doc_id": "uuid-here",
        });
        let result = QueueManager::validate_library_document_payload(&payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_library_document_payload_empty_fields() {
        let payload = serde_json::json!({
            "document_path": "",
            "library_name": "docs",
            "document_type": "page_based",
            "source_format": "pdf",
            "doc_id": "uuid",
        });
        let result = QueueManager::validate_library_document_payload(&payload);
        assert!(result.is_err()); // document_path is empty
    }
}
