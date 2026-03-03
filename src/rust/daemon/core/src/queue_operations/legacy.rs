//! Legacy queue operations and collection metadata queries.

use chrono::{DateTime, Utc};
use sqlx::Row;
use std::collections::HashMap;
use tracing::{debug, warn};
use wqm_common::timestamps;

use crate::metrics::METRICS;
use crate::queue_types::MissingTool;
use crate::unified_queue_schema::{
    FilePayload, ItemType, QueueOperation as UnifiedOp,
};

use super::{
    CollectionMetadata, CollectionType, MissingMetadataItem,
    QueueError, QueueManager, QueueResult,
};

impl QueueManager {
    /// Mark a file as having an error
    pub async fn mark_error(
        &self,
        file_path: &str,
        error_type: &str,
        error_message: &str,
        error_details: Option<&HashMap<String, serde_json::Value>>,
        max_retries: i32,
    ) -> QueueResult<(bool, i64)> {
        let mut tx = self.pool.begin().await?;

        let error_details_json = error_details
            .map(|d| serde_json::to_string(d))
            .transpose()?;

        let error_message_id =
            insert_error_message(&mut tx, error_type, error_message, error_details_json, file_path)
                .await?;

        let row = sqlx::query(
            "SELECT retry_count FROM ingestion_queue WHERE file_absolute_path = ?1",
        )
        .bind(file_path)
        .fetch_optional(&mut *tx)
        .await?;

        if let Some(row) = row {
            let current_retry_count: i32 = row.try_get("retry_count")?;
            let new_retry_count = current_retry_count + 1;
            let will_retry = new_retry_count < max_retries;
            update_ingestion_queue_on_error(
                &mut tx, file_path, error_type, error_message_id,
                new_retry_count, max_retries, will_retry,
            )
            .await?;
            tx.commit().await?;
            Ok((will_retry, error_message_id))
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

            // Enqueue to unified_queue
            match self.enqueue_unified(
                ItemType::File,
                unified_op,
                &tenant_id,
                &collection,
                &payload_json,
                Some(&branch),
                None,
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

                    tracing::info!(
                        "Retrying item from missing_metadata_queue: {} -> unified_queue {}",
                        file_path, new_queue_id
                    );
                    Ok(true)
                }
                Err(e) => {
                    tracing::error!("Failed to enqueue retry item to unified_queue: {}", e);
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
}

/// Insert an error record into the `messages` table and return its rowid.
async fn insert_error_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    error_type: &str,
    error_message: &str,
    error_details_json: Option<String>,
    file_path: &str,
) -> QueueResult<i64> {
    let result = sqlx::query(
        r#"INSERT INTO messages (
            error_type, error_message, error_details,
            file_path, collection_name
        ) VALUES (?1, ?2, ?3, ?4, (
            SELECT collection_name FROM ingestion_queue
            WHERE file_absolute_path = ?5
        ))"#,
    )
    .bind(error_type)
    .bind(error_message)
    .bind(error_details_json)
    .bind(file_path)
    .bind(file_path)
    .execute(&mut **tx)
    .await?;
    Ok(result.last_insert_rowid())
}

/// Update (or delete) the ingestion_queue row after recording an error.
async fn update_ingestion_queue_on_error(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    file_path: &str,
    error_type: &str,
    error_message_id: i64,
    new_retry_count: i32,
    max_retries: i32,
    will_retry: bool,
) -> QueueResult<()> {
    if will_retry {
        sqlx::query(
            "UPDATE ingestion_queue SET retry_count = ?1, error_message_id = ?2
             WHERE file_absolute_path = ?3",
        )
        .bind(new_retry_count)
        .bind(error_message_id)
        .bind(file_path)
        .execute(&mut **tx)
        .await?;

        METRICS.ingestion_error(error_type);
        debug!("Updated error for {}: retry {}/{}", file_path, new_retry_count, max_retries);
    } else {
        sqlx::query("DELETE FROM ingestion_queue WHERE file_absolute_path = ?1")
            .bind(file_path)
            .execute(&mut **tx)
            .await?;

        METRICS.queue_item_processed("normal", "failure", 0.0);
        METRICS.ingestion_error(error_type);
        warn!("Max retries ({}) reached for {}, removing from queue", max_retries, file_path);
    }
    Ok(())
}
