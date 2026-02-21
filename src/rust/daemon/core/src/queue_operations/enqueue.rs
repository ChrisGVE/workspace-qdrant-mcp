//! Queue initialization and item enqueue operations.

use tracing::{debug, error, info};

use crate::unified_queue_schema::{
    generate_unified_idempotency_key, ItemType, QueueOperation as UnifiedOp,
    CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL,
};

use super::{QueueError, QueueManager, QueueResult};

impl QueueManager {
    // ========================================================================
    // Unified Queue Operations (Task 37.21-37.29)
    // ========================================================================

    /// Initialize the unified_queue table schema
    ///
    /// Creates the table and indexes if they don't exist.
    pub async fn init_unified_queue(&self) -> QueueResult<()> {
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
    /// * `branch` - Git branch (default: main)
    /// * `metadata` - Optional additional metadata as JSON
    pub async fn enqueue_unified(
        &self,
        item_type: ItemType,
        op: UnifiedOp,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
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
                item_type, op, tenant_id, collection,
                branch, payload_json, metadata, idempotency_key, file_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#;

        let result = sqlx::query(insert_query)
            .bind(item_type.to_string())
            .bind(op.to_string())
            .bind(tenant_id)
            .bind(collection)
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
            // INSERT was ignored -- try idempotency_key first
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

            // Task 44: Cascade purge -- when a delete is enqueued, immediately
            // remove all other pending/in_progress items for the same tenant.
            // This prevents a delete from waiting behind hundreds of add/update items.
            if op == UnifiedOp::Delete && (item_type == ItemType::Tenant || item_type == ItemType::Collection) {
                let purged = sqlx::query(
                    r#"
                    DELETE FROM unified_queue
                    WHERE tenant_id = ?1
                      AND queue_id != ?2
                      AND status IN ('pending', 'in_progress')
                    "#
                )
                    .bind(tenant_id)
                    .bind(&queue_id)
                    .execute(&self.pool)
                    .await?;

                let purged_count = purged.rows_affected();
                if purged_count > 0 {
                    info!(
                        "Delete cascade: purged {} pending/in_progress items for tenant {}",
                        purged_count, tenant_id
                    );
                }
            }
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
}
