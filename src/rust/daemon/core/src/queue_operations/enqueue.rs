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
        let (tenant_id, collection, payload) =
            Self::validate_enqueue_params(tenant_id, collection, payload_json, item_type, op)?;

        let idempotency_key = generate_unified_idempotency_key(
            item_type, op, tenant_id, collection, payload_json,
        ).map_err(|e| QueueError::InvalidOperation(e.to_string()))?;

        let branch = branch.unwrap_or("main");
        let metadata = metadata.unwrap_or("{}");
        let file_path = Self::extract_file_path(item_type, &payload);

        let (is_new, _) = self.insert_queue_item(
            item_type, op, tenant_id, collection, branch,
            payload_json, metadata, &idempotency_key, &file_path,
        ).await?;

        let queue_id = self.resolve_queue_id(
            is_new, &idempotency_key, &file_path,
        ).await?;

        self.post_enqueue_actions(
            is_new, &queue_id, item_type, op, tenant_id, collection, &idempotency_key,
        ).await?;

        Ok((queue_id, is_new))
    }

    /// Validate and normalize enqueue parameters.
    fn validate_enqueue_params<'a>(
        tenant_id: &'a str,
        collection: &'a str,
        payload_json: &str,
        item_type: ItemType,
        op: UnifiedOp,
    ) -> QueueResult<(&'a str, &'a str, serde_json::Value)> {
        let tenant_id = tenant_id.trim();
        if tenant_id.is_empty() {
            error!("Queue validation failed: tenant_id is empty or whitespace-only");
            return Err(QueueError::EmptyTenantId);
        }

        let collection = collection.trim();
        if collection.is_empty() {
            error!("Queue validation failed: collection is empty or whitespace-only");
            return Err(QueueError::EmptyCollection);
        }

        let payload: serde_json::Value = serde_json::from_str(payload_json)
            .map_err(|e| {
                error!("Queue validation failed: invalid payload JSON - {}", e);
                QueueError::InvalidPayloadJson(e.to_string())
            })?;

        Self::validate_payload_for_type(item_type, op, &payload)?;
        Ok((tenant_id, collection, payload))
    }

    /// Extract file_path from payload for per-file deduplication (file items only).
    fn extract_file_path(item_type: ItemType, payload: &serde_json::Value) -> Option<String> {
        if item_type == ItemType::File {
            payload.get("file_path").and_then(|v| v.as_str()).map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Execute the INSERT OR IGNORE and return whether a new row was inserted.
    async fn insert_queue_item(
        &self,
        item_type: ItemType,
        op: UnifiedOp,
        tenant_id: &str,
        collection: &str,
        branch: &str,
        payload_json: &str,
        metadata: &str,
        idempotency_key: &str,
        file_path: &Option<String>,
    ) -> QueueResult<(bool, u64)> {
        let result = sqlx::query(r#"
            INSERT OR IGNORE INTO unified_queue (
                item_type, op, tenant_id, collection,
                branch, payload_json, metadata, idempotency_key, file_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#)
            .bind(item_type.to_string())
            .bind(op.to_string())
            .bind(tenant_id)
            .bind(collection)
            .bind(branch)
            .bind(payload_json)
            .bind(metadata)
            .bind(idempotency_key)
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        Ok((result.rows_affected() > 0, result.rows_affected()))
    }

    /// Resolve the queue_id after INSERT OR IGNORE (handles both new and duplicate cases).
    async fn resolve_queue_id(
        &self,
        is_new: bool,
        idempotency_key: &str,
        file_path: &Option<String>,
    ) -> QueueResult<String> {
        if is_new {
            return Ok(sqlx::query_scalar(
                "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1"
            )
                .bind(idempotency_key)
                .fetch_one(&self.pool)
                .await?);
        }

        // INSERT was ignored -- try idempotency_key first, then file_path
        if let Some(id) = sqlx::query_scalar::<_, String>(
            "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1"
        )
            .bind(idempotency_key)
            .fetch_optional(&self.pool)
            .await?
        {
            return Ok(id);
        }

        if let Some(ref fp) = file_path {
            Ok(sqlx::query_scalar(
                "SELECT queue_id FROM unified_queue WHERE file_path = ?1"
            )
                .bind(fp)
                .fetch_one(&self.pool)
                .await?)
        } else {
            Err(QueueError::InternalError(
                "INSERT OR IGNORE returned 0 rows but no matching idempotency_key or file_path found".to_string()
            ))
        }
    }

    /// Handle post-enqueue actions: cascade purge for deletes, or timestamp update for duplicates.
    async fn post_enqueue_actions(
        &self,
        is_new: bool,
        queue_id: &str,
        item_type: ItemType,
        op: UnifiedOp,
        tenant_id: &str,
        collection: &str,
        idempotency_key: &str,
    ) -> QueueResult<()> {
        if is_new {
            debug!("Enqueued unified item: {} (type={}, op={}, collection={})",
                queue_id, item_type, op, collection);

            if op == UnifiedOp::Delete && (item_type == ItemType::Tenant || item_type == ItemType::Collection) {
                let purged = sqlx::query(r#"
                    DELETE FROM unified_queue
                    WHERE tenant_id = ?1 AND queue_id != ?2
                      AND status IN ('pending', 'in_progress')
                "#)
                    .bind(tenant_id)
                    .bind(queue_id)
                    .execute(&self.pool)
                    .await?;

                if purged.rows_affected() > 0 {
                    info!("Delete cascade: purged {} pending/in_progress items for tenant {}",
                        purged.rows_affected(), tenant_id);
                }
            }
        } else {
            sqlx::query(
                "UPDATE unified_queue SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE queue_id = ?1"
            )
                .bind(queue_id)
                .execute(&self.pool)
                .await?;

            debug!("Unified item already exists: {} (idempotency_key={})", queue_id, idempotency_key);
        }
        Ok(())
    }
}
