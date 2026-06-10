//! Queue initialization and item enqueue operations.

use tracing::{debug, error, info};

use crate::monitoring::metrics_core::METRICS;
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
    #[tracing::instrument(
        name = "queue.enqueue",
        skip_all,
        fields(item_type = ?item_type, op = ?op, tenant_id = %tenant_id, collection = %collection)
    )]
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

        let key_payload = Self::idempotency_payload_json(item_type, payload_json);
        let idempotency_key =
            generate_unified_idempotency_key(item_type, op, tenant_id, collection, &key_payload)
                .map_err(|e| QueueError::InvalidOperation(e.to_string()))?;

        let branch = branch.unwrap_or("main");
        let metadata = metadata.unwrap_or("{}");
        let file_path = Self::extract_file_path(item_type, &payload);

        let (is_new, _) = self
            .insert_queue_item(
                item_type,
                op,
                tenant_id,
                collection,
                branch,
                payload_json,
                metadata,
                &idempotency_key,
                &file_path,
            )
            .await?;

        let queue_id = self
            .resolve_queue_id(
                is_new,
                &idempotency_key,
                item_type,
                op,
                tenant_id,
                collection,
                branch,
                &file_path,
            )
            .await?;

        self.post_enqueue_actions(
            is_new,
            &queue_id,
            item_type,
            op,
            tenant_id,
            collection,
            &idempotency_key,
        )
        .await?;

        if is_new {
            METRICS.unified_queue_enqueued("daemon");
        }

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

        let payload: serde_json::Value = serde_json::from_str(payload_json).map_err(|e| {
            error!("Queue validation failed: invalid payload JSON - {}", e);
            QueueError::InvalidPayloadJson(e.to_string())
        })?;

        Self::validate_payload_for_type(item_type, op, &payload)?;
        Ok((tenant_id, collection, payload))
    }

    /// Extract file_path from payload for per-file deduplication (file items only).
    fn extract_file_path(item_type: ItemType, payload: &serde_json::Value) -> Option<String> {
        if item_type == ItemType::File {
            payload
                .get("file_path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Build the payload JSON used for the idempotency key.
    ///
    /// `Folder` scan payloads carry a volatile `last_scan` baseline that is
    /// bumped to "now" after every scan pass. Hashing it into the idempotency
    /// key gave every re-scan of the same directory a fresh key, and folder
    /// items are NOT covered by the partial `file_path` UNIQUE dedup index
    /// (`WHERE file_path IS NOT NULL`, folders store NULL) — so identical scans
    /// piled up unbounded and the queue never drained (the index never settled).
    /// Strip `last_scan` so the key is stable per
    /// `(item_type, op, tenant, collection, folder_path, scan-params)`; an
    /// already-pending scan of the same directory then dedups via
    /// `INSERT OR IGNORE`. The STORED payload keeps `last_scan` for mtime pruning.
    fn idempotency_payload_json(
        item_type: ItemType,
        payload_json: &str,
    ) -> std::borrow::Cow<'_, str> {
        if item_type != ItemType::Folder {
            return std::borrow::Cow::Borrowed(payload_json);
        }
        match serde_json::from_str::<serde_json::Value>(payload_json) {
            Ok(mut value) => {
                let changed = value
                    .as_object_mut()
                    .map(|obj| obj.remove("last_scan").is_some())
                    .unwrap_or(false);
                if !changed {
                    return std::borrow::Cow::Borrowed(payload_json);
                }
                serde_json::to_string(&value)
                    .map(std::borrow::Cow::Owned)
                    .unwrap_or(std::borrow::Cow::Borrowed(payload_json))
            }
            Err(_) => std::borrow::Cow::Borrowed(payload_json),
        }
    }

    /// Bulk-enqueue `(item_type, op, tenant_id, collection, payload_json)` tuples
    /// inside a single SQLite transaction.
    ///
    /// Used by the startup ignore reconciler to avoid one lock acquisition per
    /// row on large projects (issue #59). Duplicates (same idempotency key or
    /// file path) are silently skipped via `INSERT OR IGNORE`; the returned
    /// count is the number of rows actually inserted.
    ///
    /// All items must share the same `item_type`, `op`, `tenant_id`, and
    /// `collection`. Enforcing that at the call site keeps the idempotency
    /// key / payload validation cheap.
    pub async fn enqueue_unified_batch(
        &self,
        item_type: ItemType,
        op: UnifiedOp,
        tenant_id: &str,
        collection: &str,
        payload_jsons: &[String],
        branch: Option<&str>,
    ) -> QueueResult<u64> {
        if payload_jsons.is_empty() {
            return Ok(0);
        }

        let tenant_id = tenant_id.trim();
        if tenant_id.is_empty() {
            return Err(QueueError::EmptyTenantId);
        }
        let collection = collection.trim();
        if collection.is_empty() {
            return Err(QueueError::EmptyCollection);
        }
        let branch = branch.unwrap_or("main");
        let metadata = "{}";

        let mut tx = self.pool.begin().await?;
        let mut inserted: u64 = 0;

        for payload_json in payload_jsons {
            let payload: serde_json::Value = serde_json::from_str(payload_json)
                .map_err(|e| QueueError::InvalidPayloadJson(e.to_string()))?;
            Self::validate_payload_for_type(item_type, op, &payload)?;

            let key_payload = Self::idempotency_payload_json(item_type, payload_json);
            let idempotency_key = generate_unified_idempotency_key(
                item_type,
                op,
                tenant_id,
                collection,
                &key_payload,
            )
            .map_err(|e| QueueError::InvalidOperation(e.to_string()))?;
            let file_path = Self::extract_file_path(item_type, &payload);

            let result = sqlx::query(
                r#"
                INSERT OR IGNORE INTO unified_queue (
                    item_type, op, tenant_id, collection,
                    branch, payload_json, metadata, idempotency_key, file_path
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            )
            .bind(item_type.to_string())
            .bind(op.to_string())
            .bind(tenant_id)
            .bind(collection)
            .bind(branch)
            .bind(payload_json)
            .bind(metadata)
            .bind(&idempotency_key)
            .bind(&file_path)
            .execute(&mut *tx)
            .await?;

            if result.rows_affected() > 0 {
                inserted += 1;
            }
        }

        tx.commit().await?;

        if inserted > 0 {
            METRICS.unified_queue_enqueued("daemon");
            debug!(
                "Bulk-enqueued {} items (type={}, op={}, collection={})",
                inserted, item_type, op, collection
            );
        }
        Ok(inserted)
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
        let result = sqlx::query(
            r#"
            INSERT OR IGNORE INTO unified_queue (
                item_type, op, tenant_id, collection,
                branch, payload_json, metadata, idempotency_key, file_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#,
        )
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
    ///
    /// Dedupe lookup precedence:
    ///   1. Match `idempotency_key` (covers byte-identical payload re-enqueue).
    ///   2. Match the composite uniqueness key
    ///      `(tenant_id, branch, collection, item_type, op, file_path)` —
    ///      enforced by the v36 partial UNIQUE index. The composite scope
    ///      prevents the same file path from colliding across tenants,
    ///      branches, collections, item types, or operations (F-009).
    #[allow(clippy::too_many_arguments)]
    async fn resolve_queue_id(
        &self,
        is_new: bool,
        idempotency_key: &str,
        item_type: ItemType,
        op: UnifiedOp,
        tenant_id: &str,
        collection: &str,
        branch: &str,
        file_path: &Option<String>,
    ) -> QueueResult<String> {
        if is_new {
            return Ok(sqlx::query_scalar(
                "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1",
            )
            .bind(idempotency_key)
            .fetch_one(&self.pool)
            .await?);
        }

        if let Some(id) = sqlx::query_scalar::<_, String>(
            "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?1",
        )
        .bind(idempotency_key)
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(id);
        }

        if let Some(ref fp) = file_path {
            return Ok(sqlx::query_scalar(
                "SELECT queue_id FROM unified_queue \
                 WHERE tenant_id = ?1 AND branch = ?2 AND collection = ?3 \
                       AND item_type = ?4 AND op = ?5 AND file_path = ?6",
            )
            .bind(tenant_id)
            .bind(branch)
            .bind(collection)
            .bind(item_type.to_string())
            .bind(op.to_string())
            .bind(fp)
            .fetch_one(&self.pool)
            .await?);
        }
        Err(QueueError::InternalError(
            "INSERT OR IGNORE returned 0 rows but no matching idempotency_key or composite key found".to_string()
        ))
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
            debug!(
                "Enqueued unified item: {} (type={}, op={}, collection={})",
                queue_id, item_type, op, collection
            );

            if op == UnifiedOp::Delete
                && (item_type == ItemType::Tenant || item_type == ItemType::Collection)
            {
                let purged = sqlx::query(
                    r#"
                    DELETE FROM unified_queue
                    WHERE tenant_id = ?1 AND queue_id != ?2
                      AND status IN ('pending', 'in_progress')
                "#,
                )
                .bind(tenant_id)
                .bind(queue_id)
                .execute(&self.pool)
                .await?;

                if purged.rows_affected() > 0 {
                    info!(
                        "Delete cascade: purged {} pending/in_progress items for tenant {}",
                        purged.rows_affected(),
                        tenant_id
                    );
                }
            }
        } else {
            sqlx::query(
                "UPDATE unified_queue SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE queue_id = ?1"
            )
                .bind(queue_id)
                .execute(&self.pool)
                .await?;

            debug!(
                "Unified item already exists: {} (idempotency_key={})",
                queue_id, idempotency_key
            );
        }
        Ok(())
    }
}
