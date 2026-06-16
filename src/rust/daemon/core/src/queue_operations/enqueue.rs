//! Queue initialization and item enqueue operations.

use tracing::{debug, error, info};

use crate::monitoring::metrics_core::METRICS;
use crate::unified_queue_schema::{
    generate_unified_idempotency_key, ItemType, QueueOperation as UnifiedOp,
    CREATE_UNIFIED_QUEUE_INDEXES_SQL, CREATE_UNIFIED_QUEUE_SQL,
};

use super::{QueueError, QueueManager, QueueResult};

/// Metadata JSON key carrying the W3C `traceparent` across the enqueue->dequeue
/// queue hop (PRD B2). Stored in the queue's `metadata` column; the queue
/// processor extracts it to LINK the per-item processing span back to the
/// producer span.
const TRACEPARENT_METADATA_KEY: &str = "wqm_traceparent";

/// Merge a W3C `traceparent` into the item's `metadata` JSON under
/// [`TRACEPARENT_METADATA_KEY`] without clobbering existing keys.
///
/// Returns the metadata string to store:
/// - `tp == None` -> the incoming metadata is returned unchanged (`None`).
/// - the incoming metadata fails to parse as a JSON object -> returned
///   unchanged (we never corrupt caller metadata to carry a trace link).
/// - otherwise -> the parsed object with `wqm_traceparent` inserted, re-serialized.
///
/// Returning `None` means "store the original `metadata` argument as-is" at the
/// call site, so an absent trace context costs nothing.
fn merge_traceparent(metadata: Option<&str>, tp: Option<&str>) -> Option<String> {
    let tp = tp?;
    let raw = metadata.unwrap_or("{}");
    let mut value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let obj = value.as_object_mut()?;
    obj.insert(
        TRACEPARENT_METADATA_KEY.to_string(),
        serde_json::Value::String(tp.to_string()),
    );
    serde_json::to_string(&value).ok()
}

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

        let idempotency_key =
            generate_unified_idempotency_key(item_type, op, tenant_id, collection, payload_json)
                .map_err(|e| QueueError::InvalidOperation(e.to_string()))?;

        let branch = branch.unwrap_or("main");
        // B2: thread the current trace context through the queue hop via the
        // metadata column so the eventual processing span links back here.
        // Costs nothing when tracing is off (`current_traceparent` -> None).
        let merged_metadata = merge_traceparent(
            metadata,
            crate::tracing_otel::current_traceparent().as_deref(),
        );
        let metadata = merged_metadata.as_deref().or(metadata).unwrap_or("{}");
        let file_path = Self::extract_file_path(item_type, &payload);
        let size_bytes = Self::extract_size_bytes(item_type, &payload);

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
                size_bytes,
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

    /// Extract `size_bytes` for the queue row (#133 F1, schema v45).
    ///
    /// File items: the payload's `size_bytes` when present, else a
    /// `std::fs::metadata(path).len()` stat-fallback (the #121 pattern), so a
    /// path that omitted its size still records pending bytes for drain
    /// estimation. Non-file items (rules, folder scans, content, url) carry no
    /// byte count and return `None` — the drain SUM average-imputes NULL rows.
    fn extract_size_bytes(item_type: ItemType, payload: &serde_json::Value) -> Option<i64> {
        if item_type != ItemType::File {
            return None;
        }
        if let Some(size) = payload.get("size_bytes").and_then(|v| v.as_i64()) {
            return Some(size);
        }
        payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as i64)
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

            let idempotency_key = generate_unified_idempotency_key(
                item_type,
                op,
                tenant_id,
                collection,
                payload_json,
            )
            .map_err(|e| QueueError::InvalidOperation(e.to_string()))?;
            let file_path = Self::extract_file_path(item_type, &payload);
            let size_bytes = Self::extract_size_bytes(item_type, &payload);

            let result = sqlx::query(
                r#"
                INSERT OR IGNORE INTO unified_queue (
                    item_type, op, tenant_id, collection,
                    branch, payload_json, metadata, idempotency_key, file_path, size_bytes
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
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
            .bind(size_bytes)
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
        size_bytes: Option<i64>,
    ) -> QueueResult<(bool, u64)> {
        let result = sqlx::query(
            r#"
            INSERT OR IGNORE INTO unified_queue (
                item_type, op, tenant_id, collection,
                branch, payload_json, metadata, idempotency_key, file_path, size_bytes
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
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
        .bind(size_bytes)
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

#[cfg(test)]
mod traceparent_tests {
    use super::*;

    const TP: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    #[test]
    fn none_traceparent_leaves_metadata_unchanged() {
        // No trace context -> caller stores the original metadata as-is.
        assert_eq!(merge_traceparent(Some(r#"{"k":"v"}"#), None), None);
        assert_eq!(merge_traceparent(None, None), None);
    }

    #[test]
    fn adds_traceparent_to_empty_metadata() {
        let out = merge_traceparent(None, Some(TP)).expect("merge produces metadata");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v[TRACEPARENT_METADATA_KEY], serde_json::json!(TP));
    }

    #[test]
    fn preserves_existing_keys_and_adds_traceparent() {
        let out = merge_traceparent(Some(r#"{"source":"watcher","n":3}"#), Some(TP))
            .expect("merge produces metadata");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["source"], serde_json::json!("watcher"));
        assert_eq!(v["n"], serde_json::json!(3));
        assert_eq!(v[TRACEPARENT_METADATA_KEY], serde_json::json!(TP));
    }

    #[test]
    fn non_object_metadata_is_handled_gracefully() {
        // A JSON array (non-object) cannot carry the key -> return None so the
        // caller keeps the original metadata rather than corrupting it.
        assert_eq!(merge_traceparent(Some("[1,2,3]"), Some(TP)), None);
    }

    #[test]
    fn invalid_metadata_is_handled_gracefully() {
        // Unparseable metadata -> None (keep original, never panic).
        assert_eq!(merge_traceparent(Some("not json"), Some(TP)), None);
    }
}
