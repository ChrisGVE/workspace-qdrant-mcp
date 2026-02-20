//! Queue statistics, depth queries, oldest item retrieval, and cleanup.

use sqlx::Row;
use std::collections::HashMap;
use tracing::{debug, info};
use wqm_common::timestamps;

use crate::unified_queue_schema::{
    DestinationStatus, ItemType, QueueOperation as UnifiedOp, QueueStatus,
    UnifiedQueueItem, UnifiedQueueStats,
};

use super::{QueueError, QueueManager, QueueResult};

impl QueueManager {
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
                    qdrant_status: {
                        let s: Option<String> = row.try_get("qdrant_status")?;
                        s.and_then(|v| DestinationStatus::from_str(&v))
                    },
                    search_status: {
                        let s: Option<String> = row.try_get("search_status")?;
                        s.and_then(|v| DestinationStatus::from_str(&v))
                    },
                    decision_json: row.try_get("decision_json")?,
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
}
