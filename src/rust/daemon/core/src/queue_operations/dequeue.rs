//! Queue dequeue and stale lease recovery operations.

use chrono::Duration as ChronoDuration;
use chrono::Utc;
use sqlx::Row;
use tracing::{debug, info};
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_PROJECTS};
use wqm_common::timestamps;

use crate::unified_queue_schema::{
    DestinationStatus, ItemType, QueueOperation as UnifiedOp, QueueStatus,
    UnifiedQueueItem,
};

use super::{QueueError, QueueManager, QueueResult};

impl QueueManager {
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
        let is_descending = priority_descending.unwrap_or(true);
        let priority_order = if is_descending { "DESC" } else { "ASC" };

        // Task 9: FIFO/LIFO alternation for idle processing
        let created_at_order = if is_descending { "ASC" } else { "DESC" };

        // Select queue_ids with calculated priority (Task 20)
        let queue_ids = self.select_queue_ids(
            &now_str, tenant_id, item_type, priority_order, created_at_order, batch_size,
        ).await?;

        if queue_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Update the selected items to in_progress
        self.lease_items(&queue_ids, worker_id, &lease_until_str).await?;

        // Fetch the updated items
        let mut items = self.fetch_items(&queue_ids).await?;

        // Preserve the ordering from the initial SELECT
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

    /// Select queue item IDs for dequeuing with dynamic priority ordering.
    async fn select_queue_ids(
        &self,
        now_str: &str,
        tenant_id: Option<&str>,
        item_type: Option<ItemType>,
        priority_order: &str,
        created_at_order: &str,
        batch_size: i32,
    ) -> QueueResult<Vec<String>> {
        let tenant_filter = match (tenant_id, item_type) {
            (Some(_), Some(_)) => "AND q.tenant_id = ?2 AND q.item_type = ?3",
            (Some(_), None)    => "AND q.tenant_id = ?2",
            (None, Some(_))    => "AND q.item_type = ?2",
            (None, None)       => "",
        };

        let limit_param = match (tenant_id, item_type) {
            (Some(_), Some(_)) => "?4",
            (Some(_), None) | (None, Some(_)) => "?3",
            (None, None) => "?2",
        };

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
            {tenant_filter}
            ORDER BY
                -- Tier 1: Activity priority (active project or memory = high, else low)
                CASE
                    WHEN q.collection = '{coll_memory}' THEN 1
                    WHEN q.collection = '{coll_libraries}' THEN 0
                    WHEN w.is_active = 1 THEN 1
                    ELSE 0
                END {priority_order},
                -- Tier 2: Operation ordering (subordinated to activity priority)
                CASE WHEN q.op = 'delete' THEN 10
                     WHEN q.op = 'reset' THEN 8
                     WHEN q.op = 'scan' THEN 5
                     WHEN q.op = 'update' THEN 3
                     ELSE 1
                END DESC,
                q.created_at {created_at_order}
            LIMIT {limit_param}
            "#,
            coll_projects = COLLECTION_PROJECTS,
            coll_libraries = COLLECTION_LIBRARIES,
            coll_memory = COLLECTION_RULES,
            tenant_filter = tenant_filter,
            priority_order = priority_order,
            created_at_order = created_at_order,
            limit_param = limit_param,
        );

        let result: Vec<String> = match (tenant_id, item_type) {
            (Some(tid), Some(itype)) => {
                sqlx::query_scalar::<_, String>(&query)
                    .bind(now_str)
                    .bind(tid)
                    .bind(itype.to_string())
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (Some(tid), None) => {
                sqlx::query_scalar::<_, String>(&query)
                    .bind(now_str)
                    .bind(tid)
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (None, Some(itype)) => {
                sqlx::query_scalar::<_, String>(&query)
                    .bind(now_str)
                    .bind(itype.to_string())
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
            (None, None) => {
                sqlx::query_scalar::<_, String>(&query)
                    .bind(now_str)
                    .bind(batch_size)
                    .fetch_all(&self.pool)
                    .await?
            }
        };

        Ok(result)
    }

    /// Update selected items to in_progress with a lease.
    async fn lease_items(
        &self,
        queue_ids: &[String],
        worker_id: &str,
        lease_until_str: &str,
    ) -> QueueResult<()> {
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
            .bind(lease_until_str);

        for queue_id in queue_ids {
            update_builder = update_builder.bind(queue_id);
        }

        update_builder.execute(&self.pool).await?;
        Ok(())
    }

    /// Fetch queue items by their IDs.
    async fn fetch_items(&self, queue_ids: &[String]) -> QueueResult<Vec<UnifiedQueueItem>> {
        let fetch_placeholders: Vec<String> = (1..=queue_ids.len())
            .map(|i| format!("?{}", i))
            .collect();
        let fetch_query = format!(
            "SELECT * FROM unified_queue WHERE queue_id IN ({})",
            fetch_placeholders.join(", ")
        );

        let mut fetch_builder = sqlx::query(&fetch_query);
        for queue_id in queue_ids {
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
            });
        }

        Ok(items)
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
}
