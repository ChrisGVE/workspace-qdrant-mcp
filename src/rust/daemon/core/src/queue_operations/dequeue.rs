//! Queue dequeue and stale lease recovery operations.

use chrono::Duration as ChronoDuration;
use chrono::Utc;
use sqlx::{Row, SqliteConnection};
use tracing::{debug, info};
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES};
use wqm_common::timestamps;

use crate::monitoring::metrics_core::METRICS;
use crate::unified_queue_schema::{
    DestinationStatus, ItemType, QueueOperation as UnifiedOp, QueueStatus, UnifiedQueueItem,
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
    /// * `age_promotion_warning_seconds` - Items pending longer than this get +1 priority
    ///   in the age-promotion CASE. Defaults to 300 (5 minutes) if None.
    /// * `age_promotion_critical_seconds` - Items pending longer than this get +2 priority
    ///   in the age-promotion CASE. Defaults to 900 (15 minutes) if None.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        name = "queue.dequeue",
        skip_all,
        fields(
            batch_size = batch_size,
            worker_id = %worker_id,
            tenant_id = tracing::field::Empty,
            item_type = tracing::field::Empty,
        )
    )]
    pub async fn dequeue_unified(
        &self,
        batch_size: i32,
        worker_id: &str,
        lease_duration_secs: Option<i64>,
        tenant_id: Option<&str>,
        item_type: Option<ItemType>,
        priority_descending: Option<bool>,
        age_promotion_warning_seconds: Option<u64>,
        age_promotion_critical_seconds: Option<u64>,
    ) -> QueueResult<Vec<UnifiedQueueItem>> {
        if let Some(tid) = tenant_id {
            tracing::Span::current().record("tenant_id", tid);
        }
        if let Some(it) = item_type {
            tracing::Span::current().record("item_type", tracing::field::debug(it));
        }
        let lease_duration = lease_duration_secs.unwrap_or(300);
        let lease_until = Utc::now() + ChronoDuration::seconds(lease_duration);
        let lease_until_str = timestamps::format_utc(&lease_until);
        let now_str = timestamps::now_utc();

        // Task 21: Priority direction for anti-starvation alternation
        let is_descending = priority_descending.unwrap_or(true);
        let priority_order = if is_descending { "DESC" } else { "ASC" };

        // Task 9: FIFO/LIFO alternation for idle processing
        let created_at_order = if is_descending { "ASC" } else { "DESC" };

        // Op-order is always DESC: delete always takes precedence over all other operations
        // regardless of which priority pass is running. Delete is a correctness concern
        // (stale data removal), not just a performance preference.
        let op_order = "DESC";

        // Age-based promotion thresholds (prevent tenant starvation under the
        // fairness scheduler). Items pending longer than `warning_seconds` get +1
        // in the age-promotion CASE; items pending longer than `critical_seconds`
        // get +2. Applied AFTER delete/tenant-add precedence but BEFORE the
        // project-active priority, so old low-weight scans eventually outrank
        // fresh high-weight adds from other tenants.
        let age_warning_seconds = age_promotion_warning_seconds.unwrap_or(300) as i64;
        let age_critical_seconds = age_promotion_critical_seconds.unwrap_or(900) as i64;

        // Wrap SELECT→UPDATE→SELECT in a single transaction to reduce lock churn
        let mut tx = self.pool.begin().await.map_err(QueueError::Database)?;

        // Select queue_ids with calculated priority (Task 20)
        let queue_ids = select_queue_ids(
            &mut tx,
            &now_str,
            tenant_id,
            item_type,
            priority_order,
            op_order,
            created_at_order,
            batch_size,
            age_warning_seconds,
            age_critical_seconds,
        )
        .await?;

        if queue_ids.is_empty() {
            // No items — commit (no-op) to release the transaction cleanly
            let _ = tx.commit().await;
            return Ok(Vec::new());
        }

        // Update the selected items to in_progress
        lease_items(&mut tx, &queue_ids, worker_id, &lease_until_str).await?;

        // Fetch the updated items
        let mut items = fetch_items(&mut tx, &queue_ids).await?;

        tx.commit().await.map_err(QueueError::Database)?;

        // Preserve the ordering from the initial SELECT
        {
            let id_positions: std::collections::HashMap<&str, usize> = queue_ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.as_str(), i))
                .collect();
            items.sort_by_key(|item| {
                *id_positions
                    .get(item.queue_id.as_str())
                    .unwrap_or(&usize::MAX)
            });
        }

        debug!(
            "Dequeued {} unified items for worker {}",
            items.len(),
            worker_id
        );

        for item in &items {
            METRICS.unified_queue_dequeued(&item.item_type.to_string());
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

/// Select queue item IDs for dequeuing with dynamic priority ordering.
#[allow(clippy::too_many_arguments)]
async fn select_queue_ids(
    conn: &mut SqliteConnection,
    now_str: &str,
    tenant_id: Option<&str>,
    item_type: Option<ItemType>,
    priority_order: &str,
    op_order: &str,
    created_at_order: &str,
    batch_size: i32,
    age_warning_seconds: i64,
    age_critical_seconds: i64,
) -> QueueResult<Vec<String>> {
    let query = build_dequeue_query(
        tenant_id,
        item_type,
        priority_order,
        op_order,
        created_at_order,
    );
    execute_dequeue_query(
        conn,
        &query,
        now_str,
        tenant_id,
        item_type,
        batch_size,
        age_warning_seconds,
        age_critical_seconds,
    )
    .await
}

/// Update selected items to in_progress with a lease.
async fn lease_items(
    conn: &mut SqliteConnection,
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

    update_builder.execute(&mut *conn).await?;
    Ok(())
}

/// Fetch queue items by their IDs.
async fn fetch_items(
    conn: &mut SqliteConnection,
    queue_ids: &[String],
) -> QueueResult<Vec<UnifiedQueueItem>> {
    let fetch_placeholders: Vec<String> =
        (1..=queue_ids.len()).map(|i| format!("?{}", i)).collect();
    let fetch_query = format!(
        "SELECT * FROM unified_queue WHERE queue_id IN ({})",
        fetch_placeholders.join(", ")
    );

    let mut fetch_builder = sqlx::query(&fetch_query);
    for queue_id in queue_ids {
        fetch_builder = fetch_builder.bind(queue_id);
    }

    let rows = fetch_builder.fetch_all(&mut *conn).await?;

    let mut items = Vec::with_capacity(rows.len());
    for row in rows {
        let item_type_str: String = row.try_get("item_type")?;
        let op_str: String = row.try_get("op")?;
        let status_str: String = row.try_get("status")?;

        items.push(UnifiedQueueItem {
            queue_id: row.try_get("queue_id")?,
            idempotency_key: row.try_get("idempotency_key")?,
            item_type: ItemType::parse_str(&item_type_str)
                .ok_or_else(|| QueueError::InvalidOperation(item_type_str.clone()))?,
            op: UnifiedOp::parse_str(&op_str)
                .ok_or_else(|| QueueError::InvalidOperation(op_str.clone()))?,
            tenant_id: row.try_get("tenant_id")?,
            collection: row.try_get("collection")?,
            status: QueueStatus::parse_str(&status_str)
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
            file_path: row.try_get("file_path")?,   // Task 22
            size_bytes: row.try_get("size_bytes")?, // #133 F1 (v45)
            qdrant_status: {
                let s: Option<String> = row.try_get("qdrant_status")?;
                s.and_then(|v| DestinationStatus::parse_str(&v))
            },
            search_status: {
                let s: Option<String> = row.try_get("search_status")?;
                s.and_then(|v| DestinationStatus::parse_str(&v))
            },
            decision_json: row.try_get("decision_json")?,
        });
    }

    Ok(items)
}

/// Build the dequeue SELECT query with filters substituted in.
fn build_dequeue_query(
    tenant_id: Option<&str>,
    item_type: Option<ItemType>,
    priority_order: &str,
    op_order: &str,
    created_at_order: &str,
) -> String {
    // Placeholder layout (positional binds): ?1 = now_str, then the optional
    // tenant/item filters, then the two age thresholds, then the limit — the
    // age thresholds are bound BEFORE the limit. See `execute_dequeue_query`.
    let (tenant_filter, age_warning_param, age_critical_param, limit_param) =
        match (tenant_id, item_type) {
            (Some(_), Some(_)) => (
                "AND q.tenant_id = ?2 AND q.item_type = ?3",
                "?4",
                "?5",
                "?6",
            ),
            (Some(_), None) => ("AND q.tenant_id = ?2", "?3", "?4", "?5"),
            (None, Some(_)) => ("AND q.item_type = ?2", "?3", "?4", "?5"),
            (None, None) => ("", "?2", "?3", "?4"),
        };
    format!(
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
            CASE WHEN q.op = 'delete' THEN 1
                 WHEN q.op = 'reset' THEN 1
                 ELSE 0
            END DESC,
            -- Project registrations AND their follow-up directory scan jump the
            -- line. The (Tenant, Add) registration already line-jumped (#70);
            -- promote the (Tenant, Scan) it enqueues too (#112). Without this a
            -- normal-priority (inactive) project's scan carries the lowest
            -- op-weight (1) AND loses the `is_active` tie-break, so it is starved
            -- for minutes behind active projects' file ingestion until a
            -- reconcile sweep ingests the files instead (leaving `last_scan`
            -- stale). The scan only enumerates the directory and enqueues File
            -- items — those File items still rank by `is_active` below active
            -- work, so promoting the scan gets the new project's files *queued*
            -- promptly without preempting active ingestion.
            CASE WHEN q.item_type = 'tenant' AND q.op IN ('add', 'scan') THEN 1 ELSE 0 END DESC,
            -- Age-based promotion: prevent starvation of tenants whose pending
            -- items have low op-weight by promoting old items above the
            -- project-active ranking. +1 once age >= warning threshold, +2 once
            -- age >= critical threshold. Always DESC (like delete precedence):
            -- an aged item must be promoted on BOTH the high-priority (DESC) and
            -- the anti-starvation (ASC) passes — inverting it on the ASC pass
            -- would bury exactly the items this is meant to rescue.
            CASE
                WHEN (strftime('%s','now') - strftime('%s', q.created_at))
                     >= {age_critical_param} THEN 2
                WHEN (strftime('%s','now') - strftime('%s', q.created_at))
                     >= {age_warning_param} THEN 1
                ELSE 0
            END DESC,
            CASE
                WHEN q.collection = '{coll_memory}' THEN 1
                WHEN q.collection = '{coll_libraries}' THEN 0
                WHEN w.is_active > 0 THEN 1
                ELSE 0
            END {priority_order},
            CASE WHEN q.op = 'delete' THEN 10
                 WHEN q.op = 'reset' THEN 8
                 WHEN q.op = 'add' THEN 5
                 WHEN q.op = 'update' THEN 4
                 WHEN q.op = 'rename' THEN 3
                 WHEN q.op = 'uplift' THEN 2
                 WHEN q.op = 'scan' THEN 1
                 ELSE 1
            END {op_order},
            q.created_at {created_at_order}
        LIMIT {limit_param}
        "#,
        coll_projects = COLLECTION_PROJECTS,
        coll_libraries = COLLECTION_LIBRARIES,
        coll_memory = COLLECTION_RULES,
        tenant_filter = tenant_filter,
        priority_order = priority_order,
        op_order = op_order,
        created_at_order = created_at_order,
        age_warning_param = age_warning_param,
        age_critical_param = age_critical_param,
        limit_param = limit_param,
    )
}

/// Execute the dequeue query with the appropriate bound parameters.
///
/// Binding order matches the placeholder layout produced by
/// `build_dequeue_query`: `now_str`, optional tenant_id, optional item_type,
/// `age_warning_seconds`, `age_critical_seconds`, `batch_size`.
#[allow(clippy::too_many_arguments)]
async fn execute_dequeue_query(
    conn: &mut SqliteConnection,
    query: &str,
    now_str: &str,
    tenant_id: Option<&str>,
    item_type: Option<ItemType>,
    batch_size: i32,
    age_warning_seconds: i64,
    age_critical_seconds: i64,
) -> QueueResult<Vec<String>> {
    let result = match (tenant_id, item_type) {
        (Some(tid), Some(itype)) => {
            sqlx::query_scalar::<_, String>(query)
                .bind(now_str)
                .bind(tid)
                .bind(itype.to_string())
                .bind(age_warning_seconds)
                .bind(age_critical_seconds)
                .bind(batch_size)
                .fetch_all(&mut *conn)
                .await?
        }
        (Some(tid), None) => {
            sqlx::query_scalar::<_, String>(query)
                .bind(now_str)
                .bind(tid)
                .bind(age_warning_seconds)
                .bind(age_critical_seconds)
                .bind(batch_size)
                .fetch_all(&mut *conn)
                .await?
        }
        (None, Some(itype)) => {
            sqlx::query_scalar::<_, String>(query)
                .bind(now_str)
                .bind(itype.to_string())
                .bind(age_warning_seconds)
                .bind(age_critical_seconds)
                .bind(batch_size)
                .fetch_all(&mut *conn)
                .await?
        }
        (None, None) => {
            sqlx::query_scalar::<_, String>(query)
                .bind(now_str)
                .bind(age_warning_seconds)
                .bind(age_critical_seconds)
                .bind(batch_size)
                .fetch_all(&mut *conn)
                .await?
        }
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_dequeue_query_high_priority_pass_has_op_desc() {
        // High-priority pass: op=DESC means delete(10) > add(5) > scan(1)
        let q = build_dequeue_query(None, None, "DESC", "DESC", "ASC");
        assert!(
            q.contains("END DESC"),
            "high-priority pass should have op_order=DESC"
        );
        assert!(
            q.contains("q.created_at ASC"),
            "high-priority pass should have FIFO created_at order"
        );
    }

    #[test]
    fn test_build_dequeue_query_low_priority_pass_has_op_desc() {
        // Low-priority pass: op=DESC even on the anti-starvation pass because delete
        // must always take precedence regardless of which priority direction is active.
        let q = build_dequeue_query(None, None, "ASC", "DESC", "DESC");
        assert!(
            q.contains("END DESC"),
            "low-priority pass should still have op_order=DESC (delete always first)"
        );
        assert!(
            q.contains("q.created_at DESC"),
            "low-priority pass should have LIFO created_at order"
        );
    }

    #[test]
    fn test_build_dequeue_query_tenant_filter() {
        let q = build_dequeue_query(Some("t1"), None, "DESC", "DESC", "ASC");
        assert!(q.contains("tenant_id = ?2"), "should include tenant filter");
    }

    #[test]
    fn test_build_dequeue_query_item_type_filter() {
        let q = build_dequeue_query(None, Some(ItemType::File), "DESC", "DESC", "ASC");
        assert!(
            q.contains("item_type = ?2"),
            "should include item_type filter"
        );
    }

    #[test]
    fn test_build_dequeue_query_both_filters() {
        let q = build_dequeue_query(Some("t1"), Some(ItemType::Folder), "DESC", "DESC", "ASC");
        assert!(q.contains("tenant_id = ?2"), "should include tenant filter");
        assert!(
            q.contains("item_type = ?3"),
            "should include item_type filter"
        );
        // With age-promotion thresholds added: ?4 = warning, ?5 = critical, ?6 = limit.
        assert!(
            q.contains("LIMIT ?6"),
            "limit should be ?6 with both filters + age thresholds"
        );
    }

    #[test]
    fn test_build_dequeue_query_includes_age_promotion_case() {
        // The age-promotion CASE must appear in the ORDER BY and reference both
        // threshold placeholders. No filters → warning=?2, critical=?3, limit=?4.
        let q = build_dequeue_query(None, None, "DESC", "DESC", "ASC");
        assert!(
            q.contains("strftime('%s','now')"),
            "should compute item age in seconds"
        );
        assert!(
            q.contains("strftime('%s', q.created_at)"),
            "should subtract created_at to get age"
        );
        assert!(
            q.contains(">= ?3 THEN 2"),
            "critical threshold (?3) bumps priority to +2"
        );
        assert!(
            q.contains(">= ?2 THEN 1"),
            "warning threshold (?2) bumps priority to +1"
        );
    }

    #[test]
    fn test_build_dequeue_query_age_case_placed_before_active_case() {
        // The age-promotion CASE must appear BEFORE the active-vs-inactive CASE
        // (so age can override the active ranking once thresholds are crossed)
        // and AFTER the delete CASE (delete always wins).
        let q = build_dequeue_query(None, None, "DESC", "DESC", "ASC");
        let age_idx = q
            .find(">= ?3 THEN 2")
            .expect("age-promotion CASE should be present");
        let active_idx = q
            .find("w.is_active > 0")
            .expect("active CASE should be present");
        assert!(
            age_idx < active_idx,
            "age-promotion CASE must come before active-vs-inactive CASE"
        );
        let delete_idx = q
            .find("WHEN q.op = 'delete' THEN 1")
            .expect("delete CASE should be present");
        assert!(
            delete_idx < age_idx,
            "delete CASE must take precedence over age promotion"
        );
    }

    #[test]
    fn test_build_dequeue_query_age_promotion_always_desc() {
        // The age-promotion CASE is ALWAYS `END DESC`, independent of the
        // priority_order fairness flip — an aged item must be promoted on both
        // the high-priority (DESC) and the anti-starvation (ASC) passes.
        // Inverting it on the ASC pass would bury the items it rescues.
        let q_desc = build_dequeue_query(None, None, "DESC", "DESC", "ASC");
        let q_asc = build_dequeue_query(None, None, "ASC", "DESC", "DESC");
        for q in [&q_desc, &q_asc] {
            let section = q
                .split(">= ?2 THEN 1")
                .nth(1)
                .expect("split should yield section after warning case");
            assert!(
                section
                    .trim_start()
                    .starts_with("ELSE 0\n            END DESC"),
                "age CASE must be `ELSE 0 ... END DESC` on BOTH passes, got: {}",
                &section[..section.len().min(40)]
            );
        }
    }
}
