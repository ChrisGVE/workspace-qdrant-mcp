//! Per-destination status management for queue items.

use sqlx::Row;
use wqm_common::timestamps;

use crate::unified_queue_schema::{DestinationStatus, QueueStatus};

use super::{QueueError, QueueManager, QueueResult};

impl QueueManager {
    /// Store a QueueDecision and set per-destination statuses on a queue item.
    ///
    /// Called after computing the decision but before executing Qdrant/search writes.
    /// The decision persists across daemon restarts for retry-safe execution.
    pub async fn store_queue_decision(
        &self,
        queue_id: &str,
        decision: &wqm_common::queue_types::QueueDecision,
    ) -> QueueResult<()> {
        let decision_json = serde_json::to_string(decision).map_err(|e| {
            QueueError::InvalidOperation(format!("Failed to serialize decision: {}", e))
        })?;
        let now = timestamps::now_utc();

        sqlx::query(
            r#"
            UPDATE unified_queue
            SET decision_json = ?1,
                qdrant_status = 'pending',
                search_status = 'pending',
                updated_at = ?2
            WHERE queue_id = ?3
            "#,
        )
        .bind(&decision_json)
        .bind(&now)
        .bind(queue_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Check and finalize a queue item based on per-destination statuses.
    ///
    /// If both qdrant_status and search_status are 'done', marks the item as 'done'.
    /// If either is 'failed', marks the item as 'failed'.
    /// Returns the resolved overall QueueStatus.
    pub async fn check_and_finalize(&self, queue_id: &str) -> QueueResult<QueueStatus> {
        let row = sqlx::query(
            "SELECT qdrant_status, search_status FROM unified_queue WHERE queue_id = ?1",
        )
        .bind(queue_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else {
            return Err(QueueError::InvalidOperation(format!(
                "Queue item not found: {}",
                queue_id
            )));
        };

        let qs_str: String = row
            .try_get::<String, _>("qdrant_status")
            .unwrap_or_else(|_| "pending".to_string());
        let ss_str: String = row
            .try_get::<String, _>("search_status")
            .unwrap_or_else(|_| "pending".to_string());
        let qs = DestinationStatus::from_str(&qs_str).unwrap_or(DestinationStatus::Pending);
        let ss = DestinationStatus::from_str(&ss_str).unwrap_or(DestinationStatus::Pending);

        let overall = if qs == DestinationStatus::Done && ss == DestinationStatus::Done {
            QueueStatus::Done
        } else if qs == DestinationStatus::Failed || ss == DestinationStatus::Failed {
            QueueStatus::Failed
        } else {
            QueueStatus::InProgress
        };

        // Update overall status if resolved
        if overall == QueueStatus::Done || overall == QueueStatus::Failed {
            let now = timestamps::now_utc();
            sqlx::query(
                "UPDATE unified_queue SET status = ?1, updated_at = ?2 WHERE queue_id = ?3",
            )
            .bind(overall.to_string())
            .bind(&now)
            .bind(queue_id)
            .execute(&self.pool)
            .await?;
        }

        Ok(overall)
    }

    /// Update the per-destination status for a queue item.
    ///
    /// Called after each destination (Qdrant, search DB) completes or fails.
    /// When both destinations are complete, the queue item overall status
    /// is derived via check_completion().
    pub async fn update_destination_status(
        &self,
        queue_id: &str,
        destination: &str,
        status: DestinationStatus,
    ) -> QueueResult<()> {
        let column = match destination {
            "qdrant" => "qdrant_status",
            "search" => "search_status",
            _ => {
                return Err(QueueError::InvalidOperation(format!(
                    "Unknown destination: {}",
                    destination
                )))
            }
        };

        let now = timestamps::now_utc();
        let status_str = status.to_string();

        let query = format!(
            "UPDATE unified_queue SET {} = ?1, updated_at = ?2 WHERE queue_id = ?3",
            column
        );

        sqlx::query(&query)
            .bind(&status_str)
            .bind(&now)
            .bind(queue_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Ensure any destination status still pending is resolved to 'done'.
    ///
    /// Called after process_item returns Ok(()) for handlers that don't use the
    /// per-destination state machine (orchestration-only items like Tenant/Scan,
    /// Folder/Scan, content items, URL items, etc.). This prevents items from
    /// getting stuck in the queue when neither qdrant_status nor search_status
    /// was explicitly set by the handler.
    ///
    /// Statuses already set to 'done', 'failed', or 'in_progress' are preserved.
    pub async fn ensure_destinations_resolved(&self, queue_id: &str) -> QueueResult<()> {
        let now = timestamps::now_utc();

        sqlx::query(
            r#"UPDATE unified_queue
               SET qdrant_status = CASE
                       WHEN qdrant_status IS NULL OR qdrant_status = 'pending'
                       THEN 'done' ELSE qdrant_status END,
                   search_status = CASE
                       WHEN search_status IS NULL OR search_status = 'pending'
                       THEN 'done' ELSE search_status END,
                   updated_at = ?1
               WHERE queue_id = ?2"#,
        )
        .bind(&now)
        .bind(queue_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}
