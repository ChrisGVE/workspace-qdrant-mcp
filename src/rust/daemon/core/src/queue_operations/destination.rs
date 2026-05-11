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
        let qs = DestinationStatus::parse_str(&qs_str).unwrap_or(DestinationStatus::Pending);
        let ss = DestinationStatus::parse_str(&ss_str).unwrap_or(DestinationStatus::Pending);

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

    /// Mark destination results for items that do not use the per-destination
    /// state machine.
    ///
    /// **Contract (F-010, F-056):**
    ///
    /// - **Orchestration-only items** (no `decision_json` set; e.g.
    ///   `Tenant/Scan`, `Folder/Scan`, scratchpad inserts, single-point URL/text
    ///   items, library content): both destination statuses are auto-resolved
    ///   to `Done` because the handler is responsible for the entire write and
    ///   no per-sink state was ever recorded. Returning `Ok(())` from the
    ///   handler is sufficient proof of success.
    ///
    /// - **State-machine items** (`decision_json IS NOT NULL`, set via
    ///   [`store_queue_decision`]): destination statuses are NEVER flipped by
    ///   this helper. Handlers MUST call [`update_destination_status`]
    ///   explicitly for every sink they write to (Qdrant, search DB). Pending
    ///   sinks remain `pending` so [`check_and_finalize`] keeps the item
    ///   `in_progress` and the queue processor can re-lease and retry. This
    ///   prevents the F-010 bug where a handler that only wrote to one sink
    ///   marked the other sink `done` without proof.
    ///
    /// Statuses already set to `done`, `failed`, or `in_progress` are always
    /// preserved.
    pub async fn mark_explicit_destination_results(&self, queue_id: &str) -> QueueResult<()> {
        let now = timestamps::now_utc();

        sqlx::query(
            r#"UPDATE unified_queue
               SET qdrant_status = CASE
                       WHEN decision_json IS NULL
                            AND (qdrant_status IS NULL OR qdrant_status = 'pending')
                       THEN 'done' ELSE qdrant_status END,
                   search_status = CASE
                       WHEN decision_json IS NULL
                            AND (search_status IS NULL OR search_status = 'pending')
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
