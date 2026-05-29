//! Per-destination status management for queue items.

use sqlx::{Row, SqliteConnection};
use wqm_common::timestamps;

use crate::unified_queue_schema::{DestinationStatus, QueueStatus};

use super::{QueueError, QueueManager, QueueResult};

/// SQL that auto-resolves pending destination sinks to `done` for
/// orchestration-only items (`decision_json IS NULL`). State-machine items
/// (`decision_json IS NOT NULL`) are left untouched. Shared by
/// [`QueueManager::mark_explicit_destination_results`] and the atomic
/// success-path finalizer so behavior stays identical. `?1` = updated_at,
/// `?2` = queue_id.
const MARK_EXPLICIT_DESTINATION_RESULTS_SQL: &str = r#"UPDATE unified_queue
       SET qdrant_status = CASE
               WHEN decision_json IS NULL
                    AND (qdrant_status IS NULL OR qdrant_status = 'pending')
               THEN 'done' ELSE qdrant_status END,
           search_status = CASE
               WHEN decision_json IS NULL
                    AND (search_status IS NULL OR search_status = 'pending')
               THEN 'done' ELSE search_status END,
           updated_at = ?1
       WHERE queue_id = ?2"#;

/// Auto-resolve pending sinks for orchestration-only items on the given
/// connection. See [`MARK_EXPLICIT_DESTINATION_RESULTS_SQL`] for semantics.
async fn mark_explicit_destination_results_on(
    conn: &mut SqliteConnection,
    queue_id: &str,
) -> QueueResult<()> {
    let now = timestamps::now_utc();
    sqlx::query(MARK_EXPLICIT_DESTINATION_RESULTS_SQL)
        .bind(&now)
        .bind(queue_id)
        .execute(conn)
        .await?;
    Ok(())
}

/// Read both destination statuses, resolve the overall [`QueueStatus`], and
/// persist it (when resolved to `Done`/`Failed`) on the given connection.
/// Shared by [`QueueManager::check_and_finalize`] and the atomic success-path
/// finalizer so behavior stays identical.
async fn check_and_finalize_on(
    conn: &mut SqliteConnection,
    queue_id: &str,
) -> QueueResult<QueueStatus> {
    let row =
        sqlx::query("SELECT qdrant_status, search_status FROM unified_queue WHERE queue_id = ?1")
            .bind(queue_id)
            .fetch_optional(&mut *conn)
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
        sqlx::query("UPDATE unified_queue SET status = ?1, updated_at = ?2 WHERE queue_id = ?3")
            .bind(overall.to_string())
            .bind(&now)
            .bind(queue_id)
            .execute(&mut *conn)
            .await?;
    }

    Ok(overall)
}

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
        let mut conn = self.pool.acquire().await?;
        check_and_finalize_on(&mut conn, queue_id).await
    }

    /// Atomically resolve a queue item on the handler success path (F-009).
    ///
    /// Combines [`mark_explicit_destination_results`](Self::mark_explicit_destination_results)
    /// and [`check_and_finalize`](Self::check_and_finalize) into a single
    /// transaction so the finalize read cannot observe a concurrent mid-flight
    /// `failed` write committed between the two operations. Both run against the
    /// same connection inside one `BEGIN`/`COMMIT`, so they see a consistent
    /// snapshot and other writers serialize against this transaction.
    ///
    /// Semantics are identical to calling the two methods in sequence — only the
    /// atomicity guarantee is added. Returns the resolved overall [`QueueStatus`].
    pub async fn finalize_after_success(&self, queue_id: &str) -> QueueResult<QueueStatus> {
        let mut tx = self.pool.begin().await?;
        mark_explicit_destination_results_on(&mut tx, queue_id).await?;
        let overall = check_and_finalize_on(&mut tx, queue_id).await?;
        tx.commit().await.map_err(QueueError::Database)?;
        Ok(overall)
    }

    /// Read the per-destination statuses for a queue item.
    ///
    /// Returns `(qdrant_status, search_status)` as raw strings (NULL → None).
    /// Used by batch_processing to enrich the error_message when
    /// `check_and_finalize` returns `Failed` on a success-path handler return.
    pub async fn read_destination_statuses(
        &self,
        queue_id: &str,
    ) -> QueueResult<(Option<String>, Option<String>)> {
        let row = sqlx::query_as::<_, (Option<String>, Option<String>)>(
            "SELECT qdrant_status, search_status FROM unified_queue WHERE queue_id = ?1",
        )
        .bind(queue_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.unwrap_or((None, None)))
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
        let mut conn = self.pool.acquire().await?;
        mark_explicit_destination_results_on(&mut conn, queue_id).await
    }
}
