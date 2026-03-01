//! Queue item deletion, purging, and failure marking.

use chrono::{Duration as ChronoDuration, Utc};
use sqlx::Row;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use crate::metrics::METRICS;

use super::{QueueError, QueueManager, QueueResult};

impl QueueManager {
    /// Delete a unified queue item after successful processing
    ///
    /// Per docs/specs/04-write-path.md:
    /// "On success: DELETE items from queue"
    ///
    /// This is the correct method for handling successfully processed items.
    /// Use this instead of mark_unified_done.
    pub async fn delete_unified_item(&self, queue_id: &str) -> QueueResult<bool> {
        let result = sqlx::query("DELETE FROM unified_queue WHERE queue_id = ?1")
            .bind(queue_id)
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() > 0;

        if deleted {
            debug!("Deleted unified item after successful processing: {}", queue_id);
            METRICS.queue_item_processed("unified", "deleted", 0.0);
        } else {
            warn!("Failed to delete unified item: {} (not found)", queue_id);
        }

        Ok(deleted)
    }

    /// Purge all pending and in-progress queue items for a tenant.
    ///
    /// Called as the first step of tenant deletion to prevent stale file ops
    /// from being processed after the tenant's data is removed.
    /// Excludes the delete item itself (identified by `exclude_queue_id`).
    ///
    /// Returns the number of items purged.
    pub async fn purge_pending_for_tenant(
        &self,
        tenant_id: &str,
        exclude_queue_id: &str,
    ) -> QueueResult<u64> {
        let result = sqlx::query(
            r#"DELETE FROM unified_queue
               WHERE tenant_id = ?1
               AND queue_id != ?2
               AND status IN ('pending', 'in_progress')"#,
        )
        .bind(tenant_id)
        .bind(exclude_queue_id)
        .execute(&self.pool)
        .await?;

        let purged = result.rows_affected();
        if purged > 0 {
            info!(
                "Purged {} pending/in-progress queue items for tenant={}",
                purged, tenant_id
            );
        }

        Ok(purged)
    }

    /// Mark a unified queue item as failed
    ///
    /// If `permanent` is true, skips retry logic and marks as failed immediately.
    /// Otherwise, if retries remain, increments retry_count, sets exponential
    /// backoff delay via `lease_until`, and resets to pending.
    /// If max retries exceeded, sets status to 'failed'.
    ///
    /// Backoff schedule: 60s * 2^retry_count, capped at 1 hour, with 10% jitter.
    ///
    /// Returns true if the item will be retried, false if permanently failed.
    pub async fn mark_unified_failed(
        &self,
        queue_id: &str,
        error_message: &str,
        permanent: bool,
        max_retries: i32,
    ) -> QueueResult<bool> {
        // Get current retry count
        let row = sqlx::query(
            "SELECT retry_count FROM unified_queue WHERE queue_id = ?1"
        )
            .bind(queue_id)
            .fetch_optional(&self.pool)
            .await?;

        if let Some(row) = row {
            let retry_count: i32 = row.try_get("retry_count")?;
            let new_retry_count = retry_count + 1;

            if !permanent && new_retry_count < max_retries {
                // Can retry - reset to pending with incremented retry count
                // Apply exponential backoff: 60s * 2^retry_count, capped at 3600s
                let base_delay_secs = 60.0_f64;
                let delay_secs = (base_delay_secs * 2.0_f64.powi(retry_count)).min(3600.0);
                // Add 10% jitter to prevent thundering herd
                let jitter = delay_secs * 0.1 * rand::random::<f64>();
                let total_delay = delay_secs + jitter;
                let retry_after = Utc::now() + ChronoDuration::seconds(total_delay as i64);
                let retry_after_str = timestamps::format_utc(&retry_after);

                let query = r#"
                    UPDATE unified_queue
                    SET status = 'pending',
                        retry_count = ?1,
                        error_message = ?2,
                        last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        lease_until = ?3,
                        worker_id = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE queue_id = ?4
                "#;

                sqlx::query(query)
                    .bind(new_retry_count)
                    .bind(error_message)
                    .bind(&retry_after_str)
                    .bind(queue_id)
                    .execute(&self.pool)
                    .await?;

                info!(
                    "Unified item {} failed, will retry ({}/{}) after {:.0}s backoff: {}",
                    queue_id, new_retry_count, max_retries, total_delay, error_message
                );

                Ok(true)
            } else {
                // Permanent error or max retries exceeded - mark as permanently failed
                let reason = if permanent { "permanent error" } else { "max retries exceeded" };

                let query = r#"
                    UPDATE unified_queue
                    SET status = 'failed',
                        retry_count = ?1,
                        error_message = ?2,
                        last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        lease_until = NULL,
                        worker_id = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE queue_id = ?3
                "#;

                sqlx::query(query)
                    .bind(new_retry_count)
                    .bind(error_message)
                    .bind(queue_id)
                    .execute(&self.pool)
                    .await?;

                warn!(
                    "Unified item {} permanently failed ({}, attempt {}/{}): {}",
                    queue_id, reason, new_retry_count, max_retries, error_message
                );

                METRICS.queue_item_processed("unified", "failure", 0.0);
                METRICS.ingestion_error(reason);

                Ok(false)
            }
        } else {
            warn!("Unified queue item not found: {}", queue_id);
            Err(QueueError::NotFound(queue_id.to_string()))
        }
    }
}
