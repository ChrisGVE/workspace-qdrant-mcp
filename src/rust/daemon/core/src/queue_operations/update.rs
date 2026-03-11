//! Queue item deletion, purging, and failure marking.

use chrono::{Duration as ChronoDuration, Utc};
use sqlx::Row;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use crate::metrics::METRICS;

use super::{QueueError, QueueManager, QueueResult};

impl QueueManager {
    /// Re-lease a unified queue item back to pending without incrementing retry_count.
    ///
    /// Used when the embedding subsystem is temporarily unavailable: the item is
    /// parked for `delay_secs` seconds without burning its retry budget.
    pub async fn re_lease_item(&self, queue_id: &str, delay_secs: i64) -> QueueResult<()> {
        let retry_after_str = timestamps::format_utc(
            &(chrono::Utc::now() + ChronoDuration::seconds(delay_secs)),
        );

        let rows = sqlx::query(
            r#"
            UPDATE unified_queue
            SET status = 'pending',
                lease_until = ?1,
                worker_id = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE queue_id = ?2
            "#,
        )
        .bind(&retry_after_str)
        .bind(queue_id)
        .execute(&self.pool)
        .await?
        .rows_affected();

        if rows == 0 {
            warn!("re_lease_item: queue item not found: {}", queue_id);
        } else {
            debug!(
                "Re-leased item {} for {}s (subsystem unavailable)",
                queue_id, delay_secs
            );
        }
        Ok(())
    }

    /// Reset all failed items whose error_message indicates a transient failure
    /// back to pending so they can be retried.
    ///
    /// Called periodically from the processing loop idle path (default: every hour).
    /// Only items prefixed `[transient_` are eligible — permanent failures are left
    /// as-is.
    ///
    /// Returns the number of items resurrected.
    pub async fn resurrect_failed_transient(&self) -> QueueResult<u64> {
        let result = sqlx::query(
            r#"
            UPDATE unified_queue
            SET status        = 'pending',
                retry_count   = 0,
                lease_until   = NULL,
                worker_id     = NULL,
                qdrant_status = NULL,
                search_status = NULL,
                updated_at    = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE status = 'failed'
              AND error_message LIKE '[transient_%'
            "#,
        )
        .execute(&self.pool)
        .await?;

        let count = result.rows_affected();
        if count > 0 {
            info!(
                "Resurrected {} failed transient item(s) for retry",
                count
            );
        } else {
            debug!("Resurrection pass: no transient failed items to reset");
        }
        Ok(count)
    }

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
            debug!(
                "Deleted unified item after successful processing: {}",
                queue_id
            );
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
        let row = sqlx::query("SELECT retry_count FROM unified_queue WHERE queue_id = ?1")
            .bind(queue_id)
            .fetch_optional(&self.pool)
            .await?;

        let Some(row) = row else {
            warn!("Unified queue item not found: {}", queue_id);
            return Err(QueueError::NotFound(queue_id.to_string()));
        };

        let retry_count: i32 = row.try_get("retry_count")?;
        let new_retry_count = retry_count + 1;

        if !permanent && new_retry_count < max_retries {
            mark_unified_retry(
                &self.pool,
                queue_id,
                error_message,
                retry_count,
                new_retry_count,
                max_retries,
            )
            .await
        } else {
            mark_unified_permanent(
                &self.pool,
                queue_id,
                error_message,
                permanent,
                new_retry_count,
                max_retries,
            )
            .await
        }
    }
}

/// Reset a unified queue item to pending with exponential backoff for retry.
async fn mark_unified_retry(
    pool: &sqlx::SqlitePool,
    queue_id: &str,
    error_message: &str,
    retry_count: i32,
    new_retry_count: i32,
    max_retries: i32,
) -> QueueResult<bool> {
    let delay_secs = (60.0_f64 * 2.0_f64.powi(retry_count)).min(3600.0);
    let jitter = delay_secs * 0.1 * rand::random::<f64>();
    let total_delay = delay_secs + jitter;
    let retry_after_str =
        timestamps::format_utc(&(Utc::now() + ChronoDuration::seconds(total_delay as i64)));

    sqlx::query(
        r#"
        UPDATE unified_queue
        SET status = 'pending', retry_count = ?1, error_message = ?2,
            last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
            lease_until = ?3, worker_id = NULL,
            qdrant_status = NULL, search_status = NULL,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE queue_id = ?4
    "#,
    )
    .bind(new_retry_count)
    .bind(error_message)
    .bind(&retry_after_str)
    .bind(queue_id)
    .execute(pool)
    .await?;

    info!(
        "Unified item {} failed, will retry ({}/{}) after {:.0}s backoff: {}",
        queue_id, new_retry_count, max_retries, total_delay, error_message
    );
    Ok(true)
}

/// Mark a unified queue item as permanently failed.
async fn mark_unified_permanent(
    pool: &sqlx::SqlitePool,
    queue_id: &str,
    error_message: &str,
    permanent: bool,
    new_retry_count: i32,
    max_retries: i32,
) -> QueueResult<bool> {
    let reason = if permanent {
        "permanent error"
    } else {
        "max retries exceeded"
    };

    sqlx::query(
        r#"
        UPDATE unified_queue
        SET status = 'failed', retry_count = ?1, error_message = ?2,
            last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
            lease_until = NULL, worker_id = NULL,
            qdrant_status = NULL, search_status = NULL,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE queue_id = ?3
    "#,
    )
    .bind(new_retry_count)
    .bind(error_message)
    .bind(queue_id)
    .execute(pool)
    .await?;

    warn!(
        "Unified item {} permanently failed ({}, attempt {}/{}): {}",
        queue_id, reason, new_retry_count, max_retries, error_message
    );
    METRICS.queue_item_processed("unified", "failure", 0.0);
    METRICS.ingestion_error(reason);
    Ok(false)
}
