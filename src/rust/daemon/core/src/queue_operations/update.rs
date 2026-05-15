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
        let retry_after_str =
            timestamps::format_utc(&(chrono::Utc::now() + ChronoDuration::seconds(delay_secs)));

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

    /// Reset eligible failed transient items back to pending for retry.
    ///
    /// Called periodically from the processing loop idle path (default: every hour).
    /// Only items prefixed `[transient_` are eligible — permanent failures are left
    /// as-is. Items that have been resurrected `max_resurrections` times are promoted
    /// to `[permanent_exhausted]` and stop being resurrected.
    ///
    /// Returns `(resurrected, exhausted)` counts.
    pub async fn resurrect_failed_transient(
        &self,
        max_resurrections: i32,
    ) -> QueueResult<(u64, u64)> {
        // Fetch eligible items to check resurrection_count in metadata
        let rows = sqlx::query(
            r#"SELECT queue_id, metadata, error_message
               FROM unified_queue
               WHERE status = 'failed'
                 AND error_message LIKE '[transient_%'"#,
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            debug!("Resurrection pass: no transient failed items to reset");
            return Ok((0, 0));
        }

        let mut resurrected: u64 = 0;
        let mut exhausted: u64 = 0;

        for row in &rows {
            let queue_id: &str = row.try_get("queue_id")?;
            let metadata_str: &str = row.try_get("metadata").unwrap_or("{}");
            let error_message: &str = row.try_get("error_message").unwrap_or("");

            let mut metadata: serde_json::Value =
                serde_json::from_str(metadata_str).unwrap_or_else(|_| serde_json::json!({}));
            let count = metadata
                .get("resurrection_count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;

            if count >= max_resurrections {
                // Promote to permanent — stop resurrecting
                let exhausted_msg = format!("[permanent_exhausted] {}", error_message);
                sqlx::query(
                    r#"UPDATE unified_queue
                       SET error_message = ?1,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                       WHERE queue_id = ?2"#,
                )
                .bind(&exhausted_msg)
                .bind(queue_id)
                .execute(&self.pool)
                .await?;
                exhausted += 1;
                warn!(
                    "Item {} exceeded max resurrections ({}/{}), marked permanent_exhausted",
                    queue_id, count, max_resurrections
                );
            } else {
                // Resurrect with incremented count
                metadata["resurrection_count"] = serde_json::json!(count + 1);
                let new_metadata = serde_json::to_string(&metadata).unwrap_or_default();

                sqlx::query(
                    r#"UPDATE unified_queue
                       SET status = 'pending', retry_count = 0,
                           lease_until = NULL, worker_id = NULL,
                           qdrant_status = NULL, search_status = NULL,
                           metadata = ?1,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                       WHERE queue_id = ?2"#,
                )
                .bind(&new_metadata)
                .bind(queue_id)
                .execute(&self.pool)
                .await?;
                resurrected += 1;
            }
        }

        if resurrected > 0 {
            info!(
                "Resurrected {} failed transient item(s) for retry",
                resurrected
            );
        }
        if exhausted > 0 {
            warn!(
                "Marked {} item(s) as permanently exhausted (>{} resurrections)",
                exhausted, max_resurrections
            );
        }
        Ok((resurrected, exhausted))
    }

    /// Delete a unified queue item after successful processing.
    ///
    /// Per docs/specs/04-write-path.md:
    /// "On success: DELETE items from queue"
    ///
    /// This is the correct method for handling successfully processed items.
    /// Use this instead of mark_unified_done.
    ///
    /// Post-completion side-effects (F-020, F-036):
    ///
    /// - **F-020**: For any completed file op, clears `needs_reconcile` on the
    ///   matching `tracked_files` row. The flag must only be cleared after the
    ///   repair intent has been durably executed, not at enqueue time.
    ///
    /// - **F-036**: For completed Delete ops specifically, also removes the
    ///   `tracked_files` row itself. Once the Delete handler has cleaned up
    ///   Qdrant/FTS/graph state, the tracking record is no longer needed and
    ///   must be removed so that the next startup reconciliation pass does not
    ///   re-enqueue another Delete for the same file.
    pub async fn delete_unified_item(&self, queue_id: &str) -> QueueResult<bool> {
        // Look up op + file_path before deleting so we can apply post-completion
        // side-effects for F-020 and F-036.
        let row = sqlx::query(
            "SELECT op, item_type, file_path, tenant_id FROM unified_queue WHERE queue_id = ?1",
        )
        .bind(queue_id)
        .fetch_optional(&self.pool)
        .await?;

        let (op, item_type, file_path, tenant_id): (
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        ) = match row {
            Some(r) => (
                r.try_get("op").ok(),
                r.try_get("item_type").ok(),
                r.try_get("file_path").ok(),
                r.try_get("tenant_id").ok(),
            ),
            None => (None, None, None, None),
        };

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

            if let (Some(ref abs_path), Some(ref tid)) = (file_path, tenant_id) {
                let is_file = item_type.as_deref() == Some("file");
                let is_delete = op.as_deref() == Some("delete");

                // F-036: remove the tracked_files row once a Delete op completes.
                // The handler has already cleaned up Qdrant/FTS/graph state; keep
                // the row any longer would cause the next reconciliation pass to
                // re-enqueue another Delete for the same file.
                if is_file && is_delete {
                    if let Err(e) = self.remove_tracked_file_row(abs_path, tid).await {
                        warn!(
                            "Failed to remove tracked_files row after Delete {} completed: {}",
                            queue_id, e
                        );
                    }
                } else if is_file {
                    // F-020: for non-Delete file ops, clear needs_reconcile only.
                    // The tracked_files row should remain (file still exists).
                    if let Err(e) = self.clear_needs_reconcile_for_file(abs_path, tid).await {
                        // Non-fatal: the flag will be picked up on the next pass.
                        warn!(
                            "Failed to clear needs_reconcile after item {} completed: {}",
                            queue_id, e
                        );
                    }
                }
            }
        } else {
            warn!("Failed to delete unified item: {} (not found)", queue_id);
        }

        Ok(deleted)
    }

    /// Remove the `tracked_files` row for `abs_file_path` after a Delete op
    /// completes (F-036).
    ///
    /// The row is identified by joining `watch_folders.path || '/' ||
    /// tracked_files.relative_path` against the absolute path stored in the queue,
    /// scoped to `tenant_id` to prevent cross-tenant mutations when two tenants
    /// share an identical absolute path under different watch-folder roots.
    async fn remove_tracked_file_row(
        &self,
        abs_file_path: &str,
        tenant_id: &str,
    ) -> QueueResult<()> {
        let rows = sqlx::query(
            "DELETE FROM tracked_files \
             WHERE file_id IN ( \
                 SELECT tf.file_id \
                 FROM tracked_files tf \
                 JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
                 WHERE wf.path || '/' || tf.relative_path = ?1 \
                   AND wf.tenant_id = ?2 \
             )",
        )
        .bind(abs_file_path)
        .bind(tenant_id)
        .execute(&self.pool)
        .await?
        .rows_affected();

        if rows > 0 {
            debug!(
                "Removed {} tracked_files row(s) after Delete completed: {}",
                rows, abs_file_path
            );
        }
        Ok(())
    }

    /// Clear `needs_reconcile` on the `tracked_files` row whose absolute path
    /// matches `abs_file_path` (F-020).
    ///
    /// The absolute path is reconstructed as `watch_folders.path || '/' ||
    /// tracked_files.relative_path` and compared against the stored queue
    /// `file_path`, scoped to `tenant_id` to prevent cross-tenant mutations
    /// when two tenants share an identical absolute path under different
    /// watch-folder roots.
    async fn clear_needs_reconcile_for_file(
        &self,
        abs_file_path: &str,
        tenant_id: &str,
    ) -> QueueResult<()> {
        let now = timestamps::now_utc();
        let rows = sqlx::query(
            "UPDATE tracked_files \
             SET needs_reconcile = 0, reconcile_reason = NULL, updated_at = ?1 \
             WHERE needs_reconcile = 1 \
               AND file_id IN ( \
                   SELECT tf.file_id \
                   FROM tracked_files tf \
                   JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
                   WHERE wf.path || '/' || tf.relative_path = ?2 \
                     AND wf.tenant_id = ?3 \
               )",
        )
        .bind(&now)
        .bind(abs_file_path)
        .bind(tenant_id)
        .execute(&self.pool)
        .await?
        .rows_affected();

        if rows > 0 {
            debug!(
                "Cleared needs_reconcile for {} after queue completion: {}",
                rows, abs_file_path
            );
        }
        Ok(())
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
