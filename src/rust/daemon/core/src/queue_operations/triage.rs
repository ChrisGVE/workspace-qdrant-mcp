//! Failed item triage — periodically examines failed queue items and applies
//! intelligent recovery: drop unsalvageable items, skip items awaiting
//! Qdrant recovery, and report statistics.

use sqlx::Row;
use tracing::{debug, info, warn};

use super::{QueueManager, QueueResult};

/// Statistics from a triage pass.
#[derive(Debug, Default)]
pub struct TriageStats {
    /// Items examined
    pub examined: u64,
    /// Items dropped (delete ops with no tracked points, stale files, etc.)
    pub dropped: u64,
    /// Items left in place (awaiting recovery or not actionable)
    pub skipped: u64,
}

impl QueueManager {
    /// Examine failed items and drop those that are not salvageable.
    ///
    /// Called periodically (default: every 5 minutes) from the processing loop.
    ///
    /// Triage categories:
    /// - **Delete with no tracked points**: The delete is effectively done — drop it.
    /// - **Delete for never-tracked file**: No record in tracked_files — drop it.
    /// - **Add/update for missing file**: File no longer exists on disk — drop it.
    /// - **Permanent exhausted**: Already handled by resurrection — skip.
    /// - **Awaiting recovery**: Qdrant circuit open — skip.
    pub async fn triage_failed_items(&self) -> QueueResult<TriageStats> {
        let rows = sqlx::query(
            r#"SELECT queue_id, item_type, op, file_path, tenant_id, collection, error_message
               FROM unified_queue
               WHERE status = 'failed'
               LIMIT 100"#,
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(TriageStats::default());
        }

        let mut stats = TriageStats {
            examined: rows.len() as u64,
            ..Default::default()
        };

        for row in &rows {
            let queue_id: &str = row.try_get("queue_id")?;
            let item_type: &str = row.try_get("item_type")?;
            let op: &str = row.try_get("op")?;
            let file_path: Option<&str> = row.try_get("file_path").ok();
            let tenant_id: &str = row.try_get("tenant_id")?;
            let collection: &str = row.try_get("collection")?;
            let error_message: &str = row.try_get("error_message")?;

            // Skip items already marked as permanently exhausted
            if error_message.starts_with("[permanent_exhausted]")
                || error_message.starts_with("[permanent_data]")
            {
                stats.skipped += 1;
                continue;
            }

            // Only triage file items — other types need manual attention
            if item_type != "file" {
                stats.skipped += 1;
                continue;
            }

            let should_drop = match op {
                "delete" => {
                    self.should_drop_failed_delete(file_path, tenant_id, collection)
                        .await
                }
                "add" | "update" => {
                    self.should_drop_failed_add_update(file_path, tenant_id, collection)
                        .await
                }
                _ => false,
            };

            if should_drop {
                match sqlx::query("DELETE FROM unified_queue WHERE queue_id = ?1")
                    .bind(queue_id)
                    .execute(&self.pool)
                    .await
                {
                    Ok(_) => {
                        info!(
                            "Triage: dropped unsalvageable {} {} item {} (file={:?})",
                            item_type, op, queue_id, file_path
                        );
                        stats.dropped += 1;
                    }
                    Err(e) => {
                        warn!("Triage: failed to drop item {}: {}", queue_id, e);
                        stats.skipped += 1;
                    }
                }
            } else {
                stats.skipped += 1;
            }
        }

        if stats.dropped > 0 {
            info!(
                "Triage pass: examined={}, dropped={}, skipped={}",
                stats.examined, stats.dropped, stats.skipped
            );
        } else {
            debug!("Triage pass: examined={}, nothing to drop", stats.examined);
        }

        Ok(stats)
    }

    /// Check if a failed delete item should be dropped.
    ///
    /// A delete is droppable when the file has no tracked points in the database
    /// (meaning the delete is effectively already complete) or was never tracked.
    ///
    /// Post-T7: `file_path` in the queue is the relative path anchored to the
    /// owning `watch_folders.path`. The join matches
    /// `tf.relative_path = file_path` scoped to `(tenant_id, collection)`.
    async fn should_drop_failed_delete(
        &self,
        file_path: Option<&str>,
        tenant_id: &str,
        collection: &str,
    ) -> bool {
        let Some(path) = file_path else {
            return true; // No file_path — nothing to delete
        };

        let tracked = sqlx::query(
            "SELECT tf.file_id \
             FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE tf.relative_path = ?1 \
               AND wf.tenant_id = ?2 \
               AND wf.collection = ?3 \
             LIMIT 1",
        )
        .bind(path)
        .bind(tenant_id)
        .bind(collection)
        .fetch_optional(&self.pool)
        .await;

        match tracked {
            Ok(Some(row)) => {
                // File is tracked — check if it has any Qdrant chunks
                let file_id: i64 = match row.try_get("file_id") {
                    Ok(id) => id,
                    Err(e) => {
                        warn!(
                            "Triage: failed to read file_id for {}: {} — skipping",
                            path, e
                        );
                        return false;
                    }
                };
                let chunk_count = match sqlx::query(
                    "SELECT COUNT(*) as cnt FROM qdrant_chunks WHERE file_id = ?1",
                )
                .bind(file_id)
                .fetch_one(&self.pool)
                .await
                {
                    Ok(r) => r.try_get::<i64, _>("cnt").unwrap_or(0),
                    Err(e) => {
                        warn!(
                            "Triage: failed to count qdrant_chunks for file_id={}: {} — skipping",
                            file_id, e
                        );
                        return false;
                    }
                };

                if chunk_count == 0 {
                    debug!(
                        "Triage: delete for {} has no qdrant_chunks — droppable",
                        path
                    );
                    return true;
                }
                false // Has chunks — needs Qdrant to process
            }
            Ok(None) => {
                debug!(
                    "Triage: delete for {} not in tracked_files — droppable",
                    path
                );
                true // Never tracked
            }
            Err(e) => {
                warn!(
                    "Triage: DB error checking tracked_files for {:?}: {}",
                    path, e
                );
                false
            }
        }
    }

    /// Check if a failed add/update item should be dropped.
    ///
    /// An add/update is droppable when the file no longer exists on disk —
    /// the watcher will re-enqueue it as a delete if needed.
    ///
    /// Post-T7: `file_path` in the queue is relative. Resolve the owning
    /// `watch_folders.path` via `(tenant_id, collection)` to reconstruct the
    /// absolute path for the existence check.
    async fn should_drop_failed_add_update(
        &self,
        file_path: Option<&str>,
        tenant_id: &str,
        collection: &str,
    ) -> bool {
        let Some(path) = file_path else {
            return false; // No file path — can't check
        };

        let watch_path: Option<String> = match sqlx::query_scalar(
            "SELECT path FROM watch_folders \
             WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
        )
        .bind(tenant_id)
        .bind(collection)
        .fetch_optional(&self.pool)
        .await
        {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    "Triage: failed to look up watch_folder for tenant={}, collection={}: {} — skipping",
                    tenant_id, collection, e
                );
                return false;
            }
        };

        let Some(root) = watch_path else {
            debug!(
                "Triage: no watch_folder for tenant={}, collection={} — leaving add/update for {} alone",
                tenant_id, collection, path
            );
            return false;
        };

        let abs = std::path::Path::new(&root).join(path);
        if !abs.exists() {
            debug!(
                "Triage: add/update for {} (root={}) — file no longer exists, droppable",
                path, root
            );
            return true;
        }
        false
    }
}
