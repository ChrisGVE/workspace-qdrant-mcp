//! Filesystem reconciliation — detect tracked files that no longer exist on
//! disk and enqueue delete operations to clean them up.

use async_trait::async_trait;
use sqlx::Row;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::idle::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use crate::idle::IdleState;
use crate::unified_queue_schema::{ItemType, QueueOperation};

/// Batch-checks tracked files against the filesystem.
///
/// Runs in `FullIdle` or `QdrantDownIdle` (only needs disk + SQLite).
/// Missing files get a delete operation enqueued so the normal pipeline
/// handles Qdrant cleanup when available.
pub struct FilesystemReconcileTask {
    batch_size: i64,
    offset: i64,
    total_checked: u64,
    files_missing: u64,
}

impl FilesystemReconcileTask {
    pub fn new() -> Self {
        Self {
            batch_size: 100,
            offset: 0,
            total_checked: 0,
            files_missing: 0,
        }
    }
}

#[async_trait]
impl MaintenanceTask for FilesystemReconcileTask {
    fn name(&self) -> &str {
        "filesystem_reconcile"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        60
    }

    fn cooldown_secs(&self) -> u64 {
        1800 // 30 minutes
    }

    fn reset(&mut self) {
        self.offset = 0;
        self.total_checked = 0;
        self.files_missing = 0;
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let rows = sqlx::query(
            "SELECT tf.file_id, tf.relative_path, tf.branch, tf.collection,
                    wf.tenant_id, wf.path AS watch_path
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
             ORDER BY tf.file_id
             LIMIT ?1 OFFSET ?2",
        )
        .bind(self.batch_size)
        .bind(self.offset)
        .fetch_all(ctx.pool)
        .await;

        let rows = match rows {
            Ok(r) => r,
            Err(e) => {
                warn!("Filesystem reconcile query failed: {} — will retry", e);
                return MaintenanceResult::Yielded;
            }
        };

        if rows.is_empty() {
            if self.files_missing > 0 {
                info!(
                    "Filesystem reconcile complete: checked={}, missing={}",
                    self.total_checked, self.files_missing
                );
            } else {
                debug!(
                    "Filesystem reconcile complete: checked={}, all present",
                    self.total_checked
                );
            }
            return MaintenanceResult::Done;
        }

        for row in &rows {
            if cancel.is_cancelled() {
                return MaintenanceResult::Yielded;
            }

            self.total_checked += 1;
            let relative_path: &str = row.try_get("relative_path").unwrap_or("");
            let watch_path: &str = row.try_get("watch_path").unwrap_or("");
            if relative_path.is_empty() || watch_path.is_empty() {
                continue;
            }

            // Reconstruct the absolute filesystem path from the canonical
            // watch-folder root and the validated relative path.
            let abs_path = std::path::Path::new(watch_path).join(relative_path);
            if abs_path.exists() {
                continue;
            }

            // File is missing from disk
            self.files_missing += 1;
            let tenant_id: String = row.try_get("tenant_id").unwrap_or_default();
            let branch: String = row.try_get("branch").unwrap_or_default();
            let collection: String = row.try_get("collection").unwrap_or_default();

            let abs_path_str = abs_path.to_string_lossy();
            // FilePayload.file_path is a validating RelativePath — ship the
            // watch-root-relative form straight from tracked_files. The
            // absolute join above exists only for the on-disk check and the
            // logs; enqueueing it would fail the consumer's parse as a
            // permanent InvalidPayload.
            let payload = build_missing_file_delete_payload(relative_path);
            if let Err(e) = ctx
                .queue_manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Delete,
                    &tenant_id,
                    &collection,
                    &payload,
                    Some(&branch),
                    None,
                )
                .await
            {
                warn!(
                    "Failed to enqueue delete for missing file {}: {}",
                    abs_path_str, e
                );
            } else {
                info!("Enqueued delete for missing file: {}", abs_path_str);
            }
        }

        self.offset += self.batch_size;
        MaintenanceResult::Continue
    }
}

/// Build the `file|delete` payload for a tracked file missing from disk.
///
/// The payload must round-trip through the consumer's validating
/// `FilePayload` deserialization, so it carries the watch-root-relative
/// path — never the absolute join used for the existence check.
fn build_missing_file_delete_payload(relative_path: &str) -> String {
    serde_json::json!({ "file_path": relative_path }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_queue_schema::FilePayload;

    /// Regression: this task used to enqueue the ABSOLUTE joined path,
    /// which the consumer's validating `RelativePath` deserialization
    /// rejects as a permanent `InvalidPayload` (the same failure class as
    /// the ignore-reconciliation poison items).
    #[test]
    fn delete_payload_parses_as_file_payload_with_relative_path() {
        let payload = build_missing_file_delete_payload("src/lib/missing.rs");
        let parsed: FilePayload = serde_json::from_str(&payload).unwrap();
        assert_eq!(parsed.file_path.as_str(), "src/lib/missing.rs");
    }

    #[test]
    fn absolute_path_payload_is_rejected_by_consumer_parse() {
        // Documents WHY the payload must be relative: the absolute form
        // (what this task shipped before) cannot parse.
        let payload = build_missing_file_delete_payload("/root/proj/src/missing.rs");
        assert!(serde_json::from_str::<FilePayload>(&payload).is_err());
    }
}
