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
            "SELECT tf.file_id, tf.file_path, tf.branch, tf.collection,
                    wf.tenant_id
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.id
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
            let file_path: &str = row.try_get("file_path").unwrap_or("");
            if file_path.is_empty() {
                continue;
            }

            if std::path::Path::new(file_path).exists() {
                continue;
            }

            // File is missing from disk
            self.files_missing += 1;
            let tenant_id: String = row.try_get("tenant_id").unwrap_or_default();
            let branch: String = row.try_get("branch").unwrap_or_default();
            let collection: String = row.try_get("collection").unwrap_or_default();

            let payload = serde_json::json!({ "file_path": file_path });
            if let Err(e) = ctx
                .queue_manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Delete,
                    &tenant_id,
                    &collection,
                    &payload.to_string(),
                    Some(&branch),
                    None,
                )
                .await
            {
                warn!(
                    "Failed to enqueue delete for missing file {}: {}",
                    file_path, e
                );
            } else {
                info!("Enqueued delete for missing file: {}", file_path);
            }
        }

        self.offset += self.batch_size;
        MaintenanceResult::Continue
    }
}
