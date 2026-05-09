//! Reconciliation of tracked_files flagged with needs_reconcile=1.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use crate::queue_operations::QueueManager;
use crate::tracked_files_schema;
use crate::unified_queue_schema::QueueOperation;

use super::queue::enqueue_file_op;
use super::types::FullRecoveryStats;

/// Process tracked_files flagged with needs_reconcile=1.
///
/// For each flagged file, look up its watch_folder to get routing info,
/// then re-queue it for ingestion.
pub(super) async fn reconcile_flagged_files(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    stats: &mut FullRecoveryStats,
) {
    let flagged = match tracked_files_schema::get_files_needing_reconcile(pool).await {
        Ok(files) => files,
        Err(e) => {
            warn!("Failed to query needs_reconcile files: {}", e);
            stats.reconcile_errors += 1;
            return;
        }
    };

    if flagged.is_empty() {
        debug!("No files need reconciliation");
        return;
    }

    info!("Reconciling {} flagged files", flagged.len());

    for file in &flagged {
        reconcile_single_file(pool, queue_manager, stats, file).await;
    }
}

/// Reconcile a single flagged file: look up watch folder, re-queue, clear flag.
async fn reconcile_single_file(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    stats: &mut FullRecoveryStats,
    file: &tracked_files_schema::TrackedFile,
) {
    let wf = sqlx::query_as::<_, (String, String, String)>(
        "SELECT path, collection, tenant_id FROM watch_folders WHERE watch_id = ?1",
    )
    .bind(&file.watch_folder_id)
    .fetch_optional(pool)
    .await;

    let (base_path, collection, tenant_id) = match wf {
        Ok(Some(row)) => row,
        Ok(None) => {
            warn!(
                "Watch folder {} not found for reconcile file_id={}, clearing flag",
                file.watch_folder_id, file.file_id
            );
            let _ = clear_reconcile_flag(pool, file.file_id).await;
            stats.reconcile_errors += 1;
            return;
        }
        Err(e) => {
            warn!(
                "Failed to query watch_folder {}: {}",
                file.watch_folder_id, e
            );
            stats.reconcile_errors += 1;
            return;
        }
    };

    let abs_path = Path::new(&base_path).join(&file.file_path);
    let op = if abs_path.exists() {
        QueueOperation::Update
    } else {
        QueueOperation::Delete
    };

    if let Err(e) = enqueue_file_op(
        queue_manager,
        &tenant_id,
        &collection,
        &abs_path.to_string_lossy(),
        op.clone(),
        None,
    )
    .await
    {
        warn!(
            "Failed to re-queue reconcile file {}: {}",
            file.file_path, e
        );
        stats.reconcile_errors += 1;
        return;
    }

    if let Err(e) = clear_reconcile_flag(pool, file.file_id).await {
        warn!(
            "Failed to clear reconcile flag for file_id={}: {}",
            file.file_id, e
        );
        stats.reconcile_errors += 1;
    } else {
        info!(
            "Reconciled file_id={} ({}): re-queued for {}",
            file.file_id,
            file.file_path,
            op.as_str()
        );
        stats.reconciled += 1;
    }
}

/// Clear the needs_reconcile flag for a tracked file (non-transactional).
async fn clear_reconcile_flag(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    let now = timestamps::now_utc();
    sqlx::query(
        "UPDATE tracked_files SET needs_reconcile = 0, reconcile_reason = NULL, updated_at = ?1
         WHERE file_id = ?2",
    )
    .bind(&now)
    .bind(file_id)
    .execute(pool)
    .await?;
    Ok(())
}
