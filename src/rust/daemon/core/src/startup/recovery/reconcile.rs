//! Reconciliation of tracked_files flagged with needs_reconcile=1.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::queue_operations::QueueManager;
use crate::tracked_files_schema;
use crate::unified_queue_schema::QueueOperation;

use super::queue::enqueue_file_op;
use super::types::FullRecoveryStats;

/// Metadata attached to reconcile-driven re-ingests.
///
/// `force_reingest` makes `prepare_update` bypass its unchanged-hash
/// short-circuit: `needs_reconcile` exists precisely because stored state for
/// the file is broken while its content (and therefore hash) may be
/// unchanged (#110).
const RECONCILE_METADATA: &str = r#"{"source":"needs_reconcile","force_reingest":true}"#;

/// Process tracked_files flagged with needs_reconcile=1.
///
/// For each flagged file, look up its watch_folder to get routing info,
/// then re-queue it for ingestion.
///
/// The `needs_reconcile` flag is NOT cleared here (F-020). It is deferred
/// until the enqueued queue item completes successfully (i.e. is deleted from
/// the unified_queue by `delete_unified_item`). This prevents silent loss of
/// repair intent if the daemon crashes between enqueue and processing.
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

/// Reconcile a single flagged file: look up watch folder and re-queue.
///
/// The `needs_reconcile` flag is left set regardless of whether the enqueue
/// succeeded or was deduplicated (F-020):
/// - If `is_new=true`: item enqueued; flag cleared when the item completes.
/// - If `is_new=false`: an identical item is already in flight; flag stays set
///   until that prior item completes.
/// - If watch folder is missing: flag cleared immediately (file is orphaned;
///   no repair is possible).
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
            // Watch folder gone: no repair possible, clear the flag so this
            // file does not block future reconciliation passes.
            let _ = clear_reconcile_flag_direct(pool, file.file_id).await;
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

    let abs_path = Path::new(&base_path).join(&file.relative_path.as_str());
    let op = if abs_path.exists() {
        QueueOperation::Update
    } else {
        QueueOperation::Delete
    };

    match enqueue_file_op(
        queue_manager,
        &tenant_id,
        &collection,
        &file.relative_path,
        Path::new(&base_path),
        op.clone(),
        Some(RECONCILE_METADATA),
    )
    .await
    {
        Ok(is_new) => {
            // F-020: do NOT clear needs_reconcile here. The flag is cleared by
            // `QueueManager::delete_unified_item` when the queue item completes.
            if is_new {
                info!(
                    "Reconciled file_id={} ({}): enqueued for {}",
                    file.file_id,
                    file.relative_path.as_str(),
                    op.as_str()
                );
            } else {
                debug!(
                    "Reconcile file_id={} ({}): op={} already in queue, flag kept",
                    file.file_id,
                    file.relative_path.as_str(),
                    op.as_str()
                );
            }
            stats.reconciled += 1;
        }
        Err(e) => {
            warn!(
                "Failed to re-queue reconcile file {}: {}",
                file.relative_path.as_str(),
                e
            );
            stats.reconcile_errors += 1;
        }
    }
}

/// Clear the needs_reconcile flag for a tracked file directly.
///
/// Only called when reconciliation is impossible (e.g. watch folder missing).
/// Normal completions go through `QueueManager::delete_unified_item` instead.
async fn clear_reconcile_flag_direct(pool: &SqlitePool, file_id: i64) -> Result<(), sqlx::Error> {
    let now = wqm_common::timestamps::now_utc();
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
