//! Folder deletion processing: cascade tracked files to individual delete items.

use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::queue_operations::QueueManager;
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    FilePayload, FolderPayload, ItemType, QueueOperation, UnifiedQueueItem,
};

/// Delete all tracked files under a folder path.
///
/// Looks up tracked files whose relative path falls under the folder,
/// then enqueues individual `(File, Delete)` items for each one.
pub(crate) async fn process_folder_delete(
    item: &UnifiedQueueItem,
    payload: &FolderPayload,
    queue_manager: &Arc<QueueManager>,
) -> UnifiedProcessorResult<()> {
    let start = std::time::Instant::now();
    let pool = queue_manager.pool();

    // Look up watch_folder to resolve paths
    let watch_info =
        tracked_files_schema::lookup_watch_folder(pool, &item.tenant_id, &item.collection)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Failed to lookup watch_folder: {}",
                    e
                ))
            })?;

    let (watch_folder_id, base_path) = match watch_info {
        Some((wid, bp)) => (wid, bp),
        None => {
            warn!(
                "No watch_folder for tenant_id={}, collection={} -- nothing to delete",
                item.tenant_id, item.collection
            );
            return Ok(());
        }
    };

    // Compute relative folder path from absolute folder_path and base_path
    let relative_folder =
        tracked_files_schema::compute_relative_path(&payload.folder_path, &base_path)
            .unwrap_or_else(|| payload.folder_path.clone());

    // Get all tracked files under this folder prefix
    let tracked_files =
        tracked_files_schema::get_tracked_files_by_prefix(pool, &watch_folder_id, &relative_folder)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Failed to query tracked files: {}",
                    e
                ))
            })?;

    if tracked_files.is_empty() {
        info!(
            "No tracked files found under folder '{}' (relative='{}') -- nothing to delete",
            payload.folder_path, relative_folder
        );
        return Ok(());
    }

    info!(
        "Folder delete: found {} tracked files under '{}', enqueueing file deletions",
        tracked_files.len(),
        relative_folder
    );

    let (files_queued, errors) =
        enqueue_file_deletions(item, &base_path, &tracked_files, queue_manager).await?;

    let elapsed = start.elapsed();
    info!(
        "Folder delete complete: {} files queued for deletion, {} errors in {:?} (folder={})",
        files_queued, errors, elapsed, payload.folder_path
    );

    if errors > 0 {
        warn!(
            "Folder delete had {} errors out of {} files for folder: {}",
            errors,
            tracked_files.len(),
            payload.folder_path
        );
    }

    Ok(())
}

/// Enqueue individual file deletion items for each tracked file.
///
/// Returns `(files_queued, errors)`.
async fn enqueue_file_deletions(
    item: &UnifiedQueueItem,
    base_path: &str,
    tracked_files: &[(i64, String, Option<String>)],
    queue_manager: &Arc<QueueManager>,
) -> UnifiedProcessorResult<(u64, u64)> {
    let mut files_queued = 0u64;
    let mut errors = 0u64;

    for (file_id, rel_path, _branch) in tracked_files {
        // Reconstruct absolute path for the file payload
        let abs_path = std::path::Path::new(base_path).join(rel_path);
        let abs_path_str = abs_path.to_string_lossy().to_string();

        let file_payload = FilePayload {
            file_path: abs_path_str.clone(),
            file_type: None,
            file_hash: None,
            size_bytes: None,
            old_path: None,
        };

        let payload_json = serde_json::to_string(&file_payload).map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to serialize FilePayload: {}",
                e
            ))
        })?;

        match queue_manager
            .enqueue_unified(
                ItemType::File,
                QueueOperation::Delete,
                &item.tenant_id,
                &item.collection,
                &payload_json,
                Some(&item.branch),
                None,
            )
            .await
        {
            Ok((_queue_id, true)) => {
                files_queued += 1;
                debug!(
                    "Queued file for deletion: {} (file_id={})",
                    abs_path_str, file_id
                );
            }
            Ok((_queue_id, false)) => {
                debug!(
                    "File deletion already in queue (deduplicated): {}",
                    abs_path_str
                );
            }
            Err(e) => {
                warn!("Failed to queue file deletion for {}: {}", abs_path_str, e);
                errors += 1;
            }
        }
    }

    Ok((files_queued, errors))
}
