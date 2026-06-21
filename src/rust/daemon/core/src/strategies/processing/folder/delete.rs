//! Folder deletion processing: cascade tracked files to individual delete items.

use std::sync::Arc;

use tracing::{debug, info, warn};
use wqm_common::paths::RelativePath;

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

    let (watch_folder_id, _base_path) = match watch_info {
        Some((wid, bp)) => (wid, bp),
        None => {
            warn!(
                "No watch_folder for tenant_id={}, collection={} -- nothing to delete",
                item.tenant_id, item.collection
            );
            return Ok(());
        }
    };

    // `payload.folder_path` is already relative to the watch_folder root.
    // `None` means delete every tracked file under the watch_folder root.
    let relative_folder = payload
        .folder_path
        .as_ref()
        .map(|r| r.as_str().to_string())
        .unwrap_or_default();

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

    let folder_display: &str = if relative_folder.is_empty() {
        "<root>"
    } else {
        relative_folder.as_str()
    };

    if tracked_files.is_empty() {
        info!(
            "No tracked files found under folder '{}' -- nothing to delete",
            folder_display
        );
        return Ok(());
    }

    info!(
        "Folder delete: found {} tracked files under '{}', enqueueing file deletions",
        tracked_files.len(),
        folder_display
    );

    let (files_queued, errors) =
        enqueue_file_deletions(item, &tracked_files, queue_manager).await?;

    let elapsed = start.elapsed();
    info!(
        "Folder delete complete: {} files queued for deletion, {} errors in {:?} (folder={})",
        files_queued, errors, elapsed, folder_display
    );

    if errors > 0 {
        warn!(
            "Folder delete had {} errors out of {} files for folder: {}",
            errors,
            tracked_files.len(),
            folder_display
        );
    }

    Ok(())
}

/// Enqueue individual file deletion items for each tracked file.
///
/// `tracked_files` rows already carry the validated relative path from
/// `tracked_files.relative_path` — re-build a [`RelativePath`] for each
/// directly without round-tripping through the watch_folder root.
///
/// Returns `(files_queued, errors)`.
async fn enqueue_file_deletions(
    item: &UnifiedQueueItem,
    tracked_files: &[(i64, String, String)],
    queue_manager: &Arc<QueueManager>,
) -> UnifiedProcessorResult<(u64, u64)> {
    let mut files_queued = 0u64;
    let mut errors = 0u64;

    for (file_id, rel_path, _branch) in tracked_files {
        // `rel_path` was validated on insert into `tracked_files`; use
        // `from_user_input` defensively in case persisted data drifts.
        let relative = match RelativePath::from_user_input(rel_path) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Tracked file relative_path failed validation ({}): {}",
                    rel_path, e
                );
                errors += 1;
                continue;
            }
        };

        let file_payload = FilePayload {
            file_path: relative,
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
                    rel_path, file_id
                );
            }
            Ok((_queue_id, false)) => {
                debug!(
                    "File deletion already in queue (deduplicated): {}",
                    rel_path
                );
            }
            Err(e) => {
                warn!("Failed to queue file deletion for {}: {}", rel_path, e);
                errors += 1;
            }
        }
    }

    Ok((files_queued, errors))
}
