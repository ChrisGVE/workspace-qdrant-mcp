//! Post-scan exclusion file cleanup for the tenant strategy.
//!
//! After a project or library scan completes, checks tracked files against
//! current exclusion rules and the file type allowlist, queuing deletion
//! for files that now match exclusion patterns.

use std::path::Path;
use std::sync::Arc;

use tracing::{debug, error, info, warn};

use crate::allowed_extensions::AllowedExtensions;
use crate::patterns::exclusion::should_exclude_file;
use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation, UnifiedQueueItem};

/// Clean up excluded files after a scan completes (Task 506).
///
/// Queries tracked_files (fast SQLite) instead of scrolling Qdrant.
/// Checks each tracked file against current exclusion rules and the
/// file type allowlist (Task 511), queuing deletion for files that
/// now match exclusion patterns or are no longer in the allowlist.
pub(crate) async fn cleanup_excluded_files(
    item: &UnifiedQueueItem,
    project_root: &Path,
    queue_manager: &Arc<QueueManager>,
    storage_client: &Arc<StorageClient>,
    allowed_extensions: &Arc<AllowedExtensions>,
) -> UnifiedProcessorResult<u64> {
    let pool = queue_manager.pool();

    // Look up watch_folder for this project
    let watch_info =
        tracked_files_schema::lookup_watch_folder(pool, &item.tenant_id, &item.collection)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Failed to lookup watch_folder: {}",
                    e
                ))
            })?;

    let (watch_folder_id, base_path) = match &watch_info {
        Some((wid, bp)) => (wid.as_str(), bp.as_str()),
        None => {
            // No watch_folder -- fall back to Qdrant scroll for backward compatibility
            debug!(
                "No watch_folder for tenant_id={}, falling back to Qdrant scroll for cleanup",
                item.tenant_id
            );
            return cleanup_excluded_files_qdrant_fallback(
                item,
                project_root,
                queue_manager,
                storage_client,
                allowed_extensions,
            )
            .await;
        }
    };

    // Query tracked_files for all files in this project (fast SQLite query)
    let tracked_files = tracked_files_schema::get_tracked_file_paths(pool, watch_folder_id)
        .await
        .map_err(|e| {
            error!("Failed to query tracked_files for exclusion cleanup: {}", e);
            UnifiedProcessorError::QueueOperation(e.to_string())
        })?;

    if tracked_files.is_empty() {
        debug!(
            "No tracked files for watch_folder_id='{}', skipping exclusion cleanup",
            watch_folder_id
        );
        return Ok(0);
    }

    info!(
        "Checking {} tracked files against exclusion rules (watch_folder_id={})",
        tracked_files.len(),
        watch_folder_id
    );

    let mut files_cleaned = 0u64;

    for (_file_id, rel_path, _branch) in &tracked_files {
        // Check if this file should now be excluded (pattern or allowlist)
        let should_clean = should_exclude_file(rel_path) || {
            let abs_path = Path::new(base_path).join(rel_path);
            !allowed_extensions.is_allowed(&abs_path.to_string_lossy(), &item.collection)
        };

        if !should_clean {
            continue;
        }

        // Reconstruct absolute path for the queue payload
        let abs_path = Path::new(base_path).join(rel_path);
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
                "Failed to serialize FilePayload for deletion: {}",
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
            Ok((_queue_id, is_new)) => {
                if is_new {
                    files_cleaned += 1;
                    debug!(
                        "Queued excluded file for deletion: {} (rel={})",
                        abs_path_str, rel_path
                    );
                } else {
                    debug!(
                        "Excluded file deletion already in queue (deduplicated): {}",
                        abs_path_str
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Failed to queue excluded file for deletion {}: {}",
                    abs_path_str, e
                );
            }
        }
    }

    if files_cleaned > 0 {
        info!(
            "Queued {} excluded files for deletion (watch_folder_id={})",
            files_cleaned, watch_folder_id
        );
    }

    Ok(files_cleaned)
}

/// Fallback cleanup using Qdrant scroll (for when tracked_files is not available).
async fn cleanup_excluded_files_qdrant_fallback(
    item: &UnifiedQueueItem,
    project_root: &Path,
    queue_manager: &Arc<QueueManager>,
    storage_client: &Arc<StorageClient>,
    allowed_extensions: &Arc<AllowedExtensions>,
) -> UnifiedProcessorResult<u64> {
    if !storage_client
        .collection_exists(&item.collection)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
    {
        return Ok(0);
    }

    let qdrant_file_paths = match storage_client
        .scroll_file_paths_by_tenant(&item.collection, &item.tenant_id)
        .await
    {
        Ok(paths) => paths,
        Err(e) => {
            error!("Failed to scroll Qdrant for exclusion cleanup: {}", e);
            return Ok(0);
        }
    };

    if qdrant_file_paths.is_empty() {
        return Ok(0);
    }

    let mut files_cleaned = 0u64;
    for qdrant_file in &qdrant_file_paths {
        let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
            Ok(stripped) => stripped.to_string_lossy().to_string(),
            Err(_) => qdrant_file.clone(),
        };

        let should_clean = should_exclude_file(&rel_path)
            || !allowed_extensions.is_allowed(qdrant_file, &item.collection);
        if !should_clean {
            continue;
        }

        if enqueue_file_for_deletion(item, qdrant_file, queue_manager).await? {
            files_cleaned += 1;
        }
    }

    if files_cleaned > 0 {
        info!(
            "Queued {} excluded files for deletion via Qdrant fallback (tenant_id={})",
            files_cleaned, item.tenant_id
        );
    }

    Ok(files_cleaned)
}

/// Build a delete payload and enqueue it; returns true if a new item was created.
async fn enqueue_file_for_deletion(
    item: &UnifiedQueueItem,
    file_path: &str,
    queue_manager: &Arc<QueueManager>,
) -> UnifiedProcessorResult<bool> {
    let file_payload = FilePayload {
        file_path: file_path.to_string(),
        file_type: None,
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };

    let payload_json = serde_json::to_string(&file_payload).map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!(
            "Failed to serialize FilePayload for deletion: {}",
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
        Ok((_queue_id, is_new)) => Ok(is_new),
        Err(e) => {
            warn!(
                "Failed to queue excluded file for deletion {}: {}",
                file_path, e
            );
            Ok(false)
        }
    }
}
