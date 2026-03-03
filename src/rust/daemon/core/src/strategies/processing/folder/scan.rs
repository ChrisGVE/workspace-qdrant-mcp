//! Progressive single-level directory scan logic.

use std::path::Path;
use std::sync::Arc;

use tracing::{debug, warn};

use crate::allowed_extensions::AllowedExtensions;
use crate::file_classification::classify_file_type;
use crate::patterns::exclusion::{should_exclude_directory, should_exclude_file};
use crate::queue_operations::QueueManager;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    FilePayload, FolderPayload, ItemType, ProjectPayload, QueueOperation, UnifiedQueueItem,
};

/// Progressive single-level directory scan.
///
/// Enumerates only the immediate children of `dir_path`:
/// - Files: check exclusion + allowlist, enqueue `(File, Add)`
/// - Directories: check exclusion, enqueue `(Folder, Scan)`
/// - Directories with `.git`: submodule detection, enqueue `(Tenant, Add)`
///
/// Returns `(files_queued, dirs_queued, files_excluded, errors)`.
pub(crate) async fn scan_directory_single_level(
    dir_path: &Path,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
) -> UnifiedProcessorResult<(u64, u64, u64, u64)> {
    let mut files_queued = 0u64;
    let mut dirs_queued = 0u64;
    let mut files_excluded = 0u64;
    let mut errors = 0u64;

    let entries = std::fs::read_dir(dir_path).map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!(
            "Failed to read directory {}: {}",
            dir_path.display(),
            e
        ))
    })?;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!(
                    "Failed to read dir entry in {}: {}",
                    dir_path.display(),
                    e
                );
                errors += 1;
                continue;
            }
        };

        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                warn!("Failed to get file type for {}: {}", path.display(), e);
                errors += 1;
                continue;
            }
        };

        if file_type.is_dir() {
            dirs_queued += process_directory_entry(
                &path,
                &entry.file_name().to_string_lossy().to_string(),
                item,
                queue_manager,
                &mut errors,
            )
            .await;
        } else if file_type.is_file() {
            files_queued += process_file_entry(
                &path,
                item,
                queue_manager,
                allowed_extensions,
                &mut files_excluded,
                &mut errors,
            )
            .await;
        }
        // Symlinks are skipped (no follow)
    }

    Ok((files_queued, dirs_queued, files_excluded, errors))
}

/// Process a single directory entry encountered during scan.
///
/// Returns 1 if an item was enqueued, 0 otherwise.
async fn process_directory_entry(
    path: &Path,
    dir_name: &str,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    errors: &mut u64,
) -> u64 {
    // Check directory exclusion
    if should_exclude_directory(dir_name) {
        return 0;
    }

    // Submodule detection: directory with .git -> (Tenant, Add)
    if path.join(".git").exists() {
        return enqueue_submodule(path, item, queue_manager, errors).await;
    }

    // Regular subdirectory -> (Folder, Scan)
    enqueue_subdirectory(path, item, queue_manager).await
}

/// Enqueue a submodule directory as a Tenant/Add item.
///
/// Returns 1 if enqueued successfully, 0 otherwise.
async fn enqueue_submodule(
    path: &Path,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    errors: &mut u64,
) -> u64 {
    let submodule_payload = ProjectPayload {
        project_root: path.to_string_lossy().to_string(),
        git_remote: None,
        project_type: None,
        old_tenant_id: None,
        is_active: None,
    };
    let payload_json = serde_json::to_string(&submodule_payload)
        .unwrap_or_else(|_| format!(r#"{{"project_root":"{}"}}"#, path.display()));

    let submodule_tenant = wqm_common::project_id::calculate_tenant_id(path);

    match queue_manager
        .enqueue_unified(
            ItemType::Tenant,
            QueueOperation::Add,
            &submodule_tenant,
            &item.collection,
            &payload_json,
            None,
            None,
        )
        .await
    {
        Ok((_, true)) => {
            debug!("Enqueued submodule as Tenant/Add: {}", path.display());
            1
        }
        Ok((_, false)) => 0,
        Err(e) => {
            warn!("Failed to enqueue submodule {}: {}", path.display(), e);
            *errors += 1;
            0
        }
    }
}

/// Enqueue a regular subdirectory as a Folder/Scan item.
///
/// Returns 1 if enqueued successfully, 0 otherwise.
async fn enqueue_subdirectory(
    path: &Path,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
) -> u64 {
    let folder_payload = FolderPayload {
        folder_path: path.to_string_lossy().to_string(),
        recursive: false,
        recursive_depth: 0,
        patterns: vec![],
        ignore_patterns: vec![],
        old_path: None,
    };
    let payload_json = serde_json::to_string(&folder_payload)
        .unwrap_or_else(|_| format!(r#"{{"folder_path":"{}"}}"#, path.display()));

    match queue_manager
        .enqueue_unified(
            ItemType::Folder,
            QueueOperation::Scan,
            &item.tenant_id,
            &item.collection,
            &payload_json,
            None,
            None,
        )
        .await
    {
        Ok((_, true)) => 1,
        _ => 0,
    }
}

/// Process a single file entry encountered during scan.
///
/// Returns 1 if the file was enqueued, 0 otherwise.
async fn process_file_entry(
    path: &Path,
    item: &UnifiedQueueItem,
    queue_manager: &Arc<QueueManager>,
    allowed_extensions: &Arc<AllowedExtensions>,
    files_excluded: &mut u64,
    errors: &mut u64,
) -> u64 {
    let abs_path = path.to_string_lossy();

    // Check exclusion rules
    if should_exclude_file(&abs_path) {
        *files_excluded += 1;
        return 0;
    }

    // Check file type allowlist
    if !allowed_extensions.is_allowed(&abs_path, &item.collection) {
        *files_excluded += 1;
        return 0;
    }

    // Get file metadata
    let metadata = match path.metadata() {
        Ok(m) => m,
        Err(e) => {
            warn!("Failed to get metadata for {}: {}", abs_path, e);
            *errors += 1;
            return 0;
        }
    };

    // Skip files that are too large (100MB limit)
    const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;
    if metadata.len() > MAX_FILE_SIZE {
        debug!(
            "Skipping large file: {} ({} bytes)",
            abs_path,
            metadata.len()
        );
        *files_excluded += 1;
        return 0;
    }

    let file_type_class = classify_file_type(path);
    let file_payload = FilePayload {
        file_path: abs_path.to_string(),
        file_type: Some(file_type_class.as_str().to_string()),
        file_hash: None,
        size_bytes: Some(metadata.len()),
        old_path: None,
    };

    let payload_json = match serde_json::to_string(&file_payload) {
        Ok(j) => j,
        Err(e) => {
            warn!("Failed to serialize FilePayload for {}: {}", abs_path, e);
            *errors += 1;
            return 0;
        }
    };

    match queue_manager
        .enqueue_unified(
            ItemType::File,
            QueueOperation::Add,
            &item.tenant_id,
            &item.collection,
            &payload_json,
            Some(&item.branch),
            None,
        )
        .await
    {
        Ok((_, true)) => 1,
        Ok((_, false)) => 0,
        Err(e) => {
            warn!("Failed to queue file {}: {}", abs_path, e);
            *errors += 1;
            0
        }
    }
}
