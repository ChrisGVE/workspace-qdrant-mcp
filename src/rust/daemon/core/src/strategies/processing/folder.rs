//! Folder processing strategy.
//!
//! Handles `ItemType::Folder` queue items: directory scanning (progressive
//! single-level enumeration), folder deletion (cascading to tracked files),
//! and folder update/add (treated as rescan).

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::allowed_extensions::AllowedExtensions;
use crate::context::ProcessingContext;
use crate::file_classification::classify_file_type;
use crate::patterns::exclusion::{should_exclude_directory, should_exclude_file};
use crate::queue_operations::QueueManager;
use crate::specs::parse_payload;
use crate::strategies::ProcessingStrategy;
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    FilePayload, FolderPayload, ItemType, ProjectPayload, QueueOperation, UnifiedQueueItem,
};
use wqm_common::constants::COLLECTION_PROJECTS;

/// Strategy for processing folder queue items.
///
/// Routes to folder scan, delete, or rescan based on the queue item operation.
pub struct FolderStrategy;

#[async_trait]
impl ProcessingStrategy for FolderStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::Folder
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        Self::process_folder_item(ctx, item).await
    }

    fn name(&self) -> &'static str {
        "folder"
    }
}

impl FolderStrategy {
    /// Main folder processing entry point.
    ///
    /// Parses the folder payload and dispatches to scan, delete, or rescan.
    pub(crate) async fn process_folder_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing folder item: {} (op={:?}, collection={})",
            item.queue_id, item.op, item.collection
        );

        let payload: FolderPayload = parse_payload(item)?;

        match item.op {
            QueueOperation::Scan => {
                // For project collection, use progressive single-level scan
                if item.collection == COLLECTION_PROJECTS {
                    let dir_path = Path::new(&payload.folder_path);
                    if !dir_path.is_dir() {
                        warn!(
                            "Folder scan target is not a directory: {}",
                            payload.folder_path
                        );
                        return Ok(());
                    }
                    let (files, dirs, excluded, errs) = Self::scan_directory_single_level(
                        dir_path,
                        item,
                        &ctx.queue_manager,
                        &ctx.allowed_extensions,
                    )
                    .await?;
                    info!(
                        "Folder scan: {} files, {} subdirs enqueued, {} excluded, {} errors ({})",
                        files, dirs, excluded, errs, payload.folder_path
                    );
                    Ok(())
                } else {
                    crate::strategies::processing::tenant::TenantStrategy::scan_library_directory(
                        item,
                        &payload.folder_path,
                        &ctx.queue_manager,
                        &ctx.storage_client,
                        &ctx.allowed_extensions,
                    )
                    .await
                }
            }
            QueueOperation::Delete => {
                Self::process_folder_delete(item, &payload, &ctx.queue_manager).await
            }
            QueueOperation::Update | QueueOperation::Add => {
                // Folder update/add is equivalent to a rescan
                info!(
                    "Folder {:?} operation treated as rescan for: {}",
                    item.op, payload.folder_path
                );
                if item.collection == COLLECTION_PROJECTS {
                    let dir_path = Path::new(&payload.folder_path);
                    if !dir_path.is_dir() {
                        warn!(
                            "Folder scan target is not a directory: {}",
                            payload.folder_path
                        );
                        return Ok(());
                    }
                    let (files, dirs, excluded, errs) = Self::scan_directory_single_level(
                        dir_path,
                        item,
                        &ctx.queue_manager,
                        &ctx.allowed_extensions,
                    )
                    .await?;
                    info!(
                        "Folder rescan: {} files, {} subdirs, {} excluded, {} errors ({})",
                        files, dirs, excluded, errs, payload.folder_path
                    );
                    Ok(())
                } else {
                    crate::strategies::processing::tenant::TenantStrategy::scan_library_directory(
                        item,
                        &payload.folder_path,
                        &ctx.queue_manager,
                        &ctx.storage_client,
                        &ctx.allowed_extensions,
                    )
                    .await
                }
            }
            QueueOperation::Rename => {
                // Folder rename: not yet implemented
                info!(
                    "Folder rename not yet implemented for queue_id={}",
                    item.queue_id
                );
                Ok(())
            }
            _ => {
                // Uplift, Reset not valid for folders
                warn!(
                    "Unsupported operation {:?} for folder item {}",
                    item.op, item.queue_id
                );
                Ok(())
            }
        }
    }

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
                let dir_name = entry.file_name().to_string_lossy().to_string();

                // Check directory exclusion
                if should_exclude_directory(&dir_name) {
                    continue;
                }

                // Submodule detection: directory with .git -> (Tenant, Add)
                if path.join(".git").exists() {
                    let submodule_payload = ProjectPayload {
                        project_root: path.to_string_lossy().to_string(),
                        git_remote: None,
                        project_type: None,
                        old_tenant_id: None,
                        is_active: None,
                    };
                    let payload_json = serde_json::to_string(&submodule_payload).unwrap_or_else(
                        |_| format!(r#"{{"project_root":"{}"}}"#, path.display()),
                    );

                    let submodule_tenant =
                        wqm_common::project_id::calculate_tenant_id(&path);

                    if let Ok((_, true)) = queue_manager
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
                        dirs_queued += 1;
                        debug!("Enqueued submodule as Tenant/Add: {}", path.display());
                    }
                    continue;
                }

                // Regular subdirectory -> (Folder, Scan)
                let folder_payload = FolderPayload {
                    folder_path: path.to_string_lossy().to_string(),
                    recursive: false,
                    recursive_depth: 0,
                    patterns: vec![],
                    ignore_patterns: vec![],
                    old_path: None,
                };
                let payload_json = serde_json::to_string(&folder_payload).unwrap_or_else(|_| {
                    format!(r#"{{"folder_path":"{}"}}"#, path.display())
                });

                if let Ok((_, true)) = queue_manager
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
                    dirs_queued += 1;
                }
            } else if file_type.is_file() {
                let abs_path = path.to_string_lossy();

                // Check exclusion rules
                if should_exclude_file(&abs_path) {
                    files_excluded += 1;
                    continue;
                }

                // Check file type allowlist
                if !allowed_extensions.is_allowed(&abs_path, &item.collection) {
                    files_excluded += 1;
                    continue;
                }

                // Get file metadata
                let metadata = match path.metadata() {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to get metadata for {}: {}", abs_path, e);
                        errors += 1;
                        continue;
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
                    files_excluded += 1;
                    continue;
                }

                let file_type_class = classify_file_type(&path);
                let file_payload = FilePayload {
                    file_path: abs_path.to_string(),
                    file_type: Some(file_type_class.as_str().to_string()),
                    file_hash: None,
                    size_bytes: Some(metadata.len()),
                    old_path: None,
                };

                let payload_json =
                    serde_json::to_string(&file_payload).map_err(|e| {
                        UnifiedProcessorError::ProcessingFailed(format!(
                            "Failed to serialize FilePayload: {}",
                            e
                        ))
                    })?;

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
                    Ok((_, is_new)) => {
                        if is_new {
                            files_queued += 1;
                        }
                    }
                    Err(e) => {
                        warn!("Failed to queue file {}: {}", abs_path, e);
                        errors += 1;
                    }
                }
            }
            // Symlinks are skipped (no follow)
        }

        Ok((files_queued, dirs_queued, files_excluded, errors))
    }

    /// Delete all tracked files under a folder path.
    ///
    /// Looks up tracked files whose relative path falls under the folder,
    /// then enqueues individual `(File, Delete)` items for each one.
    async fn process_folder_delete(
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
        let tracked_files = tracked_files_schema::get_tracked_files_by_prefix(
            pool,
            &watch_folder_id,
            &relative_folder,
        )
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

        let mut files_queued = 0u64;
        let mut errors = 0u64;

        for (file_id, rel_path, _branch) in &tracked_files {
            // Reconstruct absolute path for the file payload
            let abs_path = std::path::Path::new(&base_path).join(rel_path);
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
                Ok((_queue_id, is_new)) => {
                    if is_new {
                        files_queued += 1;
                        debug!(
                            "Queued file for deletion: {} (file_id={})",
                            abs_path_str, file_id
                        );
                    } else {
                        debug!(
                            "File deletion already in queue (deduplicated): {}",
                            abs_path_str
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to queue file deletion for {}: {}",
                        abs_path_str, e
                    );
                    errors += 1;
                }
            }
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_folder_strategy_handles_folder_items() {
        let strategy = FolderStrategy;
        assert!(strategy.handles(&ItemType::Folder, &QueueOperation::Scan));
        assert!(strategy.handles(&ItemType::Folder, &QueueOperation::Add));
        assert!(strategy.handles(&ItemType::Folder, &QueueOperation::Delete));
    }

    #[test]
    fn test_folder_strategy_rejects_non_folder_items() {
        let strategy = FolderStrategy;
        assert!(!strategy.handles(&ItemType::File, &QueueOperation::Scan));
        assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Tenant, &QueueOperation::Delete));
    }

    #[test]
    fn test_folder_strategy_name() {
        let strategy = FolderStrategy;
        assert_eq!(strategy.name(), "folder");
    }
}
