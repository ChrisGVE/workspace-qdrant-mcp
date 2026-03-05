//! FolderStrategy trait implementation and main dispatch.

use async_trait::async_trait;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::specs::parse_payload;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FolderPayload, ItemType, QueueOperation, UnifiedQueueItem};
use wqm_common::constants::COLLECTION_PROJECTS;

use super::delete::process_folder_delete;
use super::scan::scan_directory_single_level;
use crate::allowed_extensions::AllowedExtensions;
use crate::queue_operations::QueueManager;
use std::path::Path;
use std::sync::Arc;

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
        process_folder_item(ctx, item).await
    }

    fn name(&self) -> &'static str {
        "folder"
    }
}

impl FolderStrategy {
    /// Delegation shim: callers that reference `FolderStrategy::process_folder_item`.
    pub(crate) async fn process_folder_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        process_folder_item(ctx, item).await
    }

    /// Delegation shim: callers that reference `FolderStrategy::scan_directory_single_level`.
    pub(crate) async fn scan_directory_single_level(
        dir_path: &Path,
        item: &UnifiedQueueItem,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<(u64, u64, u64, u64)> {
        scan_directory_single_level(dir_path, item, queue_manager, allowed_extensions).await
    }
}

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
                let dir_path = std::path::Path::new(&payload.folder_path);
                if !dir_path.is_dir() {
                    warn!(
                        "Folder scan target is not a directory: {}",
                        payload.folder_path
                    );
                    return Ok(());
                }
                let (files, dirs, excluded, errs) = scan_directory_single_level(
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
                crate::strategies::processing::tenant::scan_library_directory(
                    item,
                    &payload.folder_path,
                    &ctx.queue_manager,
                    &ctx.storage_client,
                    &ctx.allowed_extensions,
                )
                .await
            }
        }
        QueueOperation::Delete => process_folder_delete(item, &payload, &ctx.queue_manager).await,
        QueueOperation::Update | QueueOperation::Add => {
            // Folder update/add is equivalent to a rescan
            info!(
                "Folder {:?} operation treated as rescan for: {}",
                item.op, payload.folder_path
            );
            if item.collection == COLLECTION_PROJECTS {
                let dir_path = std::path::Path::new(&payload.folder_path);
                if !dir_path.is_dir() {
                    warn!(
                        "Folder scan target is not a directory: {}",
                        payload.folder_path
                    );
                    return Ok(());
                }
                let (files, dirs, excluded, errs) = scan_directory_single_level(
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
                crate::strategies::processing::tenant::scan_library_directory(
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
