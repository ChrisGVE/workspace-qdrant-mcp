//! FolderStrategy trait implementation and main dispatch.

use async_trait::async_trait;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::specs::parse_payload;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FolderPayload, ItemType, QueueOperation, UnifiedQueueItem};
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::paths::CanonicalPath;

use super::delete::process_folder_delete;
use super::scan::scan_directory_single_level;
use crate::allowed_extensions::AllowedExtensions;
use crate::queue_operations::QueueManager;
use crate::watching_queue::WatchManager;
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
        watch_folder_root: &CanonicalPath,
        item: &UnifiedQueueItem,
        queue_manager: &Arc<QueueManager>,
        allowed_extensions: &Arc<AllowedExtensions>,
        last_scan: Option<&str>,
    ) -> UnifiedProcessorResult<(u64, u64, u64, u64)> {
        scan_directory_single_level(
            dir_path,
            watch_folder_root,
            item,
            queue_manager,
            allowed_extensions,
            last_scan,
        )
        .await
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
    let last_scan_for_scan = payload.last_scan.as_deref();

    match item.op {
        QueueOperation::Scan => dispatch_scan(ctx, item, &payload, last_scan_for_scan).await,
        QueueOperation::Delete => process_folder_delete(item, &payload, &ctx.queue_manager).await,
        QueueOperation::Update | QueueOperation::Add => {
            info!(
                "Folder {:?} operation treated as rescan for: {}",
                item.op,
                folder_path_display(&payload),
            );
            dispatch_scan(ctx, item, &payload, None).await
        }
        QueueOperation::Rename => {
            info!(
                "Folder rename not yet implemented for queue_id={}",
                item.queue_id
            );
            Ok(())
        }
        _ => {
            warn!(
                "Unsupported operation {:?} for folder item {}",
                item.op, item.queue_id
            );
            Ok(())
        }
    }
}

/// Dispatch a scan operation to either the project or library handler.
///
/// `last_scan` controls mtime pruning: `None` forces a full rescan.
async fn dispatch_scan(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &FolderPayload,
    last_scan: Option<&str>,
) -> UnifiedProcessorResult<()> {
    // Resolve the watch_folder root that anchors this scan. Required for
    // both branches: project scans need it to compute relative paths of
    // subdir enqueues, library scans need it to anchor the walk.
    let root = match lookup_watch_folder_root(ctx, &item.tenant_id, &item.collection).await? {
        Some(r) => r,
        None => {
            warn!(
                "No watch_folder for tenant_id={}, collection={} -- cannot scan",
                item.tenant_id, item.collection
            );
            return Ok(());
        }
    };

    // Reconstruct the absolute scan target: None payload.folder_path means
    // scan the watch_folder root; Some(rel) joins onto the root.
    let abs_scan_target = match &payload.folder_path {
        None => root.clone(),
        Some(rel) => rel.to_absolute(&root),
    };

    if item.collection == COLLECTION_PROJECTS {
        let dir_path = std::path::Path::new(abs_scan_target.as_str());
        if !dir_path.is_dir() {
            warn!(
                "Folder scan target is not a directory: {}",
                abs_scan_target.as_str()
            );
            return Ok(());
        }

        // Git fast-path: on the ROOT scan of a git project (folder_path None),
        // enumerate the whole git index in one pass so the queue `total`
        // reflects the real tracked-file count immediately instead of growing
        // organically through the progressive single-level FS walk. Falls back
        // to the FS walk when the target is not a git repo / the index is
        // unreadable, or when disabled via WQM_GIT_FASTPATH_DISCOVERY=0.
        if payload.folder_path.is_none() && super::git_scan::git_fastpath_enabled() {
            if let Some(result) = super::git_scan::enumerate_git_index(
                dir_path,
                &root,
                item,
                &ctx.queue_manager,
                &ctx.allowed_extensions,
                last_scan,
            )
            .await
            {
                let (files, submodules, excluded, errs) = result?;
                info!(
                    "Git fast-path scan: {} files, {} submodules enqueued, {} excluded, {} errors ({})",
                    files,
                    submodules,
                    excluded,
                    errs,
                    abs_scan_target.as_str()
                );
                return Ok(());
            }
            // Not a git repo / index unreadable -> fall through to FS scan.
        }

        let (files, dirs, excluded, errs) = scan_directory_single_level(
            dir_path,
            &root,
            item,
            &ctx.queue_manager,
            &ctx.allowed_extensions,
            last_scan,
        )
        .await?;
        let label = if last_scan.is_some() {
            "scan"
        } else {
            "rescan"
        };
        info!(
            "Folder {}: {} files, {} subdirs enqueued, {} excluded, {} errors ({})",
            label,
            files,
            dirs,
            excluded,
            errs,
            abs_scan_target.as_str()
        );
        Ok(())
    } else {
        crate::strategies::processing::tenant::scan_library_directory(
            item,
            abs_scan_target.as_str(),
            &ctx.queue_manager,
            &ctx.storage_client,
            &ctx.allowed_extensions,
        )
        .await
    }
}

/// Lookup helper: resolve the `watch_folders.path` row for a tenant/collection
/// and parse it as a [`CanonicalPath`] for downstream join/strip operations.
async fn lookup_watch_folder_root(
    ctx: &ProcessingContext,
    tenant_id: &str,
    collection: &str,
) -> UnifiedProcessorResult<Option<CanonicalPath>> {
    let row: Option<String> = sqlx::query_scalar(
        "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2",
    )
    .bind(tenant_id)
    .bind(collection)
    .fetch_optional(ctx.queue_manager.pool())
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("Failed to lookup watch_folder: {}", e))
    })?;

    match row {
        None => Ok(None),
        Some(path) => {
            let resolved_path = WatchManager::resolve_local_watch_path(&path);
            let resolved_path = resolved_path.to_string_lossy().to_string();
            CanonicalPath::from_user_input(&resolved_path)
                .map(Some)
                .map_err(|e| {
                    UnifiedProcessorError::InvalidPayload(format!(
                        "watch_folder.path stored for tenant_id={} is not canonical: {}",
                        tenant_id, e
                    ))
                })
        }
    }
}

/// Helper used by log macros that previously dereferenced
/// `payload.folder_path` as a string. Returns either the relative path or
/// the placeholder `"<root>"` for the None case.
fn folder_path_display(payload: &FolderPayload) -> &str {
    payload
        .folder_path
        .as_ref()
        .map(|r| r.as_str())
        .unwrap_or("<root>")
}
