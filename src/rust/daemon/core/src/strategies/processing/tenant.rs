//! Tenant processing strategy.
//!
//! Handles `ItemType::Tenant` and `ItemType::Doc` queue items:
//! project registration, scanning, library management, tenant deletion,
//! document deletion, and tenant renaming.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

use crate::allowed_extensions::AllowedExtensions;
use crate::context::ProcessingContext;
use crate::file_classification::classify_file_type;
use crate::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};
use crate::patterns::exclusion::{should_exclude_directory, should_exclude_file};
use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;
use crate::strategies::ProcessingStrategy;
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    FilePayload, ItemType, LibraryPayload, ProjectPayload, QueueOperation, UnifiedQueueItem,
};
use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_MEMORY, COLLECTION_PROJECTS, COLLECTION_SCRATCHPAD,
};

/// Strategy for processing tenant and document queue items.
///
/// Routes `ItemType::Tenant` items by operation and collection, and handles
/// `ItemType::Doc` delete operations.
pub struct TenantStrategy;

#[async_trait]
impl ProcessingStrategy for TenantStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::Tenant || *item_type == ItemType::Doc
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        match item.item_type {
            ItemType::Doc => Self::process_doc_item(ctx, item).await,
            _ => Self::process_tenant_item(ctx, item).await,
        }
    }

    fn name(&self) -> &'static str {
        "tenant"
    }
}

impl TenantStrategy {
    // =========================================================================
    // Tenant dispatch
    // =========================================================================

    /// Main tenant processing entry point.
    ///
    /// Routes by operation: Delete, Rename, or Add/Scan/Update (further
    /// sub-routed by collection).
    pub(crate) async fn process_tenant_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        match item.op {
            QueueOperation::Delete => {
                Self::process_delete_tenant_item(ctx, item).await
            }
            QueueOperation::Rename => {
                Self::process_tenant_rename_item(item, &ctx.storage_client).await
            }
            _ => {
                // Add, Scan, Update -- route by collection
                match item.collection.as_str() {
                    "libraries" => {
                        Self::process_library_item(ctx, item).await
                    }
                    _ => {
                        Self::process_project_item(ctx, item).await
                    }
                }
            }
        }
    }

    /// Main Doc dispatch -- currently only Delete and Uplift (placeholder).
    pub(crate) async fn process_doc_item(
        _ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        match item.op {
            QueueOperation::Delete => {
                Self::process_delete_document_item(item, &_ctx.storage_client).await
            }
            QueueOperation::Uplift => {
                // Placeholder: no enrichment logic yet
                info!(
                    "Doc uplift placeholder for queue_id={} tenant={}",
                    item.queue_id, item.tenant_id
                );
                Ok(())
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for Doc item {}",
                    item.op, item.queue_id
                );
                Ok(())
            }
        }
    }

    // =========================================================================
    // Project operations
    // =========================================================================

    /// Process project item -- create/manage project collections.
    async fn process_project_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing project item: {} (op={:?})",
            item.queue_id, item.op
        );

        let payload: ProjectPayload =
            serde_json::from_str(&item.payload_json).map_err(|e| {
                UnifiedProcessorError::InvalidPayload(format!(
                    "Failed to parse ProjectPayload: {}",
                    e
                ))
            })?;

        match item.op {
            QueueOperation::Add => {
                crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

                // 2. Detect git status (Task 11)
                let git_status = crate::git_integration::detect_git_status(
                    std::path::Path::new(&payload.project_root),
                );

                // 3. Create watch_folder entry (idempotent -- skip if already exists)
                let now = wqm_common::timestamps::now_utc();
                let watch_id = uuid::Uuid::new_v4().to_string();
                let is_active: i32 =
                    if payload.is_active.unwrap_or(false) { 1 } else { 0 };
                let insert_result = sqlx::query(
                    r#"INSERT OR IGNORE INTO watch_folders (
                        watch_id, path, collection, tenant_id, is_active,
                        git_remote_url, last_activity_at, follow_symlinks, enabled,
                        cleanup_on_disable, is_git_tracked, last_commit_hash, created_at, updated_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 1, 0, ?8, ?9, ?7, ?7)"#,
                )
                .bind(&watch_id)
                .bind(&payload.project_root)
                .bind(&item.collection)
                .bind(&item.tenant_id)
                .bind(is_active)
                .bind(&payload.git_remote)
                .bind(&now)
                .bind(git_status.is_git as i32)
                .bind(&git_status.commit_hash)
                .execute(ctx.queue_manager.pool())
                .await;

                match insert_result {
                    Ok(result) => {
                        if result.rows_affected() > 0 {
                            info!(
                                "Created watch_folder for tenant={} path={} (active={}, git={}, branch={}, worktree={})",
                                item.tenant_id, payload.project_root, is_active,
                                git_status.is_git, git_status.branch, git_status.is_worktree,
                            );
                        } else {
                            info!(
                                "Watch folder already exists for tenant={} (idempotent)",
                                item.tenant_id
                            );
                        }
                    }
                    Err(e) => {
                        return Err(UnifiedProcessorError::ProcessingFailed(format!(
                            "Failed to create watch_folder: {}",
                            e
                        )));
                    }
                }

                // 3. Enqueue (Tenant, Scan) to trigger directory scanning
                let scan_payload_json = serde_json::to_string(&payload).map_err(|e| {
                    UnifiedProcessorError::InvalidPayload(format!(
                        "Failed to serialize scan payload: {}",
                        e
                    ))
                })?;

                match ctx
                    .queue_manager
                    .enqueue_unified(
                        ItemType::Tenant,
                        QueueOperation::Scan,
                        &item.tenant_id,
                        &item.collection,
                        &scan_payload_json,
                        0,
                        None,
                        None,
                    )
                    .await
                {
                    Ok((queue_id, is_new)) => {
                        if is_new {
                            info!(
                                "Enqueued project scan for tenant={} queue_id={}",
                                item.tenant_id, queue_id
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to enqueue project scan for tenant={}: {} (non-critical)",
                            item.tenant_id, e
                        );
                    }
                }
            }
            QueueOperation::Scan => {
                // Scan project directory and queue file ingestion items
                Self::scan_project_directory(ctx, item, &payload).await?;

                // Update last_scan timestamp for this project's watch_folder
                let update_result = sqlx::query(
                    "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE tenant_id = ?1 AND collection = ?2"
                )
                    .bind(&item.tenant_id)
                    .bind(COLLECTION_PROJECTS)
                    .execute(ctx.queue_manager.pool())
                    .await;

                match update_result {
                    Ok(result) => {
                        if result.rows_affected() > 0 {
                            info!(
                                "Updated last_scan for project tenant_id={}",
                                item.tenant_id
                            );
                        } else {
                            debug!(
                                "No watch_folder found for tenant_id={} (may not be watched)",
                                item.tenant_id
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to update last_scan for tenant_id={}: {} (non-critical)",
                            item.tenant_id, e
                        );
                    }
                }
            }
            QueueOperation::Delete => {
                // Delete project data (tenant-scoped, not the whole collection)
                if ctx
                    .storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!(
                        "Deleting project data for tenant={} from collection={}",
                        item.tenant_id, item.collection
                    );
                    ctx.storage_client
                        .delete_points_by_tenant(&item.collection, &item.tenant_id)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
            }
            QueueOperation::Uplift => {
                // Query tracked_files for all files of this tenant, enqueue (Doc, Uplift) for each
                let files: Vec<(String, String)> = sqlx::query_as(
                    "SELECT file_id, file_path FROM tracked_files WHERE tenant_id = ?1",
                )
                .bind(&item.tenant_id)
                .fetch_all(ctx.queue_manager.pool())
                .await
                .map_err(|e| {
                    UnifiedProcessorError::ProcessingFailed(format!(
                        "Failed to query tracked_files for uplift: {}",
                        e
                    ))
                })?;

                let mut enqueued = 0u32;
                for (file_id, file_path) in &files {
                    let doc_payload = serde_json::json!({
                        "file_id": file_id,
                        "file_path": file_path,
                    })
                    .to_string();

                    if let Ok((_, true)) = ctx
                        .queue_manager
                        .enqueue_unified(
                            ItemType::Doc,
                            QueueOperation::Uplift,
                            &item.tenant_id,
                            &item.collection,
                            &doc_payload,
                            0,
                            None,
                            None,
                        )
                        .await
                    {
                        enqueued += 1;
                    }
                }
                info!(
                    "Tenant uplift: enqueued {}/{} doc uplift items for tenant={}",
                    enqueued,
                    files.len(),
                    item.tenant_id
                );
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for project item {}",
                    item.op, item.queue_id
                );
            }
        }

        info!(
            "Successfully processed project item {} (project_root={})",
            item.queue_id, payload.project_root
        );

        Ok(())
    }

    /// Scan a project directory and queue file ingestion items.
    async fn scan_project_directory(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
        payload: &ProjectPayload,
    ) -> UnifiedProcessorResult<()> {
        let project_root = Path::new(&payload.project_root);

        if !project_root.exists() {
            return Err(UnifiedProcessorError::FileNotFound(format!(
                "Project root does not exist: {}",
                payload.project_root
            )));
        }

        if !project_root.is_dir() {
            return Err(UnifiedProcessorError::InvalidPayload(format!(
                "Project root is not a directory: {}",
                payload.project_root
            )));
        }

        crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Scanning project directory (single level): {} (tenant_id={})",
            payload.project_root, item.tenant_id
        );

        // Progressive scan: enumerate only the immediate children of the directory.
        // Subdirectories are enqueued as (Folder, Scan) for deferred processing.
        let (files_queued, dirs_queued, files_excluded, errors) =
            crate::strategies::processing::folder::FolderStrategy::scan_directory_single_level(
                project_root,
                item,
                &ctx.queue_manager,
                &ctx.allowed_extensions,
            )
            .await?;

        // After scanning, clean up any excluded files that were previously indexed
        let files_cleaned = Self::cleanup_excluded_files(
            item,
            project_root,
            &ctx.queue_manager,
            &ctx.storage_client,
            &ctx.allowed_extensions,
        )
        .await?;

        info!(
            "Project scan complete: {} files, {} subdirs enqueued, {} excluded, {} cleaned, {} errors (project={})",
            files_queued, dirs_queued, files_excluded, files_cleaned, errors, payload.project_root
        );

        Ok(())
    }

    // =========================================================================
    // Library operations
    // =========================================================================

    /// Process library item -- create/manage library collections.
    async fn process_library_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing library item: {} (op={:?})",
            item.queue_id, item.op
        );

        let payload: LibraryPayload =
            serde_json::from_str(&item.payload_json).map_err(|e| {
                UnifiedProcessorError::InvalidPayload(format!(
                    "Failed to parse LibraryPayload: {}",
                    e
                ))
            })?;

        match item.op {
            QueueOperation::Add => {
                crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            }
            QueueOperation::Scan => {
                // Scan library directory - look up path from watch_folders
                let pool = ctx.queue_manager.pool();
                let folder_path: Option<String> = sqlx::query_scalar(
                    "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2",
                )
                .bind(&item.tenant_id)
                .bind(COLLECTION_LIBRARIES)
                .fetch_optional(pool)
                .await
                .map_err(|e| {
                    UnifiedProcessorError::QueueOperation(format!(
                        "Failed to lookup library path: {}",
                        e
                    ))
                })?;

                match folder_path {
                    Some(path) => {
                        Self::scan_library_directory(
                            item,
                            &path,
                            &ctx.queue_manager,
                            &ctx.storage_client,
                            &ctx.allowed_extensions,
                        )
                        .await?;
                    }
                    None => {
                        warn!(
                            "Library '{}' not found in watch_folders",
                            item.tenant_id
                        );
                    }
                }
            }
            QueueOperation::Delete => {
                // Delete library data from collection
                if ctx
                    .storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!(
                        "Deleting library data for tenant={} from collection={}",
                        item.tenant_id, item.collection
                    );
                    ctx.storage_client
                        .delete_points_by_tenant(&item.collection, &item.tenant_id)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for library item {}",
                    item.op, item.queue_id
                );
            }
        }

        info!(
            "Successfully processed library item {} (library={})",
            item.queue_id, payload.library_name
        );

        Ok(())
    }

    /// Scan a library directory and enqueue files for ingestion (Task 523).
    ///
    /// Similar to scan_project_directory but for library folders:
    /// - Uses `tenant_id` as library name
    /// - Targets the `libraries` collection
    /// - No branch tracking (libraries are not Git repos)
    pub(crate) async fn scan_library_directory(
        item: &UnifiedQueueItem,
        folder_path: &str,
        queue_manager: &Arc<QueueManager>,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<()> {
        let library_root = Path::new(folder_path);

        if !library_root.exists() {
            return Err(UnifiedProcessorError::FileNotFound(format!(
                "Library path does not exist: {}",
                folder_path
            )));
        }

        if !library_root.is_dir() {
            return Err(UnifiedProcessorError::InvalidPayload(format!(
                "Library path is not a directory: {}",
                folder_path
            )));
        }

        crate::shared::ensure_collection(storage_client, &item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Scanning library directory: {} (tenant_id={})",
            folder_path, item.tenant_id
        );

        let mut files_queued = 0u64;
        let mut files_excluded = 0u64;
        let mut errors = 0u64;
        let start_time = std::time::Instant::now();

        for entry in WalkDir::new(library_root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                if e.file_type().is_dir() && e.depth() > 0 {
                    let dir_name = e.file_name().to_string_lossy();
                    !should_exclude_directory(&dir_name)
                } else {
                    true
                }
            })
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let rel_path = path
                .strip_prefix(library_root)
                .unwrap_or(path)
                .to_string_lossy();

            if should_exclude_file(&rel_path) {
                files_excluded += 1;
                continue;
            }

            let abs_path = path.to_string_lossy();
            if should_exclude_file(&abs_path) {
                files_excluded += 1;
                continue;
            }

            if !allowed_extensions.is_allowed(&abs_path, &item.collection) {
                files_excluded += 1;
                continue;
            }

            let metadata = match path.metadata() {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to get metadata for {}: {}", abs_path, e);
                    errors += 1;
                    continue;
                }
            };

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

            let file_type = classify_file_type(path);

            let file_payload = FilePayload {
                file_path: abs_path.to_string(),
                file_type: Some(file_type.as_str().to_string()),
                file_hash: None,
                size_bytes: Some(metadata.len()),
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
                    QueueOperation::Add,
                    &item.tenant_id,
                    &item.collection,
                    &payload_json,
                    0,
                    Some(""), // No branch for libraries
                    None,
                )
                .await
            {
                Ok((queue_id, is_new)) => {
                    if is_new {
                        files_queued += 1;
                        debug!(
                            "Queued library file for ingestion: {} (queue_id={})",
                            abs_path, queue_id
                        );
                    } else {
                        debug!(
                            "Library file already in queue (deduplicated): {}",
                            abs_path
                        );
                    }
                }
                Err(e) => {
                    warn!("Failed to queue library file {}: {}", abs_path, e);
                    errors += 1;
                }
            }

            if files_queued % 100 == 0 && files_queued > 0 {
                tokio::task::yield_now().await;
            }
        }

        // Update last_scan timestamp for this library's watch_folder
        let update_result = sqlx::query(
            "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE tenant_id = ?1 AND collection = ?2"
        )
            .bind(&item.tenant_id)
            .bind(COLLECTION_LIBRARIES)
            .execute(queue_manager.pool())
            .await;

        if let Err(e) = update_result {
            warn!(
                "Failed to update last_scan for library {}: {}",
                item.tenant_id, e
            );
        }

        let elapsed = start_time.elapsed();
        info!(
            "Library scan complete: {} files queued, {} excluded, {} errors in {:?} (library={})",
            files_queued, files_excluded, errors, elapsed, folder_path
        );

        Ok(())
    }

    // =========================================================================
    // Exclusion cleanup
    // =========================================================================

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
                return Self::cleanup_excluded_files_qdrant_fallback(
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
                error!(
                    "Failed to query tracked_files for exclusion cleanup: {}",
                    e
                );
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
                    0,
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

            // Check exclusion patterns and allowlist (Task 511)
            let should_clean = should_exclude_file(&rel_path)
                || !allowed_extensions.is_allowed(qdrant_file, &item.collection);

            if !should_clean {
                continue;
            }

            let file_payload = FilePayload {
                file_path: qdrant_file.clone(),
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
                    0,
                    Some(&item.branch),
                    None,
                )
                .await
            {
                Ok((_queue_id, is_new)) => {
                    if is_new {
                        files_cleaned += 1;
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to queue excluded file for deletion {}: {}",
                        qdrant_file, e
                    );
                }
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

    // =========================================================================
    // Tenant delete (surgical cascade across all 4 collections)
    // =========================================================================

    /// Process delete tenant item -- surgically delete all data for a tenant.
    ///
    /// Full deletion cascade across all 4 canonical collections:
    /// 1. Purge pending queue items for this tenant
    /// 2. Collect tracked files + qdrant_chunks point IDs, grouped by collection
    /// 3. Delete from Qdrant in batches of point IDs (projects, libraries)
    /// 4. Scroll + batch-delete from memory collection
    /// 5. Scroll + batch-delete from scratchpad collection
    /// 6. SQLite cleanup: qdrant_chunks, tracked_files, keywords, tags,
    ///    keyword_baskets, tag_hierarchy_edges, canonical_tags, watch_folders
    /// 7. Final verification: count remaining tenant points in all 4 collections
    async fn process_delete_tenant_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        const QDRANT_BATCH_SIZE: usize = 100;

        info!(
            "Processing surgical delete for tenant={} (queue_id={})",
            item.tenant_id, item.queue_id
        );

        let pool = ctx.queue_manager.pool();

        // -- Step 1: Purge queue entries for this tenant --
        match ctx
            .queue_manager
            .purge_pending_for_tenant(&item.tenant_id, &item.queue_id)
            .await
        {
            Ok(purged) => {
                if purged > 0 {
                    info!(
                        "Step 1: purged {} queue items for tenant={}",
                        purged, item.tenant_id
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Step 1: queue purge failed for tenant={}: {} (continuing)",
                    item.tenant_id, e
                );
            }
        }

        // -- Step 2: Collect tracked files with point IDs, grouped by collection --
        let file_data: Vec<(i64, String, String, Option<String>)> = sqlx::query_as(
            r#"SELECT tf.file_id, tf.file_path, tf.collection, qc.point_id
               FROM tracked_files tf
               JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
               LEFT JOIN qdrant_chunks qc ON qc.file_id = tf.file_id
               WHERE wf.tenant_id = ?1"#,
        )
        .bind(&item.tenant_id)
        .fetch_all(pool)
        .await
        .map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to query tracked files for tenant={}: {}",
                item.tenant_id, e
            ))
        })?;

        // Group point_ids by collection, track unique file_ids
        let mut points_by_collection: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let mut file_count: std::collections::HashSet<i64> = std::collections::HashSet::new();

        for (file_id, _file_path, collection, point_id) in &file_data {
            file_count.insert(*file_id);
            if let Some(pid) = point_id {
                points_by_collection
                    .entry(collection.clone())
                    .or_default()
                    .push(pid.clone());
            }
        }

        info!(
            "Step 2: found {} tracked files across {} collections for tenant={}",
            file_count.len(),
            points_by_collection.len(),
            item.tenant_id
        );

        // -- Step 3: Batch-delete tracked points from Qdrant by ID --
        let mut total_qdrant_deleted = 0u64;
        for (collection, point_ids) in &points_by_collection {
            if !ctx
                .storage_client
                .collection_exists(collection)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
            {
                warn!(
                    "Step 3: collection '{}' does not exist, skipping",
                    collection
                );
                continue;
            }

            for batch in point_ids.chunks(QDRANT_BATCH_SIZE) {
                let batch_vec: Vec<String> = batch.to_vec();
                let deleted = ctx
                    .storage_client
                    .delete_points_by_ids(collection, &batch_vec)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                total_qdrant_deleted += deleted;
            }

            info!(
                "Step 3: deleted {} points from '{}' for tenant={}",
                point_ids.len(),
                collection,
                item.tenant_id
            );
        }

        // -- Step 3b: Sweep libraries collection for orphaned/untracked points --
        if !points_by_collection.contains_key(COLLECTION_LIBRARIES) {
            if ctx
                .storage_client
                .collection_exists(COLLECTION_LIBRARIES)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
            {
                let library_ids = ctx
                    .storage_client
                    .scroll_point_ids_by_tenant(COLLECTION_LIBRARIES, &item.tenant_id)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

                if !library_ids.is_empty() {
                    for batch in library_ids.chunks(QDRANT_BATCH_SIZE) {
                        let batch_vec: Vec<String> = batch.to_vec();
                        ctx.storage_client
                            .delete_points_by_ids(COLLECTION_LIBRARIES, &batch_vec)
                            .await
                            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                    }
                    total_qdrant_deleted += library_ids.len() as u64;
                    info!(
                        "Step 3b: deleted {} orphaned library points for tenant={}",
                        library_ids.len(),
                        item.tenant_id
                    );
                }
            }
        }

        // -- Step 4: Handle memory collection --
        if ctx
            .storage_client
            .collection_exists(COLLECTION_MEMORY)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            let memory_ids = ctx
                .storage_client
                .scroll_point_ids_by_tenant(COLLECTION_MEMORY, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

            if !memory_ids.is_empty() {
                for batch in memory_ids.chunks(QDRANT_BATCH_SIZE) {
                    let batch_vec: Vec<String> = batch.to_vec();
                    ctx.storage_client
                        .delete_points_by_ids(COLLECTION_MEMORY, &batch_vec)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
                total_qdrant_deleted += memory_ids.len() as u64;
                info!(
                    "Step 4: deleted {} memory points for tenant={}",
                    memory_ids.len(),
                    item.tenant_id
                );
            }
        }

        // -- Step 5: Handle scratchpad collection --
        if ctx
            .storage_client
            .collection_exists(COLLECTION_SCRATCHPAD)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            let scratchpad_ids = ctx
                .storage_client
                .scroll_point_ids_by_tenant(COLLECTION_SCRATCHPAD, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

            if !scratchpad_ids.is_empty() {
                for batch in scratchpad_ids.chunks(QDRANT_BATCH_SIZE) {
                    let batch_vec: Vec<String> = batch.to_vec();
                    ctx.storage_client
                        .delete_points_by_ids(COLLECTION_SCRATCHPAD, &batch_vec)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
                total_qdrant_deleted += scratchpad_ids.len() as u64;
                info!(
                    "Step 5: deleted {} scratchpad points for tenant={}",
                    scratchpad_ids.len(),
                    item.tenant_id
                );
            }
        }

        // -- Step 6: SQLite cleanup in single transaction --
        let mut tx = pool.begin().await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to begin delete transaction: {}",
                e
            ))
        })?;

        Self::sqlite_cascade_delete(&mut tx, &item.tenant_id).await;

        tx.commit().await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!(
                "Failed to commit delete transaction: {}",
                e
            ))
        })?;

        // -- Step 6i: FTS5 cleanup --
        if let Some(sdb) = &ctx.search_db {
            let processor = FtsBatchProcessor::new(sdb, FtsBatchConfig::default());
            match processor.delete_tenant(&item.tenant_id).await {
                Ok(deleted) => {
                    if deleted > 0 {
                        info!(
                            "Step 6i: deleted {} FTS5 code_lines for tenant={}",
                            deleted, item.tenant_id
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Step 6i: FTS5 cleanup failed for tenant={}: {} (non-fatal)",
                        item.tenant_id, e
                    );
                }
            }
        }

        // -- Step 7: Final verification --
        let canonical = [
            COLLECTION_PROJECTS,
            COLLECTION_LIBRARIES,
            COLLECTION_MEMORY,
            COLLECTION_SCRATCHPAD,
        ];

        for coll in &canonical {
            if let Ok(true) = ctx.storage_client.collection_exists(coll).await {
                match ctx
                    .storage_client
                    .count_points(coll, Some(&item.tenant_id))
                    .await
                {
                    Ok(remaining) if remaining > 0 => {
                        warn!(
                            "Step 7: ORPHAN POINTS: {} remaining in '{}' for tenant={}",
                            remaining, coll, item.tenant_id
                        );
                    }
                    Ok(_) => {
                        debug!(
                            "Step 7: verified 0 remaining in '{}' for tenant={}",
                            coll, item.tenant_id
                        );
                    }
                    Err(e) => {
                        warn!("Step 7: verification failed for '{}': {}", coll, e);
                    }
                }
            }
        }

        info!(
            "Successfully deleted tenant={}: {} Qdrant points, {} tracked files (queue_id={})",
            item.tenant_id,
            total_qdrant_deleted,
            file_count.len(),
            item.queue_id
        );

        Ok(())
    }

    /// Run the SQLite cascade delete inside a transaction.
    ///
    /// Deletes: qdrant_chunks, tracked_files, keywords, keyword_baskets,
    /// tags, tag_hierarchy_edges, canonical_tags, watch_folders.
    async fn sqlite_cascade_delete(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        tenant_id: &str,
    ) {
        // 6a. Delete qdrant_chunks (child of tracked_files, through watch_folders)
        match sqlx::query(
            r#"DELETE FROM qdrant_chunks WHERE file_id IN (
                SELECT tf.file_id FROM tracked_files tf
                JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
                WHERE wf.tenant_id = ?1
            )"#,
        )
        .bind(tenant_id)
        .execute(&mut **tx)
        .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6a: deleted {} qdrant_chunks for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6a: qdrant_chunks delete: {}", e);
            }
        }

        // 6b. Delete tracked_files (through watch_folders)
        match sqlx::query(
            r#"DELETE FROM tracked_files WHERE watch_folder_id IN (
                SELECT watch_id FROM watch_folders WHERE tenant_id = ?1
            )"#,
        )
        .bind(tenant_id)
        .execute(&mut **tx)
        .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6b: deleted {} tracked_files for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6b: tracked_files delete: {}", e);
            }
        }

        // 6c. Delete keywords
        match sqlx::query("DELETE FROM keywords WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut **tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6c: deleted {} keywords for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6c: keywords delete: {}", e);
            }
        }

        // 6d. Delete keyword_baskets
        match sqlx::query("DELETE FROM keyword_baskets WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut **tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6d: deleted {} keyword_baskets for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6d: keyword_baskets delete: {}", e);
            }
        }

        // 6e. Delete tags
        match sqlx::query("DELETE FROM tags WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut **tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6e: deleted {} tags for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6e: tags delete: {}", e);
            }
        }

        // 6f. Delete tag_hierarchy_edges
        match sqlx::query("DELETE FROM tag_hierarchy_edges WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut **tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6f: deleted {} tag_hierarchy_edges for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6f: tag_hierarchy_edges delete: {}", e);
            }
        }

        // 6g. Delete canonical_tags
        match sqlx::query("DELETE FROM canonical_tags WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut **tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6g: deleted {} canonical_tags for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6g: canonical_tags delete: {}", e);
            }
        }

        // 6h. Delete watch_folders (last, after all children are removed)
        match sqlx::query("DELETE FROM watch_folders WHERE tenant_id = ?1")
            .bind(tenant_id)
            .execute(&mut **tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!(
                        "Step 6h: deleted {} watch_folders for tenant={}",
                        result.rows_affected(),
                        tenant_id
                    );
                }
            }
            Err(e) => {
                debug!("Step 6h: watch_folders delete: {}", e);
            }
        }
    }

    // =========================================================================
    // Document delete
    // =========================================================================

    /// Process delete document item -- delete specific document by document_id.
    async fn process_delete_document_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        info!("Processing delete document item: {}", item.queue_id);

        // Use typed payload deserialization (matches validation in queue_operations.rs)
        let payload = item.parse_delete_document_payload().map_err(|e| {
            UnifiedProcessorError::InvalidPayload(format!(
                "Failed to parse DeleteDocumentPayload: {}",
                e
            ))
        })?;

        if payload.document_id.trim().is_empty() {
            return Err(UnifiedProcessorError::InvalidPayload(
                "document_id must not be empty".to_string(),
            ));
        }

        if storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            storage_client
                .delete_points_by_document_id(&item.collection, &payload.document_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Successfully deleted document {} from {} (tenant={})",
            payload.document_id, item.collection, item.tenant_id
        );

        Ok(())
    }

    // =========================================================================
    // Tenant rename
    // =========================================================================

    /// Process tenant rename item -- update tenant_id on all matching Qdrant points.
    ///
    /// Uses `ProjectPayload` with `old_tenant_id` field.
    async fn process_tenant_rename_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        let payload: ProjectPayload =
            serde_json::from_str(&item.payload_json).map_err(|e| {
                UnifiedProcessorError::InvalidPayload(format!(
                    "Failed to parse ProjectPayload for rename: {}",
                    e
                ))
            })?;

        let old_tenant = payload.old_tenant_id.as_deref().ok_or_else(|| {
            UnifiedProcessorError::InvalidPayload(
                "Missing old_tenant_id in tenant rename payload".to_string(),
            )
        })?;
        let new_tenant = &item.tenant_id;

        // Extract reason from metadata if available
        let reason = item
            .metadata
            .as_deref()
            .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
            .and_then(|v| v.get("reason").and_then(|r| r.as_str().map(String::from)))
            .unwrap_or_else(|| "unknown".to_string());

        info!(
            "Processing tenant rename: {} -> {} in collection '{}' (reason: {})",
            old_tenant, new_tenant, item.collection, reason
        );

        use qdrant_client::qdrant::{Condition, Filter};
        let filter = Filter::must([Condition::matches(
            "tenant_id",
            old_tenant.to_string(),
        )]);

        let mut new_payload = std::collections::HashMap::new();
        new_payload.insert(
            "tenant_id".to_string(),
            serde_json::Value::String(new_tenant.to_string()),
        );

        storage_client
            .set_payload_by_filter(&item.collection, filter, new_payload)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed tenant rename {} -> {} in '{}'",
            old_tenant, new_tenant, item.collection
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_strategy_handles_tenant_items() {
        let strategy = TenantStrategy;
        assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Add));
        assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Scan));
        assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Delete));
        assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Rename));
    }

    #[test]
    fn test_tenant_strategy_handles_doc_items() {
        let strategy = TenantStrategy;
        assert!(strategy.handles(&ItemType::Doc, &QueueOperation::Delete));
        assert!(strategy.handles(&ItemType::Doc, &QueueOperation::Uplift));
    }

    #[test]
    fn test_tenant_strategy_rejects_non_tenant_items() {
        let strategy = TenantStrategy;
        assert!(!strategy.handles(&ItemType::File, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Folder, &QueueOperation::Scan));
    }

    #[test]
    fn test_tenant_strategy_name() {
        let strategy = TenantStrategy;
        assert_eq!(strategy.name(), "tenant");
    }

    /// Test that the exclusion check logic correctly identifies files that should be cleaned up.
    /// This tests the core decision logic used by cleanup_excluded_files without needing
    /// Qdrant or SQLite connections.
    #[test]
    fn test_cleanup_exclusion_logic_identifies_hidden_files() {
        let project_root = Path::new("/home/user/project");

        // Simulate file paths as they would be stored in Qdrant (absolute paths)
        let qdrant_paths = vec![
            "/home/user/project/src/main.rs",
            "/home/user/project/.hidden_file",
            "/home/user/project/src/.secret",
            "/home/user/project/.git/config",
            "/home/user/project/src/lib.rs",
            "/home/user/project/node_modules/package/index.js",
            "/home/user/project/.env",
            "/home/user/project/README.md",
            "/home/user/project/src/.cache/data",
            "/home/user/project/.github/workflows/ci.yml",
        ];

        let mut should_delete = Vec::new();
        let mut should_keep = Vec::new();

        for qdrant_file in &qdrant_paths {
            let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
                Ok(stripped) => stripped.to_string_lossy().to_string(),
                Err(_) => qdrant_file.to_string(),
            };

            if should_exclude_file(&rel_path) {
                should_delete.push(qdrant_file.to_string());
            } else {
                should_keep.push(qdrant_file.to_string());
            }
        }

        // Hidden files should be marked for deletion
        assert!(
            should_delete.contains(&"/home/user/project/.hidden_file".to_string()),
            "Expected .hidden_file to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/src/.secret".to_string()),
            "Expected src/.secret to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/.git/config".to_string()),
            "Expected .git/config to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/.env".to_string()),
            "Expected .env to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/src/.cache/data".to_string()),
            "Expected src/.cache/data to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/node_modules/package/index.js".to_string()),
            "Expected node_modules content to be excluded"
        );

        // Normal files should NOT be deleted
        assert!(
            should_keep.contains(&"/home/user/project/src/main.rs".to_string()),
            "Expected src/main.rs to be kept"
        );
        assert!(
            should_keep.contains(&"/home/user/project/src/lib.rs".to_string()),
            "Expected src/lib.rs to be kept"
        );
        assert!(
            should_keep.contains(&"/home/user/project/README.md".to_string()),
            "Expected README.md to be kept"
        );

        // .github/ should be whitelisted (not excluded)
        assert!(
            should_keep.contains(&"/home/user/project/.github/workflows/ci.yml".to_string()),
            "Expected .github/workflows/ci.yml to be kept (whitelisted)"
        );
    }

    #[test]
    fn test_cleanup_exclusion_logic_with_non_strippable_paths() {
        // Test when Qdrant paths don't share the project root prefix
        let project_root = Path::new("/home/user/project");
        let qdrant_file = "/different/root/src/.hidden";

        let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
            Ok(stripped) => stripped.to_string_lossy().to_string(),
            Err(_) => qdrant_file.to_string(),
        };

        // Should still detect hidden component even with full path fallback
        assert!(
            should_exclude_file(&rel_path),
            "Expected .hidden to be excluded even when path can't be stripped"
        );
    }

    #[test]
    fn test_cleanup_exclusion_logic_empty_paths() {
        // Verify no panic with edge cases
        let project_root = Path::new("/home/user/project");
        let qdrant_paths: Vec<String> = vec![];

        let mut count = 0u64;
        for qdrant_file in &qdrant_paths {
            let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
                Ok(stripped) => stripped.to_string_lossy().to_string(),
                Err(_) => qdrant_file.clone(),
            };

            if should_exclude_file(&rel_path) {
                count += 1;
            }
        }

        assert_eq!(count, 0, "Empty path list should produce zero deletions");
    }
}
