//! Project operations for the tenant strategy.
//!
//! Handles `ItemType::Tenant` items routed to the `projects` collection:
//! Add (create watch_folder + enqueue scan), Scan (directory enumeration),
//! Delete (tenant-scoped point removal), and Uplift (cascade to per-file items).

use std::path::Path;

use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::specs::parse_payload;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{ItemType, ProjectPayload, QueueOperation, UnifiedQueueItem};
use wqm_common::constants::COLLECTION_PROJECTS;

use super::grammar_warm;

/// Process project item -- create/manage project collections.
pub(crate) async fn process_project_item(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    info!(
        "Processing project item: {} (op={:?})",
        item.queue_id, item.op
    );

    let payload: ProjectPayload = parse_payload(item)?;

    match item.op {
        QueueOperation::Add => {
            handle_project_add(ctx, item, &payload).await?;
        }
        QueueOperation::Scan => {
            handle_project_scan(ctx, item, &payload).await?;
        }
        QueueOperation::Delete => {
            handle_project_delete(ctx, item).await?;
        }
        QueueOperation::Uplift => {
            handle_project_uplift(ctx, item).await?;
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

/// Handle Tenant/Add for the projects collection.
///
/// Creates the Qdrant collection (idempotent), inserts a watch_folder row,
/// and enqueues a follow-up (Tenant, Scan) item.
async fn handle_project_add(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
) -> UnifiedProcessorResult<()> {
    crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    let git_status = crate::git::detect_git_status(std::path::Path::new(&payload.project_root));
    insert_watch_folder(ctx, item, payload, &git_status).await?;
    enqueue_project_scan(ctx, item, payload).await;

    // Spawn background grammar pre-warming for this project's languages.
    // Non-blocking: project registration returns immediately.
    if let Some(ref gm) = ctx.grammar_manager {
        grammar_warm::spawn_grammar_warming(gm.clone(), payload.project_root.clone());
    }

    Ok(())
}

async fn insert_watch_folder(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
    git_status: &crate::git::GitStatus,
) -> UnifiedProcessorResult<()> {
    // Guard: reject registering a subdirectory of an already-registered project.
    // This prevents test fixtures, language directories, etc. from becoming
    // spurious watch folders.
    let is_subdirectory: bool = sqlx::query_scalar(
        r#"SELECT COUNT(*) > 0 FROM watch_folders
           WHERE collection = ?1
             AND ?2 != path
             AND ?2 LIKE path || '/' || '%'"#,
    )
    .bind(COLLECTION_PROJECTS)
    .bind(&payload.project_root)
    .fetch_one(ctx.queue_manager.pool())
    .await
    .unwrap_or(false);

    if is_subdirectory {
        info!(
            "Skipping watch folder creation: {} is a subdirectory of an already-registered project",
            payload.project_root
        );
        return Ok(());
    }

    let now = wqm_common::timestamps::now_utc();
    let watch_id = uuid::Uuid::new_v4().to_string();
    let is_active: i32 = if payload.is_active.unwrap_or(false) {
        1
    } else {
        0
    };

    let result = sqlx::query(
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
    .await
    .map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!("Failed to create watch_folder: {}", e))
    })?;

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
    Ok(())
}

async fn enqueue_project_scan(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
) {
    let scan_payload_json = match serde_json::to_string(payload) {
        Ok(s) => s,
        Err(e) => {
            warn!(
                "Failed to serialize scan payload for tenant={}: {} (non-critical)",
                item.tenant_id, e
            );
            return;
        }
    };

    match ctx
        .queue_manager
        .enqueue_unified(
            ItemType::Tenant,
            QueueOperation::Scan,
            &item.tenant_id,
            &item.collection,
            &scan_payload_json,
            None,
            None,
        )
        .await
    {
        Ok((queue_id, true)) => {
            info!(
                "Enqueued project scan for tenant={} queue_id={}",
                item.tenant_id, queue_id
            );
        }
        Ok((_, false)) => {}
        Err(e) => {
            warn!(
                "Failed to enqueue project scan for tenant={}: {} (non-critical)",
                item.tenant_id, e
            );
        }
    }
}

/// Handle Tenant/Scan for the projects collection.
///
/// Enumerates the project directory (single-level progressive scan) and
/// queues file ingestion items, then cleans up excluded files.
async fn handle_project_scan(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
) -> UnifiedProcessorResult<()> {
    scan_project_directory(ctx, item, payload).await?;

    // Update last_scan timestamp for this project's watch_folder
    let update_result = sqlx::query(
        "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
         updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE tenant_id = ?1 AND collection = ?2",
    )
    .bind(&item.tenant_id)
    .bind(COLLECTION_PROJECTS)
    .execute(ctx.queue_manager.pool())
    .await;

    match update_result {
        Ok(result) => {
            if result.rows_affected() > 0 {
                info!("Updated last_scan for project tenant_id={}", item.tenant_id);
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

    Ok(())
}

/// Handle Tenant/Delete scoped to the projects collection.
///
/// Full cleanup: cancel pending queue items, delete tracked_files,
/// delete watch_folders (including child folders), delete Qdrant points.
async fn handle_project_delete(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    let pool = ctx.queue_manager.pool();

    // 1. Cancel all pending/in_progress queue items for this tenant+collection
    let cancelled = sqlx::query(
        "UPDATE unified_queue SET status = 'cancelled' \
         WHERE tenant_id = ?1 AND collection = ?2 \
         AND status IN ('pending', 'in_progress') \
         AND queue_id != ?3",
    )
    .bind(&item.tenant_id)
    .bind(&item.collection)
    .bind(&item.queue_id)
    .execute(pool)
    .await
    .map(|r| r.rows_affected())
    .unwrap_or(0);

    if cancelled > 0 {
        info!(
            "Cancelled {} pending queue items for tenant={}",
            cancelled, item.tenant_id
        );
    }

    // 2. Delete tracked_files for this tenant
    let files_deleted = sqlx::query("DELETE FROM tracked_files WHERE tenant_id = ?1")
        .bind(&item.tenant_id)
        .execute(pool)
        .await
        .map(|r| r.rows_affected())
        .unwrap_or(0);

    info!(
        "Deleted {} tracked_files for tenant={}",
        files_deleted, item.tenant_id
    );

    // 3. Delete watch_folders for this tenant+collection (children first via CASCADE,
    //    but also explicitly delete children in case CASCADE isn't configured)
    let folders_deleted =
        sqlx::query("DELETE FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2")
            .bind(&item.tenant_id)
            .bind(&item.collection)
            .execute(pool)
            .await
            .map(|r| r.rows_affected())
            .unwrap_or(0);

    info!(
        "Deleted {} watch_folders for tenant={}",
        folders_deleted, item.tenant_id
    );

    // 4. Delete Qdrant points
    if ctx
        .storage_client
        .collection_exists(&item.collection)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
    {
        info!(
            "Deleting Qdrant points for tenant={} from collection={}",
            item.tenant_id, item.collection
        );
        ctx.storage_client
            .delete_points_by_tenant(&item.collection, &item.tenant_id)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    }

    info!(
        "Project delete complete: tenant={}, cancelled={} queue items, \
         deleted={} files, {} folders",
        item.tenant_id, cancelled, files_deleted, folders_deleted
    );

    Ok(())
}

/// Handle Tenant/Uplift: cascade to per-file (Doc, Uplift) items.
async fn handle_project_uplift(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    let files: Vec<(String, String)> =
        sqlx::query_as("SELECT file_id, file_path FROM tracked_files WHERE tenant_id = ?1")
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
    let files_cleaned = super::cleanup::cleanup_excluded_files(
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

