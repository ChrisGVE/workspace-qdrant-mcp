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
use super::project_worktree::resolve_worktree_info;

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
/// enqueues a follow-up (Tenant, Scan) item, and updates project grouping
/// (workspace membership and git org).
async fn handle_project_add(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
) -> UnifiedProcessorResult<()> {
    crate::shared::ensure_collection(
        &ctx.storage_client,
        &item.collection,
        ctx.embedding_generator.dense_dim() as u64,
    )
    .await
    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    let git_status = crate::git::detect_git_status(std::path::Path::new(&payload.project_root));
    insert_watch_folder(ctx, item, payload, &git_status).await?;
    enqueue_project_scan(ctx, item, payload).await;

    // Update project grouping (non-critical side effects).
    update_project_groups(ctx, item, payload).await;

    // Spawn background grammar pre-warming for this project's languages.
    // Non-blocking: project registration returns immediately.
    if let Some(ref gm) = ctx.grammar_manager {
        grammar_warm::spawn_grammar_warming(gm.clone(), payload.project_root.clone());
    }

    Ok(())
}

/// Update workspace and git org group memberships for a newly registered project.
///
/// Both operations are non-critical: failures are logged as warnings but do not
/// block the project registration pipeline.
async fn update_project_groups(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
) {
    let pool = ctx.queue_manager.pool();
    let project_path = Path::new(&payload.project_root);

    // Workspace grouping: detect Cargo/npm/Go workspace membership and link
    // this project with sibling workspace members in project_groups.
    match crate::workspace_grouper::update_project_workspace_group(
        pool,
        &item.tenant_id,
        project_path,
    )
    .await
    {
        Ok(true) => {
            info!(
                "Updated workspace group for tenant={} path={}",
                item.tenant_id, payload.project_root
            );
        }
        Ok(false) => {
            debug!(
                "No workspace detected for tenant={} path={}",
                item.tenant_id, payload.project_root
            );
        }
        Err(e) => {
            warn!(
                "Failed to update workspace group for tenant={}: {} (non-critical)",
                item.tenant_id, e
            );
        }
    }

    // Git org grouping: extract org/user from the git remote and link this
    // project with other projects sharing the same org.
    if let Some(ref remote_url) = payload.git_remote {
        match crate::git_org_grouper::update_project_org_group(pool, &item.tenant_id, remote_url)
            .await
        {
            Ok(true) => {
                info!(
                    "Updated git org group for tenant={} remote={}",
                    item.tenant_id, remote_url
                );
            }
            Ok(false) => {
                debug!(
                    "No org extractable for tenant={} remote={}",
                    item.tenant_id, remote_url
                );
            }
            Err(e) => {
                warn!(
                    "Failed to update git org group for tenant={}: {} (non-critical)",
                    item.tenant_id, e
                );
            }
        }
    }
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

    // Detect worktree status and resolve main worktree watch_id if applicable
    let (is_worktree, main_worktree_watch_id) =
        resolve_worktree_info(ctx, item, payload, git_status).await;

    let result = sqlx::query(
        r#"INSERT OR IGNORE INTO watch_folders (
            watch_id, path, collection, tenant_id, is_active,
            git_remote_url, last_activity_at, follow_symlinks, enabled,
            cleanup_on_disable, is_git_tracked, last_commit_hash,
            is_worktree, main_worktree_watch_id,
            created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 1, 0, ?8, ?9, ?10, ?11, ?7, ?7)"#,
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
    .bind(is_worktree as i32)
    .bind(&main_worktree_watch_id)
    .execute(ctx.queue_manager.pool())
    .await
    .map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!("Failed to create watch_folder: {}", e))
    })?;

    if result.rows_affected() > 0 {
        info!(
            "Created watch_folder for tenant={} path={} \
             (active={}, git={}, branch={}, worktree={}, main_wt_id={:?})",
            item.tenant_id,
            payload.project_root,
            is_active,
            git_status.is_git,
            git_status.branch,
            is_worktree,
            main_worktree_watch_id,
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
/// Reads `last_scan` from the watch_folder record before scanning so that
/// mtime-based pruning can skip files unchanged since the previous scan.
/// When `payload.rebuild` is true, `last_scan` is ignored to force full
/// re-discovery of all files.
/// Enumerates the project directory (single-level progressive scan) and
/// queues file ingestion items, then cleans up excluded files.
async fn handle_project_scan(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
) -> UnifiedProcessorResult<()> {
    let last_scan: Option<String> = if payload.rebuild {
        info!(
            "Rebuild flag set — ignoring last_scan for tenant_id={}",
            item.tenant_id
        );
        None
    } else {
        sqlx::query_scalar(
            "SELECT last_scan FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2",
        )
        .bind(&item.tenant_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(ctx.queue_manager.pool())
        .await
        .unwrap_or(None)
        .flatten()
    };

    scan_project_directory(ctx, item, payload, last_scan.as_deref()).await?;

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
/// delete watch_folders (including child folders), remove git-org group
/// membership, and delete Qdrant points.
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

    // 2. Delete tracked_files for this tenant. tracked_files has no tenant_id
    //    column — tenancy is reached through watch_folders (must run BEFORE
    //    step 3 deletes the watch_folders rows the subquery resolves against).
    let files_deleted = sqlx::query(
        "DELETE FROM tracked_files WHERE watch_folder_id IN \
         (SELECT watch_id FROM watch_folders WHERE tenant_id = ?1)",
    )
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

    // 4. Remove project from all groups (workspace, git_org, dependency, affinity)
    if let Err(e) = sqlx::query("DELETE FROM project_groups WHERE tenant_id = ?1")
        .bind(&item.tenant_id)
        .execute(pool)
        .await
    {
        warn!(
            "Failed to remove tenant={} from project groups: {} (non-critical)",
            item.tenant_id, e
        );
    }

    // 5. Delete Qdrant points
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
    // tracked_files has no tenant_id/file_path columns — tenancy goes through
    // watch_folders and paths are stored watch-root-relative.
    let files: Vec<(i64, String)> = sqlx::query_as(
        "SELECT tf.file_id, tf.relative_path FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = ?1",
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
    for (file_id, relative_path) in &files {
        let doc_payload = serde_json::json!({
            "file_id": file_id,
            "file_path": relative_path,
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
///
/// `last_scan` is the ISO 8601 timestamp from the watch_folder record.
/// It is threaded into the scan so files unchanged since that time are
/// skipped, avoiding redundant embedding work on restart.
async fn scan_project_directory(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &ProjectPayload,
    last_scan: Option<&str>,
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

    crate::shared::ensure_collection(
        &ctx.storage_client,
        &item.collection,
        ctx.embedding_generator.dense_dim() as u64,
    )
    .await
    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    info!(
        "Scanning project directory (single level): {} (tenant_id={}, last_scan={:?})",
        payload.project_root, item.tenant_id, last_scan
    );

    // Parse the project_root canonical form so the scanner can compute
    // relative paths for enqueued subdirectories without re-querying the
    // database. `payload.project_root` originates from watch_folders.path
    // which is canonical by construction (handler-validated at registration).
    let watch_folder_root =
        wqm_common::paths::CanonicalPath::from_user_input(&payload.project_root).map_err(|e| {
            UnifiedProcessorError::InvalidPayload(format!(
                "project_root {} is not canonical: {}",
                payload.project_root, e
            ))
        })?;

    // Progressive scan: enumerate only the immediate children of the directory.
    // Subdirectories are enqueued as (Folder, Scan) for deferred processing.
    // last_scan enables mtime pruning to skip unchanged files.
    let (files_queued, dirs_queued, files_excluded, errors) =
        crate::strategies::processing::folder::FolderStrategy::scan_directory_single_level(
            project_root,
            &watch_folder_root,
            item,
            &ctx.queue_manager,
            &ctx.allowed_extensions,
            last_scan,
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
