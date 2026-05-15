//! Queue operations for branch switch: enqueueing file changes and tenant scans.

use std::path::Path;

use wqm_common::paths::{CanonicalPath, RelativePath};

use crate::git::{FileChange, FileChangeStatus};
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation};
use crate::watching_queue::get_current_branch;

/// Enqueue a single changed file based on its diff-tree status.
/// Returns the operation type that was enqueued.
pub async fn enqueue_changed_file(
    queue_manager: &QueueManager,
    change: &FileChange,
    tenant_id: &str,
    collection: &str,
    project_root: &str,
    branch: &str,
) -> Result<QueueOperation, String> {
    let (op, old_rel_opt) = match &change.status {
        FileChangeStatus::Modified => (QueueOperation::Update, None),
        FileChangeStatus::Added => (QueueOperation::Add, None),
        FileChangeStatus::Deleted => (QueueOperation::Delete, None),
        FileChangeStatus::Renamed { old_path, .. } => {
            // Delete old path, add new path. Both paths come from git diff-tree
            // and are repository-relative — exactly what RelativePath needs.
            let old_rel = RelativePath::from_user_input(old_path)
                .map_err(|e| format!("invalid old_path from diff-tree {:?}: {}", old_path, e))?;
            enqueue_file_op_rel(
                queue_manager,
                tenant_id,
                collection,
                &old_rel,
                QueueOperation::Delete,
                branch,
            )
            .await?;
            (QueueOperation::Add, Some(old_rel))
        }
        FileChangeStatus::Copied { .. } => (QueueOperation::Add, None),
        FileChangeStatus::TypeChanged => (QueueOperation::Update, None),
    };

    // `change.path` is the diff-tree-reported path, repository-relative.
    let new_rel = RelativePath::from_user_input(&change.path).map_err(|e| {
        format!(
            "invalid change.path from diff-tree {:?}: {}",
            change.path, e
        )
    })?;
    enqueue_file_op_rel(
        queue_manager,
        tenant_id,
        collection,
        &new_rel,
        op.clone(),
        branch,
    )
    .await?;

    // For rename, report as Update for stats (it's logically an update, just with path change)
    if old_rel_opt.is_some() {
        return Ok(QueueOperation::Update);
    }

    // Project root is unused on this path now that we don't reconstruct
    // absolute paths; keep the parameter for caller signature stability
    // but acknowledge it's silently unused via `_`.
    let _ = project_root;
    Ok(op)
}

/// Enqueue a file operation to the unified queue (absolute path entry point).
///
/// `abs_file_path` MUST live under a watch_folder root whose path matches
/// the on-disk prefix; the function derives the relative form for the
/// payload by stripping that root.
#[allow(dead_code)]
pub async fn enqueue_file_op(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    abs_file_path: &str,
    op: QueueOperation,
    branch: &str,
) -> Result<(), String> {
    // Look up the watch_folder root to anchor the path.
    let root = lookup_watch_folder_root(queue_manager, tenant_id, collection)
        .await?
        .ok_or_else(|| {
            format!(
                "No watch_folder found for tenant_id={}, collection={} -- cannot anchor file path",
                tenant_id, collection
            )
        })?;
    let abs = CanonicalPath::from_user_input(abs_file_path)
        .map_err(|e| format!("invalid absolute path {:?}: {}", abs_file_path, e))?;
    let rel = RelativePath::from_absolute_and_root(&abs, &root).map_err(|e| {
        format!(
            "file {} is not under watch_folder root {}: {}",
            abs_file_path,
            root.as_str(),
            e
        )
    })?;
    enqueue_file_op_rel(queue_manager, tenant_id, collection, &rel, op, branch).await
}

/// Internal: build the FilePayload from a pre-validated [`RelativePath`].
async fn enqueue_file_op_rel(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    rel: &RelativePath,
    op: QueueOperation,
    branch: &str,
) -> Result<(), String> {
    let file_payload = FilePayload {
        file_path: rel.clone(),
        file_type: None,
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };

    let payload_json = serde_json::to_string(&file_payload)
        .map_err(|e| format!("Failed to serialize FilePayload: {}", e))?;

    queue_manager
        .enqueue_unified(
            ItemType::File,
            op,
            tenant_id,
            collection,
            &payload_json,
            Some(branch),
            None,
        )
        .await
        .map(|_| ())
        .map_err(|e| format!("Failed to enqueue: {}", e))
}

/// Lookup the watch_folder canonical root for (tenant_id, collection).
///
/// Returns `Ok(None)` if no watch_folder row exists. Returns `Err` if the
/// stored path fails canonical validation.
#[allow(dead_code)]
async fn lookup_watch_folder_root(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
) -> Result<Option<CanonicalPath>, String> {
    let row: Option<String> = sqlx::query_scalar(
        "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
    )
    .bind(tenant_id)
    .bind(collection)
    .fetch_optional(queue_manager.pool())
    .await
    .map_err(|e| format!("Failed to lookup watch_folder: {}", e))?;
    row.map(|p| {
        CanonicalPath::from_user_input(&p)
            .map_err(|e| format!("watch_folder.path is not canonical: {}", e))
    })
    .transpose()
}

/// Enqueue a full tenant scan (used for reset events).
pub async fn enqueue_tenant_scan(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    project_root: &str,
) -> Result<(), String> {
    let payload = serde_json::json!({
        "project_root": project_root,
        "recovery": false,
    })
    .to_string();

    let branch = get_current_branch(Path::new(project_root));

    queue_manager
        .enqueue_unified(
            ItemType::Tenant,
            QueueOperation::Scan,
            tenant_id,
            collection,
            &payload,
            Some(&branch),
            None,
        )
        .await
        .map(|_| ())
        .map_err(|e| format!("Failed to enqueue tenant scan: {}", e))
}
