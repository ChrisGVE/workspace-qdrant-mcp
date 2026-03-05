//! Queue operations for branch switch: enqueueing file changes and tenant scans.

use std::path::Path;

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
    let abs_path = Path::new(project_root).join(&change.path);
    let abs_str = abs_path.to_string_lossy().to_string();

    let (op, old_path) = match &change.status {
        FileChangeStatus::Modified => (QueueOperation::Update, None),
        FileChangeStatus::Added => (QueueOperation::Add, None),
        FileChangeStatus::Deleted => (QueueOperation::Delete, None),
        FileChangeStatus::Renamed { old_path, .. } => {
            // Delete old path, add new path
            let old_abs = Path::new(project_root).join(old_path);
            enqueue_file_op(
                queue_manager,
                tenant_id,
                collection,
                &old_abs.to_string_lossy(),
                QueueOperation::Delete,
                branch,
            )
            .await?;
            (QueueOperation::Add, Some(old_path.clone()))
        }
        FileChangeStatus::Copied { .. } => (QueueOperation::Add, None),
        FileChangeStatus::TypeChanged => (QueueOperation::Update, None),
    };

    enqueue_file_op(
        queue_manager,
        tenant_id,
        collection,
        &abs_str,
        op.clone(),
        branch,
    )
    .await?;

    // For rename, report as Update for stats (it's logically an update, just with path change)
    if old_path.is_some() {
        return Ok(QueueOperation::Update);
    }

    Ok(op)
}

/// Enqueue a file operation to the unified queue.
pub async fn enqueue_file_op(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    abs_file_path: &str,
    op: QueueOperation,
    branch: &str,
) -> Result<(), String> {
    let file_payload = FilePayload {
        file_path: abs_file_path.to_string(),
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
