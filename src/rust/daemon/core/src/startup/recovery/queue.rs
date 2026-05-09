//! Queue helpers: enqueue progressive scans and file operations.

use std::path::Path;

use crate::file_classification::classify_file_type;
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation};

use super::types::RecoveryStats;

/// Enqueue a progressive scan for async file discovery.
pub(super) async fn enqueue_progressive_scan(
    queue_manager: &QueueManager,
    root: &Path,
    tenant_id: &str,
    collection: &str,
    stats: &mut RecoveryStats,
) -> Result<(), String> {
    let scan_payload = serde_json::json!({
        "project_root": root.to_string_lossy(),
        "recovery": true,
    })
    .to_string();

    let branch = crate::watching_queue::get_current_branch(root);

    queue_manager
        .enqueue_unified(
            ItemType::Tenant,
            QueueOperation::Scan,
            tenant_id,
            collection,
            &scan_payload,
            Some(&branch),
            None,
        )
        .await
        .map(|_| ())
        .map_err(|e| format!("Failed to enqueue progressive scan: {}", e))?;

    stats.progressive_scans_enqueued += 1;
    Ok(())
}

/// Enqueue a file operation (ingest, update, or delete).
pub(super) async fn enqueue_file_op(
    queue_manager: &QueueManager,
    tenant_id: &str,
    collection: &str,
    abs_file_path: &str,
    op: QueueOperation,
    metadata: Option<&str>,
) -> Result<(), String> {
    let file_type = if op != QueueOperation::Delete {
        Some(
            classify_file_type(Path::new(abs_file_path))
                .as_str()
                .to_string(),
        )
    } else {
        None
    };

    let file_payload = FilePayload {
        file_path: abs_file_path.to_string(),
        file_type,
        file_hash: None,
        size_bytes: None,
        old_path: None,
    };

    let payload_json = serde_json::to_string(&file_payload)
        .map_err(|e| format!("Failed to serialize FilePayload: {}", e))?;

    let branch = crate::watching_queue::get_current_branch(Path::new(abs_file_path));

    queue_manager
        .enqueue_unified(
            ItemType::File,
            op,
            tenant_id,
            collection,
            &payload_json,
            Some(&branch),
            metadata,
        )
        .await
        .map(|_| ())
        .map_err(|e| format!("Failed to enqueue: {}", e))
}
