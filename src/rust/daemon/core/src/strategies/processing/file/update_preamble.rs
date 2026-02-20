//! Update preamble: hash comparison and reference-counted old point deletion.
//!
//! Called when `QueueOperation::Update` is in effect. Compares the new file
//! hash against the tracked record, stores a `QueueDecision` for retry-safe
//! execution, and deletes old Qdrant points only if no other watch folder
//! references the same base_point.

use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FilePayload, UnifiedQueueItem};

/// Execute the deletion part of an update operation (reference-counted).
///
/// Called after hash comparison determines the file has changed.
/// Stores a `QueueDecision` for retry-safe execution and deletes old points
/// only if no other watch folder references the same base_point.
pub(super) async fn execute_update_deletion(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    payload: &FilePayload,
    existing: &tracked_files_schema::TrackedFile,
    new_hash: &str,
) -> UnifiedProcessorResult<()> {
    // Compute new base_point for comparison
    let new_base_point = wqm_common::hashing::compute_base_point(
        &item.tenant_id,
        &item.branch,
        relative_path,
        new_hash,
    );

    // Reference-counted deletion: check if another watch folder still references the old base_point
    let delete_old = if let Some(ref old_bp) = existing.base_point {
        let has_refs = ctx
            .queue_manager
            .has_other_references(old_bp, watch_folder_id)
            .await
            .unwrap_or(false);
        if has_refs {
            info!(
                "Old base_point {} still referenced by another watch folder, skipping Qdrant deletion",
                old_bp
            );
        }
        !has_refs
    } else {
        // No old base_point recorded -- safe to delete by filter
        true
    };

    // Build and store QueueDecision for retry-safe execution
    let decision = wqm_common::queue_types::QueueDecision {
        delete_old,
        old_base_point: existing.base_point.clone(),
        new_base_point: new_base_point.clone(),
        old_file_hash: Some(existing.file_hash.clone()),
        new_file_hash: new_hash.to_string(),
    };

    if let Err(e) = ctx
        .queue_manager
        .store_queue_decision(&item.queue_id, &decision)
        .await
    {
        warn!(
            "Failed to store QueueDecision for {}: {}",
            item.queue_id, e
        );
        // Non-fatal: proceed without stored decision
    }

    // Execute old point deletion only if no other references
    if delete_old {
        let old_point_ids =
            tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
                .await
                .unwrap_or_default();
        if !old_point_ids.is_empty() {
            ctx.storage_client
                .delete_points_by_filter(
                    &item.collection,
                    &payload.file_path,
                    &item.tenant_id,
                )
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }
    }
    // Old chunk records will be cleaned up atomically in the transaction below

    Ok(())
}
