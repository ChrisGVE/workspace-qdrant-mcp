//! Update preamble: hash comparison and reference-counted old point deletion.
//!
//! Called when `QueueOperation::Update` is in effect. Compares the new file
//! hash against the tracked record, stores a `QueueDecision` for retry-safe
//! execution, and deletes old Qdrant points only if no other watch folder
//! references the same base_point.

use std::time::Instant;

use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::processing_timings::{self, PhaseTiming};
use crate::tracked_files_schema;
use crate::tree_sitter::detect_language;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FilePayload, UnifiedQueueItem};

/// Execute the deletion part of an update operation (reference-counted).
///
/// Called after hash comparison determines the file has changed.
/// Stores a `QueueDecision` for retry-safe execution and deletes old points
/// only if no other watch folder references the same base_point.
#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_update_deletion(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
    payload: &FilePayload,
    existing: &tracked_files_schema::TrackedFile,
    new_hash: &str,
) -> UnifiedProcessorResult<()> {
    let preamble_start = Instant::now();

    let new_base_point =
        wqm_common::hashing::compute_base_point(&item.tenant_id, relative_path, new_hash);

    let delete_old = resolve_delete_old(ctx, existing, watch_folder_id).await;

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
        warn!("Failed to store QueueDecision for {}: {}", item.queue_id, e);
    }

    if delete_old {
        let old_point_ids = tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
            .await
            .unwrap_or_default();
        if !old_point_ids.is_empty() {
            ctx.storage_client
                .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }
    }

    record_preamble_timing(pool, item, payload, abs_file_path, preamble_start).await;
    Ok(())
}

/// Determine whether old Qdrant points should be deleted (reference-counted).
async fn resolve_delete_old(
    ctx: &ProcessingContext,
    existing: &tracked_files_schema::TrackedFile,
    watch_folder_id: &str,
) -> bool {
    if let Some(ref old_bp) = existing.base_point {
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
        true
    }
}

async fn record_preamble_timing(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
    payload: &FilePayload,
    abs_file_path: &str,
    preamble_start: Instant,
) {
    let detected_language = detect_language(std::path::Path::new(abs_file_path));
    processing_timings::record_timings(
        pool,
        &item.queue_id,
        item.item_type.as_str(),
        "update_preamble",
        &item.tenant_id,
        &item.collection,
        detected_language,
        payload.file_type.as_deref(),
        // Preamble updates rewrite metadata only — no embedding is computed.
        None,
        &[PhaseTiming {
            phase: "update_preamble",
            duration_ms: preamble_start.elapsed().as_millis() as u64,
        }],
    )
    .await;
}
