//! File delete processing.
//!
//! Handles `QueueOperation::Delete` for file items: reference-counted Qdrant
//! point deletion, tracked_files cleanup, FTS5 cleanup, and missing-file
//! reconciliation.

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FilePayload, UnifiedQueueItem};

/// Process file delete operation with tracked_files awareness (Task 506 + Task 519).
pub(super) async fn process_file_delete(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
) -> UnifiedProcessorResult<()> {
    let delete_start = std::time::Instant::now();

    // Safety net: check incremental flag before deleting
    if let Ok(true) = tracked_files_schema::is_incremental(pool, abs_file_path).await {
        info!(
            "Skipping delete for incremental file (safety net): {}",
            abs_file_path
        );
        return Ok(());
    }

    // Try tracked_files lookup first for precise deletion
    if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    {
        // Reference-counted deletion: check if another watch folder still references this base_point
        let delete_from_qdrant = if let Some(ref bp) = existing.base_point {
            let has_refs = ctx
                .queue_manager
                .has_other_references(bp, watch_folder_id)
                .await
                .unwrap_or(false);
            if has_refs {
                info!(
                    "base_point {} still referenced by another watch folder, skipping Qdrant deletion for: {}",
                    bp, relative_path
                );
            }
            !has_refs
        } else {
            true // No base_point recorded -- safe to delete
        };

        if delete_from_qdrant {
            // Get point_ids from qdrant_chunks for targeted deletion
            let point_ids =
                tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
                    .await
                    .unwrap_or_default();

            if !point_ids.is_empty() {
                // Delete from Qdrant first (irreversible), scoped to tenant
                ctx.storage_client
                    .delete_points_by_filter(
                        &item.collection,
                        abs_file_path,
                        &item.tenant_id,
                    )
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            }
        }

        // Always clean up SQLite records (this watch folder's tracked entry)
        // CASCADE handles qdrant_chunks
        let tx_result: Result<(), UnifiedProcessorError> = async {
            let mut tx = pool.begin().await.map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;
            tracked_files_schema::delete_tracked_file_tx(&mut tx, existing.file_id)
                .await
                .map_err(|e| {
                    UnifiedProcessorError::QueueOperation(format!(
                        "tracked_files delete failed: {}",
                        e
                    ))
                })?;
            tx.commit().await.map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Transaction commit failed: {}",
                    e
                ))
            })?;
            Ok(())
        }
        .await;

        if let Err(e) = tx_result {
            warn!(
                "SQLite transaction failed after Qdrant delete for {}: {}. Marked for reconciliation on next startup.",
                relative_path, e
            );
            let _ = tracked_files_schema::mark_needs_reconcile(
                pool,
                existing.file_id,
                &format!("delete_tx_failed: {}", e),
            )
            .await;
        } else {
            // FTS5 cleanup: delete code_lines + file_metadata for this file (Task 52)
            if let Some(sdb) = &ctx.search_db {
                let processor =
                    FtsBatchProcessor::new(sdb, FtsBatchConfig::default());
                if let Err(e) = processor.delete_file(existing.file_id).await {
                    warn!(
                        "FTS5: failed to delete code_lines for file_id={}: {} (non-fatal)",
                        existing.file_id, e
                    );
                } else {
                    debug!(
                        "FTS5: deleted code_lines for file_id={}",
                        existing.file_id
                    );
                }
            }
            // Graph edge cleanup (graph-rag Task 3): non-blocking
            super::graph_ingest::delete_graph_edges(ctx, &item.tenant_id, relative_path)
                .await;

            // Keyword/tag cleanup: remove SQLite keyword/tag records
            let doc_id = crate::generate_document_id(&item.tenant_id, abs_file_path);
            super::keyword_persist::delete_extraction(pool, &doc_id).await;

            info!(
                "Deleted tracked file for: {} in {}ms (qdrant_delete={})",
                relative_path,
                delete_start.elapsed().as_millis(),
                delete_from_qdrant
            );
        }
        return Ok(());
    }

    // Fallback: file not in tracked_files, attempt Qdrant filter delete (tenant-scoped)
    warn!(
        "File not in tracked_files, falling back to Qdrant filter delete: {}",
        abs_file_path
    );
    ctx.storage_client
        .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    info!(
        "Deleted points for file (fallback) in {}ms: {}",
        delete_start.elapsed().as_millis(),
        abs_file_path
    );
    Ok(())
}

/// Clean up tracked records and Qdrant points for a file that no longer exists on disk.
pub(super) async fn cleanup_missing_file(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    payload: &FilePayload,
) {
    if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    {
        debug!(
            "File no longer exists, cleaning up tracked record and Qdrant points: {}",
            relative_path
        );

        // Get point IDs from qdrant_chunks before deletion
        let point_ids = tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
            .await
            .unwrap_or_default();

        // Delete Qdrant points first (irreversible), scoped to tenant
        if !point_ids.is_empty() {
            if let Err(e) = ctx
                .storage_client
                .delete_points_by_filter(
                    &item.collection,
                    &payload.file_path,
                    &item.tenant_id,
                )
                .await
            {
                // Qdrant deletion failed but may already be gone - log and continue cleanup
                warn!(
                    "Qdrant point deletion failed for missing file {}: {} (points may already be gone)",
                    relative_path, e
                );
            }
        }

        // Clean up SQLite records in a transaction (CASCADE handles qdrant_chunks)
        let tx_result: Result<(), UnifiedProcessorError> = async {
            let mut tx = pool.begin().await.map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;
            tracked_files_schema::delete_tracked_file_tx(&mut tx, existing.file_id)
                .await
                .map_err(|e| {
                    UnifiedProcessorError::QueueOperation(format!(
                        "tracked_files delete failed: {}",
                        e
                    ))
                })?;
            tx.commit().await.map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Transaction commit failed: {}",
                    e
                ))
            })?;
            Ok(())
        }
        .await;

        if let Err(e) = tx_result {
            warn!(
                "SQLite transaction failed during file-not-found cleanup for {}: {}. Marked for reconciliation on next startup.",
                relative_path, e
            );
            let _ = tracked_files_schema::mark_needs_reconcile(
                pool,
                existing.file_id,
                &format!("file_not_found_cleanup_tx_failed: {}", e),
            )
            .await;
        } else {
            info!(
                "Cleaned up {} Qdrant points and tracked record for missing file: {}",
                point_ids.len(),
                relative_path
            );
        }
    }
}

/// Handle Qdrant insert failure by cleaning up stale SQLite state.
pub(super) async fn handle_qdrant_failure(
    _ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    qdrant_err: &str,
) {
    // Old Qdrant points were deleted but new ones failed to insert.
    // Clean up stale qdrant_chunks so SQLite doesn't reference non-existent points.
    if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    {
        let cleanup_result: Result<(), String> = async {
            let mut tx = pool
                .begin()
                .await
                .map_err(|e| format!("begin tx: {}", e))?;
            tracked_files_schema::delete_qdrant_chunks_tx(&mut tx, existing.file_id)
                .await
                .map_err(|e| format!("delete chunks: {}", e))?;
            tx.commit()
                .await
                .map_err(|e| format!("commit: {}", e))?;
            Ok(())
        }
        .await;

        match cleanup_result {
            Ok(()) => {
                warn!(
                    "Qdrant insert failed for {}; cleaned up stale SQLite chunks. Error: {}",
                    relative_path, qdrant_err
                );
            }
            Err(cleanup_err) => {
                warn!(
                    "Qdrant insert failed AND chunk cleanup failed for {}: insert={}, cleanup={}",
                    relative_path, qdrant_err, cleanup_err
                );
                let _ = tracked_files_schema::mark_needs_reconcile(
                    pool,
                    existing.file_id,
                    &format!(
                        "qdrant_insert_failed_cleanup_failed: {}",
                        cleanup_err
                    ),
                )
                .await;
            }
        }
    }
}
