//! File delete processing.
//!
//! Handles `QueueOperation::Delete` for file items: reference-counted Qdrant
//! point deletion, tracked_files cleanup, FTS5 cleanup, and missing-file
//! reconciliation.

use std::time::Instant;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};
use crate::processing_timings::{self, PhaseTiming};
use crate::tracked_files_schema;
use crate::tree_sitter::detect_language;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::UnifiedQueueItem;

/// Process file delete operation with tracked_files awareness (Task 506 + Task 519).
///
/// **F-035 contract:** Qdrant delete failures block SQLite cleanup and return
/// `Err`. The tracked_files row stays intact and the queue row enters retry
/// via `mark_unified_failed`. Without this, stale Qdrant vectors stayed
/// retrievable after the local row was wiped, with no record that cleanup
/// had failed.
pub(super) async fn process_file_delete(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
) -> UnifiedProcessorResult<()> {
    let delete_start = Instant::now();
    let mut timings: Vec<PhaseTiming> = Vec::new();
    let detected_language = detect_language(std::path::Path::new(abs_file_path));

    if let Ok(true) = tracked_files_schema::is_incremental(pool, abs_file_path).await {
        info!(
            "Skipping delete for incremental file (safety net): {}",
            abs_file_path
        );
        return Ok(());
    }

    if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    {
        timings.push(PhaseTiming {
            phase: "lookup",
            duration_ms: delete_start.elapsed().as_millis() as u64,
        });

        let delete_result = delete_tracked_file(
            ctx,
            item,
            pool,
            watch_folder_id,
            relative_path,
            abs_file_path,
            &existing,
            &mut timings,
            delete_start,
        )
        .await;

        record_delete_timings(ctx, item, pool, detected_language, &timings).await;
        return delete_result;
    }

    // Fallback: file not in tracked_files — attempt Qdrant filter delete.
    // F-035: a fallback Qdrant delete failure is still a real failure — the
    // points may exist but cannot be deleted. Surface it so retry metadata is
    // populated.
    let fallback_result =
        fallback_qdrant_delete(ctx, item, abs_file_path, delete_start, &mut timings).await;
    record_delete_timings(ctx, item, pool, detected_language, &timings).await;
    fallback_result
}

/// Delete a file that is present in tracked_files: Qdrant (ref-counted), SQLite,
/// FTS5, graph edges, and keyword extractions.
///
/// **F-035:** if the Qdrant delete fails, `tracked_files` row is preserved and
/// `Err` is returned so the queue row gets retry metadata. Without this guard,
/// stale vectors stayed in Qdrant while local tracking was already wiped.
#[allow(clippy::too_many_arguments)]
async fn delete_tracked_file(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
    existing: &tracked_files_schema::TrackedFile,
    timings: &mut Vec<PhaseTiming>,
    delete_start: Instant,
) -> UnifiedProcessorResult<()> {
    let delete_from_qdrant =
        check_qdrant_deletion_needed(ctx, watch_folder_id, relative_path, existing).await;

    let t0 = Instant::now();
    if delete_from_qdrant {
        if let Err(e) =
            delete_qdrant_points(ctx, item, pool, relative_path, abs_file_path, existing).await
        {
            timings.push(PhaseTiming {
                phase: "qdrant_delete",
                duration_ms: t0.elapsed().as_millis() as u64,
            });
            warn!(
                "Qdrant delete failed for {} — leaving tracked_files row intact and queuing retry: {}",
                relative_path, e
            );
            return Err(e);
        }
    }
    timings.push(PhaseTiming {
        phase: "qdrant_delete",
        duration_ms: t0.elapsed().as_millis() as u64,
    });

    let t0 = Instant::now();
    let sqlite_ok = delete_tracked_file_sqlite(pool, relative_path, existing, timings, t0).await;

    if sqlite_ok {
        let t0 = Instant::now();
        cleanup_fts5(ctx, existing).await;
        timings.push(PhaseTiming {
            phase: "fts5_cleanup",
            duration_ms: t0.elapsed().as_millis() as u64,
        });

        let t0 = Instant::now();
        super::graph_ingest::delete_graph_edges(ctx, &item.tenant_id, relative_path).await;
        timings.push(PhaseTiming {
            phase: "graph_cleanup",
            duration_ms: t0.elapsed().as_millis() as u64,
        });

        let t0 = Instant::now();
        let doc_id = crate::generate_document_id(&item.tenant_id, abs_file_path);
        super::keyword_persist::delete_extraction(pool, &doc_id).await;
        timings.push(PhaseTiming {
            phase: "keyword_cleanup",
            duration_ms: t0.elapsed().as_millis() as u64,
        });

        info!(
            "Deleted tracked file for: {} in {}ms (qdrant_delete={})",
            relative_path,
            delete_start.elapsed().as_millis(),
            delete_from_qdrant
        );
    }
    Ok(())
}

/// Determine whether Qdrant points should be deleted (reference-count check).
async fn check_qdrant_deletion_needed(
    ctx: &ProcessingContext,
    watch_folder_id: &str,
    relative_path: &str,
    existing: &tracked_files_schema::TrackedFile,
) -> bool {
    if let Some(ref bp) = existing.base_point {
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
        true
    }
}

/// Delete Qdrant points for a tracked file.
///
/// **F-035:** propagate Qdrant errors so the caller can short-circuit SQLite
/// cleanup. Returning `Ok(())` on failure silently dropped the local row
/// while leaving the vectors retrievable in Qdrant.
async fn delete_qdrant_points(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    relative_path: &str,
    abs_file_path: &str,
    existing: &tracked_files_schema::TrackedFile,
) -> UnifiedProcessorResult<()> {
    let point_ids = tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
        .await
        .unwrap_or_default();

    if point_ids.is_empty() {
        return Ok(());
    }

    ctx.storage_client
        .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
        .await
        .map_err(|e| {
            UnifiedProcessorError::Storage(format!(
                "Qdrant delete failed for {} ({} tracked points): {}",
                relative_path,
                point_ids.len(),
                e
            ))
        })?;
    Ok(())
}

/// Delete the tracked_files row in a transaction. Returns `true` on success.
async fn delete_tracked_file_sqlite(
    pool: &SqlitePool,
    relative_path: &str,
    existing: &tracked_files_schema::TrackedFile,
    timings: &mut Vec<PhaseTiming>,
    t0: Instant,
) -> bool {
    let tx_result: Result<(), UnifiedProcessorError> = async {
        let mut tx = pool.begin().await.map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!("Failed to begin transaction: {}", e))
        })?;
        tracked_files_schema::delete_tracked_file_tx(&mut tx, existing.file_id)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!("tracked_files delete failed: {}", e))
            })?;
        tx.commit().await.map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!("Transaction commit failed: {}", e))
        })?;
        Ok(())
    }
    .await;

    timings.push(PhaseTiming {
        phase: "sqlite_cleanup",
        duration_ms: t0.elapsed().as_millis() as u64,
    });

    match tx_result {
        Ok(()) => true,
        Err(e) => {
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
            false
        }
    }
}

/// Clean up FTS5 code_lines for a deleted file (non-fatal).
async fn cleanup_fts5(ctx: &ProcessingContext, existing: &tracked_files_schema::TrackedFile) {
    if let Some(sdb) = &ctx.search_db {
        let processor = FtsBatchProcessor::new(sdb, FtsBatchConfig::default());
        if let Err(e) = processor.delete_file(existing.file_id).await {
            warn!(
                "FTS5: failed to delete code_lines for file_id={}: {} (non-fatal)",
                existing.file_id, e
            );
        } else {
            debug!("FTS5: deleted code_lines for file_id={}", existing.file_id);
        }
    }
}

/// Fallback delete when the file is not in tracked_files: attempt Qdrant filter delete.
///
/// **F-035:** Qdrant errors are real failures (network/auth/server fault),
/// not "points might not exist" — the Qdrant client returns Ok with zero
/// affected points if nothing matches the filter. Propagate Err so the
/// caller queues a retry instead of silently swallowing the failure.
async fn fallback_qdrant_delete(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    abs_file_path: &str,
    delete_start: Instant,
    timings: &mut Vec<PhaseTiming>,
) -> UnifiedProcessorResult<()> {
    debug!(
        "File not in tracked_files, attempting Qdrant filter delete: {}",
        abs_file_path
    );
    let t0 = Instant::now();
    let result = ctx
        .storage_client
        .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
        .await;
    timings.push(PhaseTiming {
        phase: "qdrant_delete",
        duration_ms: t0.elapsed().as_millis() as u64,
    });
    match result {
        Ok(_) => {
            info!(
                "Deleted points for file (fallback) in {}ms: {}",
                delete_start.elapsed().as_millis(),
                abs_file_path
            );
            Ok(())
        }
        Err(e) => Err(UnifiedProcessorError::Storage(format!(
            "Qdrant fallback delete failed for {}: {}",
            abs_file_path, e
        ))),
    }
}

/// Record timing data for delete operations.
async fn record_delete_timings(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    language: Option<&str>,
    timings: &[PhaseTiming],
) {
    let _ = ctx; // ProcessingContext not needed for recording, but kept for consistency
    processing_timings::record_timings(
        pool,
        &item.queue_id,
        item.item_type.as_str(),
        item.op.as_str(),
        &item.tenant_id,
        &item.collection,
        language,
        // Deletes remove points — no file_type classification or embedding.
        None,
        None,
        timings,
    )
    .await;
}

/// Clean up tracked records and Qdrant points for a file that no longer exists on disk.
///
/// **F-035:** if Qdrant deletion fails, returns `Err` and leaves the
/// tracked_files row intact. Without this, missing-file cleanup wiped the
/// local row while Qdrant still held vectors that could surface in search.
pub(super) async fn cleanup_missing_file(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
) -> UnifiedProcessorResult<()> {
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

        // Delete Qdrant points first (irreversible), scoped to tenant.
        // F-035: surface Qdrant errors so the queue row can retry with metadata.
        if !point_ids.is_empty() {
            ctx.storage_client
                .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
                .await
                .map_err(|e| {
                    UnifiedProcessorError::Storage(format!(
                        "Qdrant delete for missing file {} failed: {}",
                        relative_path, e
                    ))
                })?;
        }

        // Clean up SQLite records in a transaction (CASCADE handles qdrant_chunks)
        let tx_result: Result<(), UnifiedProcessorError> = async {
            let mut tx = pool.begin().await.map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!("Failed to begin transaction: {}", e))
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
                UnifiedProcessorError::QueueOperation(format!("Transaction commit failed: {}", e))
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
    Ok(())
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
            let mut tx = pool.begin().await.map_err(|e| format!("begin tx: {}", e))?;
            tracked_files_schema::delete_qdrant_chunks_tx(&mut tx, existing.file_id)
                .await
                .map_err(|e| format!("delete chunks: {}", e))?;
            tx.commit().await.map_err(|e| format!("commit: {}", e))?;
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
                    &format!("qdrant_insert_failed_cleanup_failed: {}", cleanup_err),
                )
                .await;
            }
        }
    }
}
