//! Qdrant upsert and tracked_files/qdrant_chunks SQLite transaction.
//!
//! After chunk embedding and keyword extraction, this module handles the
//! atomic persistence: batch upsert to Qdrant, then tracked_files +
//! qdrant_chunks insert/update within a single SQLite transaction.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::file_classification::{get_extension_for_storage, is_test_file};
use crate::storage::DocumentPoint;
use crate::tracked_files_schema::{self, ProcessingStatus};
use crate::unified_queue_processor::UnifiedProcessorError;
use crate::unified_queue_schema::{DestinationStatus, UnifiedQueueItem};
use crate::DocumentContent;

use super::chunk_embed::ChunkRecord;
use super::delete;

/// Upsert points to Qdrant and record in tracked_files + qdrant_chunks atomically.
///
/// Returns the `file_id` from `tracked_files` on success.
#[allow(clippy::too_many_arguments)]
pub(super) async fn upsert_and_track(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    points: Vec<DocumentPoint>,
    chunk_records: &[ChunkRecord],
    watch_folder_id: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_path: &Path,
    document_content: &DocumentContent,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    payload_file_type: Option<&str>,
) -> Result<i64, UnifiedProcessorError> {
    // Upsert points to Qdrant
    // Task 555: If insert fails after old points were deleted (update path),
    // clean up stale SQLite chunk records before propagating the error.
    let qdrant_insert_failed = if !points.is_empty() {
        info!(
            "Inserting {} points into {}",
            points.len(),
            item.collection
        );
        let upsert_start = std::time::Instant::now();
        match ctx
            .storage_client
            .insert_points_batch(&item.collection, points, Some(100))
            .await
        {
            Ok(_stats) => {
                info!(
                    "Qdrant upsert completed: {} points in {}ms",
                    chunk_records.len(),
                    upsert_start.elapsed().as_millis()
                );
                None
            }
            Err(e) => Some(e.to_string()),
        }
    } else {
        None
    };

    // If Qdrant insert failed, clean up stale SQLite state before propagating
    if let Some(ref qdrant_err) = qdrant_insert_failed {
        delete::handle_qdrant_failure(
            ctx,
            item,
            pool,
            watch_folder_id,
            relative_path,
            qdrant_err,
        )
        .await;
        let _ = ctx
            .queue_manager
            .update_destination_status(
                &item.queue_id,
                "qdrant",
                DestinationStatus::Failed,
            )
            .await;
        return Err(UnifiedProcessorError::Storage(qdrant_err.clone()));
    }

    // After Qdrant success: record in tracked_files + qdrant_chunks atomically (Task 519)
    let file_mtime = tracked_files_schema::get_file_mtime(file_path)
        .unwrap_or_else(|_| wqm_common::timestamps::now_utc());
    let language = document_content.metadata.get("language").cloned();
    let chunking_method = if treesitter_status == ProcessingStatus::Done {
        Some("tree_sitter")
    } else {
        Some("text")
    };
    let extension = get_extension_for_storage(file_path);
    let is_test = is_test_file(file_path);

    // Check if file is already tracked (read outside transaction)
    let existing = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!(
            "tracked_files lookup failed: {}",
            e
        ))
    })?;

    // Convert ChunkRecords to the tuple format expected by insert_qdrant_chunks_tx
    let chunk_tuples: Vec<(
        String,
        i32,
        String,
        Option<tracked_files_schema::ChunkType>,
        Option<String>,
        Option<i32>,
        Option<i32>,
    )> = chunk_records
        .iter()
        .map(|cr| {
            (
                cr.point_id.clone(),
                cr.chunk_index,
                cr.content_hash.clone(),
                cr.chunk_type,
                cr.symbol_name.clone(),
                cr.start_line,
                cr.end_line,
            )
        })
        .collect();

    // Begin SQLite transaction for atomic tracked_files + qdrant_chunks writes
    let tx_result: Result<i64, UnifiedProcessorError> = async {
        let mut tx = pool.begin().await.map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!(
                "Failed to begin transaction: {}",
                e
            ))
        })?;

        let file_id = match &existing {
            Some(existing_file) => {
                // Update existing record
                tracked_files_schema::update_tracked_file_tx(
                    &mut tx,
                    existing_file.file_id,
                    &file_mtime,
                    file_hash,
                    chunk_records.len() as i32,
                    chunking_method,
                    lsp_status,
                    treesitter_status,
                    Some(base_point),
                )
                .await
                .map_err(|e| {
                    UnifiedProcessorError::QueueOperation(format!(
                        "tracked_files update failed: {}",
                        e
                    ))
                })?;
                // Delete old chunks before inserting new
                tracked_files_schema::delete_qdrant_chunks_tx(
                    &mut tx,
                    existing_file.file_id,
                )
                .await
                .map_err(|e| {
                    UnifiedProcessorError::QueueOperation(format!(
                        "qdrant_chunks delete failed: {}",
                        e
                    ))
                })?;
                existing_file.file_id
            }
            None => {
                // Insert new record
                tracked_files_schema::insert_tracked_file_tx(
                    &mut tx,
                    watch_folder_id,
                    relative_path,
                    Some(item.branch.as_str()),
                    payload_file_type,
                    language.as_deref(),
                    &file_mtime,
                    file_hash,
                    chunk_records.len() as i32,
                    chunking_method,
                    lsp_status,
                    treesitter_status,
                    Some(&item.collection),
                    extension.as_deref(),
                    is_test,
                    Some(base_point),
                    Some(relative_path),
                )
                .await
                .map_err(|e| {
                    UnifiedProcessorError::QueueOperation(format!(
                        "tracked_files insert failed: {}",
                        e
                    ))
                })?
            }
        };

        // Insert qdrant_chunks
        if !chunk_tuples.is_empty() {
            tracked_files_schema::insert_qdrant_chunks_tx(
                &mut tx,
                file_id,
                &chunk_tuples,
            )
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "qdrant_chunks insert failed: {}",
                    e
                ))
            })?;
        }

        tx.commit().await.map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!(
                "Transaction commit failed: {}",
                e
            ))
        })?;

        debug!(
            "Recorded {} chunks in tracked_files for file_id={} ({})",
            chunk_records.len(),
            file_id,
            relative_path
        );
        Ok(file_id)
    }
    .await;

    // Handle transaction failure: Qdrant has points but SQLite state is inconsistent.
    if let Err(ref e) = tx_result {
        warn!(
            "SQLite transaction failed after Qdrant upsert for {}: {}. Queue item will be retried.",
            relative_path, e
        );
        if let Some(existing_file) = &existing {
            let _ = tracked_files_schema::mark_needs_reconcile(
                pool,
                existing_file.file_id,
                &format!("ingest_tx_failed: {}", e),
            )
            .await;
        }
    }

    tx_result
}
