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
use crate::tracked_files_schema::{self, ProcessingStatus, TrackedFile};
use crate::unified_queue_processor::UnifiedProcessorError;
use crate::unified_queue_schema::{DestinationStatus, UnifiedQueueItem};
use crate::DocumentContent;

use super::chunk_embed::ChunkRecord;
use super::delete;

/// Chunk tuple format expected by `insert_qdrant_chunks_tx`.
type ChunkTuple = (
    String,
    i32,
    String,
    Option<tracked_files_schema::ChunkType>,
    Option<String>,
    Option<i32>,
    Option<i32>,
);

/// Upsert points to Qdrant and record in tracked_files + qdrant_chunks atomically.
///
/// `detected_branch` is the branch resolved from the `BranchCache` at
/// processing time; it may differ from `item.branch` after a branch switch.
/// Used for `tracked_files.primary_branch` and the lookup key.
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
    component: Option<String>,
    detected_branch: &str,
) -> Result<i64, UnifiedProcessorError> {
    // Look the file up BEFORE upserting so a changed-content re-ingest can GC
    // its previous generation of Qdrant points first.
    let existing = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(detected_branch),
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("tracked_files lookup failed: {}", e))
    })?;

    // GC superseded Qdrant points on the Add path. Point IDs embed the content
    // hash (base_point = SHA256(tenant|path|file_hash)), so re-ingesting changed
    // content produces brand-new IDs and the old generation is orphaned — the
    // Update path deletes the old points in its preamble, but the Add path
    // (folder rescans, reembed, startup recovery) historically did not. The
    // orphans surfaced as phantom/duplicate search hits (#92) and unbounded
    // Qdrant growth across reembeds (#96). Mirror the Update path's
    // reference-counted delete-by-filter (see update_preamble::execute_update_deletion).
    gc_superseded_points(
        ctx,
        item,
        existing.as_ref(),
        watch_folder_id,
        base_point,
        file_path,
    )
    .await?;

    upsert_to_qdrant(
        ctx,
        item,
        pool,
        points,
        chunk_records,
        watch_folder_id,
        relative_path,
    )
    .await?;

    let chunk_tuples = build_chunk_tuples(chunk_records);

    let tx_result = run_tracking_transaction(
        pool,
        item,
        &existing,
        &chunk_tuples,
        chunk_records.len(),
        watch_folder_id,
        relative_path,
        base_point,
        file_hash,
        file_path,
        document_content,
        lsp_status,
        treesitter_status,
        payload_file_type,
        component.as_deref(),
        detected_branch,
    )
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

/// Delete a tracked file's previous Qdrant generation before an Add re-ingest.
///
/// No-op when the file is new (`existing` is `None`) or its content is
/// unchanged (`base_point` matches) — in the unchanged case the new point IDs
/// equal the old ones, so the upsert overwrites in place and there is nothing
/// to GC. When the content changed, the old generation's IDs differ and would
/// be orphaned, so they are deleted by `(file_path, tenant_id)` filter —
/// reference-counted exactly like the Update path: if another watch folder
/// still references the old `base_point` (content-hash dedup across clones),
/// the points are shared and must NOT be deleted.
async fn gc_superseded_points(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    existing: Option<&TrackedFile>,
    watch_folder_id: &str,
    new_base_point: &str,
    file_path: &Path,
) -> Result<(), UnifiedProcessorError> {
    let Some(existing_file) = existing else {
        return Ok(());
    };
    let old_base_point = existing_file.base_point.as_deref();
    if old_base_point == Some(new_base_point) {
        return Ok(());
    }

    let delete_old = match old_base_point {
        Some(old_bp) => !ctx
            .queue_manager
            .has_other_references(old_bp, watch_folder_id)
            .await
            .unwrap_or(false),
        None => true,
    };
    if !delete_old {
        info!(
            "Add-path GC: old base_point still referenced by another watch folder, keeping superseded points for: {}",
            file_path.display()
        );
        return Ok(());
    }

    let abs_file_path = file_path.to_string_lossy();
    ctx.storage_client
        .delete_points_by_filter(&item.collection, &abs_file_path, &item.tenant_id)
        .await
        .map_err(|e| {
            UnifiedProcessorError::Storage(format!(
                "Add-path GC delete failed for {}: {}",
                abs_file_path, e
            ))
        })?;
    debug!(
        "Add-path GC: deleted superseded Qdrant points for changed file: {}",
        file_path.display()
    );
    Ok(())
}

/// Upsert document points to Qdrant. On failure, cleans up stale SQLite state
/// and returns an error.
async fn upsert_to_qdrant(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    points: Vec<DocumentPoint>,
    chunk_records: &[ChunkRecord],
    watch_folder_id: &str,
    relative_path: &str,
) -> Result<(), UnifiedProcessorError> {
    if points.is_empty() {
        return Ok(());
    }

    info!("Inserting {} points into {}", points.len(), item.collection);
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
            Ok(())
        }
        Err(e) => {
            // Task 555: clean up stale SQLite chunk records before propagating the error
            let qdrant_err = e.to_string();
            delete::handle_qdrant_failure(
                ctx,
                item,
                pool,
                watch_folder_id,
                relative_path,
                &qdrant_err,
            )
            .await;
            let _ = ctx
                .queue_manager
                .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::Failed)
                .await;
            Err(UnifiedProcessorError::Storage(qdrant_err))
        }
    }
}

/// Convert `ChunkRecord` slice to the tuple format expected by `insert_qdrant_chunks_tx`.
fn build_chunk_tuples(chunk_records: &[ChunkRecord]) -> Vec<ChunkTuple> {
    chunk_records
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
        .collect()
}

struct FileTrackMeta<'a> {
    file_mtime: String,
    language: Option<String>,
    chunking_method: Option<&'a str>,
    extension: Option<String>,
    is_test: bool,
}

fn build_file_track_meta<'a>(
    file_path: &Path,
    document_content: &DocumentContent,
    treesitter_status: ProcessingStatus,
) -> FileTrackMeta<'a> {
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
    FileTrackMeta {
        file_mtime,
        language,
        chunking_method,
        extension,
        is_test,
    }
}

/// Execute the SQLite transaction that records tracked_files + qdrant_chunks.
///
/// Returns the `file_id` assigned to this file.
#[allow(clippy::too_many_arguments)]
async fn run_tracking_transaction(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
    existing: &Option<TrackedFile>,
    chunk_tuples: &[ChunkTuple],
    chunk_count: usize,
    watch_folder_id: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_path: &Path,
    document_content: &DocumentContent,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    payload_file_type: Option<&str>,
    component: Option<&str>,
    detected_branch: &str,
) -> Result<i64, UnifiedProcessorError> {
    let meta = build_file_track_meta(file_path, document_content, treesitter_status);

    let mut tx = pool.begin().await.map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("Failed to begin transaction: {}", e))
    })?;

    let file_id = match existing {
        Some(existing_file) => {
            upsert_existing_tracked_file(
                &mut tx,
                existing_file,
                &meta.file_mtime,
                file_hash,
                chunk_count,
                meta.chunking_method,
                lsp_status,
                treesitter_status,
                base_point,
                component,
            )
            .await?
        }
        None => {
            insert_new_tracked_file(
                &mut tx,
                item,
                watch_folder_id,
                relative_path,
                payload_file_type,
                meta.language.as_deref(),
                &meta.file_mtime,
                file_hash,
                chunk_count,
                meta.chunking_method,
                lsp_status,
                treesitter_status,
                meta.extension.as_deref(),
                meta.is_test,
                base_point,
                component,
                detected_branch,
            )
            .await?
        }
    };

    if !chunk_tuples.is_empty() {
        tracked_files_schema::insert_qdrant_chunks_tx(&mut tx, file_id, chunk_tuples)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!("qdrant_chunks insert failed: {}", e))
            })?;
    }

    tx.commit().await.map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("Transaction commit failed: {}", e))
    })?;

    debug!(
        "Recorded {} chunks in tracked_files for file_id={} ({})",
        chunk_count, file_id, relative_path
    );
    Ok(file_id)
}

/// Update an existing `tracked_files` row and delete its old chunks within `tx`.
#[allow(clippy::too_many_arguments)]
async fn upsert_existing_tracked_file(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    existing_file: &TrackedFile,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: usize,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    base_point: &str,
    component: Option<&str>,
) -> Result<i64, UnifiedProcessorError> {
    tracked_files_schema::update_tracked_file_tx(
        tx,
        existing_file.file_id,
        file_mtime,
        file_hash,
        chunk_count as i32,
        chunking_method,
        lsp_status,
        treesitter_status,
        Some(base_point),
        component,
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("tracked_files update failed: {}", e))
    })?;

    tracked_files_schema::delete_qdrant_chunks_tx(tx, existing_file.file_id)
        .await
        .map_err(|e| {
            UnifiedProcessorError::QueueOperation(format!("qdrant_chunks delete failed: {}", e))
        })?;

    Ok(existing_file.file_id)
}

/// Insert a new `tracked_files` row within `tx`.
#[allow(clippy::too_many_arguments)]
async fn insert_new_tracked_file(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    item: &UnifiedQueueItem,
    watch_folder_id: &str,
    relative_path: &str,
    payload_file_type: Option<&str>,
    language: Option<&str>,
    file_mtime: &str,
    file_hash: &str,
    chunk_count: usize,
    chunking_method: Option<&str>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    extension: Option<&str>,
    is_test: bool,
    base_point: &str,
    component: Option<&str>,
    detected_branch: &str,
) -> Result<i64, UnifiedProcessorError> {
    tracked_files_schema::insert_tracked_file_tx(
        tx,
        watch_folder_id,
        relative_path,
        Some(detected_branch),
        payload_file_type,
        language,
        file_mtime,
        file_hash,
        chunk_count as i32,
        chunking_method,
        lsp_status,
        treesitter_status,
        Some(&item.collection),
        extension,
        is_test,
        Some(base_point),
        component,
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!("tracked_files insert failed: {}", e))
    })
}
