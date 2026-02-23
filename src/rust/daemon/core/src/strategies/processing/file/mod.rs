//! File processing strategy.
//!
//! Handles `ItemType::File` queue items: ingestion (add/update) and deletion,
//! including tracked_files management, Qdrant upsert, FTS5 indexing, LSP
//! enrichment, and keyword/tag extraction.
//!
//! Split into focused submodules:
//! - `chunk_embed` — per-chunk embedding, payload construction, LSP enrichment
//! - `delete` — delete operation, missing-file cleanup, Qdrant failure handling
//! - `fts5_index` — FTS5 code search index updates
//! - `keyword_extract` — keyword/tag extraction pipeline
//! - `lsp_payload` — LSP enrichment payload serialization
//! - `store_track` — Qdrant upsert + tracked_files/qdrant_chunks transaction
//! - `update_preamble` — hash comparison + reference-counted old point deletion

mod chunk_embed;
mod delete;
mod fts5_index;
mod graph_ingest;
mod keyword_extract;
pub(crate) mod lsp_payload;
mod store_track;
mod update_preamble;

use std::path::Path;

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, error, info};

use crate::context::ProcessingContext;
use crate::strategies::ProcessingStrategy;
use crate::tracked_files_schema;
use crate::specs::parse_payload;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    DestinationStatus, FilePayload, ItemType, QueueOperation, UnifiedQueueItem,
};
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS};

/// Strategy for processing file queue items.
///
/// Routes to file ingestion (add/update) or file deletion based on the
/// queue item operation.
pub struct FileStrategy;

#[async_trait]
impl ProcessingStrategy for FileStrategy {
    fn handles(&self, item_type: &ItemType, _op: &QueueOperation) -> bool {
        *item_type == ItemType::File
    }

    async fn process(
        &self,
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> Result<(), UnifiedProcessorError> {
        Self::process_file_item(ctx, item).await
    }

    fn name(&self) -> &'static str {
        "file"
    }
}

impl FileStrategy {
    /// Main file processing entry point.
    ///
    /// Parses the file payload, validates the watch folder, then dispatches
    /// to delete or ingest/update paths as appropriate.
    pub(crate) async fn process_file_item(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing file item: {} -> collection: {} (op={:?})",
            item.queue_id, item.collection, item.op
        );

        // Parse the file payload
        let payload: FilePayload = parse_payload(item)?;

        // File type allowlist check (Task 511) - skip for delete operations
        if item.op != QueueOperation::Delete
            && !ctx
                .allowed_extensions
                .is_allowed(&payload.file_path, &item.collection)
        {
            debug!(
                "File type not in allowlist, skipping: {} (collection={})",
                payload.file_path, item.collection
            );
            return Ok(());
        }

        let file_path = Path::new(&payload.file_path);
        let pool = ctx.queue_manager.pool();

        // Look up watch_folder for tracked_files context
        let (watch_folder_id, base_path) =
            resolve_watch_folder(pool, item).await?;

        let relative_path =
            tracked_files_schema::compute_relative_path(&payload.file_path, &base_path)
                .unwrap_or_else(|| payload.file_path.clone());

        crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        // === DELETE OPERATION ===
        if item.op == QueueOperation::Delete {
            return delete::process_file_delete(
                ctx,
                item,
                pool,
                &watch_folder_id,
                &relative_path,
                &payload.file_path,
            )
            .await;
        }

        // For ingest/update: check if file exists on disk
        if !file_path.exists() {
            delete::cleanup_missing_file(
                ctx, item, pool, &watch_folder_id, &relative_path, &payload,
            )
            .await;
            // File is gone — cleanup already handled above. Treat as a no-op
            // success so the queue item is deleted rather than stuck in 'failed'.
            info!(
                "File no longer exists, cleaned up and dequeuing: {}",
                payload.file_path
            );
            return Ok(());
        }

        // === UPDATE OPERATION: hash comparison + reference-counted deletion ===
        // This block is inlined (not in a helper) because the `return Ok(())`
        // on hash-match must exit process_file_item entirely.
        if item.op == QueueOperation::Update {
            let new_hash = tracked_files_schema::compute_file_hash(file_path).map_err(|e| {
                UnifiedProcessorError::ProcessingFailed(format!("Failed to hash file: {}", e))
            })?;

            if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
                pool,
                &watch_folder_id,
                &relative_path,
                Some(item.branch.as_str()),
            )
            .await
            {
                if existing.file_hash == new_hash {
                    info!(
                        "File unchanged (hash match), skipping update: {}",
                        relative_path
                    );
                    return Ok(());
                }

                update_preamble::execute_update_deletion(
                    ctx, item, pool, &watch_folder_id, &relative_path, &payload, &existing,
                    &new_hash,
                )
                .await?;
            } else {
                // Not tracked yet -- defensive cleanup: delete by filter as fallback for update
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

        ingest_file_content(
            ctx, item, pool, file_path, &payload, &watch_folder_id, &relative_path,
        )
        .await
    }
}

/// Resolve the watch folder for a file item (with library fallback).
async fn resolve_watch_folder(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
) -> Result<(String, String), UnifiedProcessorError> {
    let watch_info = tracked_files_schema::lookup_watch_folder(
        pool,
        &item.tenant_id,
        &item.collection,
    )
    .await
    .map_err(|e| {
        UnifiedProcessorError::QueueOperation(format!(
            "Failed to lookup watch_folder: {}",
            e
        ))
    })?;

    // CRITICAL: watch_folders lookup MUST succeed before ingestion.
    // For library-routed files from project folders (Task 568), the item's collection
    // is "libraries" but the watch_folder has collection="projects". Fall back to
    // looking up by "projects" when the primary lookup fails.
    match watch_info {
        Some((wid, bp)) => Ok((wid, bp)),
        None if item.collection == COLLECTION_LIBRARIES => {
            // Try fallback: file may originate from a project watch folder
            let fallback = tracked_files_schema::lookup_watch_folder(
                pool,
                &item.tenant_id,
                COLLECTION_PROJECTS,
            )
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Fallback watch_folder lookup failed: {}",
                    e
                ))
            })?;

            match fallback {
                Some((wid, bp)) => {
                    debug!(
                        "Library-routed file resolved via project watch_folder: tenant={}, watch_id={}",
                        item.tenant_id, wid
                    );
                    Ok((wid, bp))
                }
                None => {
                    error!(
                        "watch_folders validation failed: tenant_id={}, collection={} (also tried 'projects') -- refusing ingestion",
                        item.tenant_id, item.collection
                    );
                    Err(UnifiedProcessorError::QueueOperation(format!(
                        "No watch_folder found for tenant_id={}, collection={} or projects. Cannot ingest without tracked_files context.",
                        item.tenant_id, item.collection
                    )))
                }
            }
        }
        None => {
            error!(
                "watch_folders validation failed: tenant_id={}, collection={} -- refusing ingestion to prevent orphaned data",
                item.tenant_id, item.collection
            );
            Err(UnifiedProcessorError::QueueOperation(format!(
                "No watch_folder found for tenant_id={}, collection={}. Cannot ingest without tracked_files context.",
                item.tenant_id, item.collection
            )))
        }
    }
}

/// Ingest file content: embedding, Qdrant upsert, tracked_files, FTS5.
///
/// Shared by both add and update paths (after update preamble completes).
#[allow(clippy::too_many_arguments)]
async fn ingest_file_content(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    payload: &FilePayload,
    watch_folder_id: &str,
    relative_path: &str,
) -> UnifiedProcessorResult<()> {
    // === PER-DESTINATION RETRY SKIP (Task 6) ===
    // If qdrant_status is already 'done' from a previous attempt, skip directly
    // to the search DB (FTS5) update.
    let qdrant_already_done = item.qdrant_status == Some(DestinationStatus::Done);
    if qdrant_already_done {
        info!(
            "Qdrant already done for {} (retry), skipping to search DB update",
            item.queue_id
        );

        // We still need file_id for FTS5 -- look it up from tracked_files
        if let Some(sdb) = &ctx.search_db {
            if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
                pool,
                watch_folder_id,
                relative_path,
                Some(item.branch.as_str()),
            )
            .await
            {
                let _ = ctx
                    .queue_manager
                    .update_destination_status(
                        &item.queue_id,
                        "search",
                        DestinationStatus::InProgress,
                    )
                    .await;
                fts5_index::update_fts5_for_file(
                    sdb,
                    pool,
                    existing.file_id,
                    &payload.file_path,
                    &item.tenant_id,
                    Some(&item.branch),
                    existing.base_point.as_deref(),
                    Some(relative_path),
                    Some(existing.file_hash.as_str()),
                )
                .await;
                let _ = ctx
                    .queue_manager
                    .update_destination_status(
                        &item.queue_id,
                        "search",
                        DestinationStatus::Done,
                    )
                    .await;
            } else {
                let _ = ctx
                    .queue_manager
                    .update_destination_status(
                        &item.queue_id,
                        "search",
                        DestinationStatus::Done,
                    )
                    .await;
            }
        } else {
            let _ = ctx
                .queue_manager
                .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
                .await;
        }

        return Ok(());
    }

    // Mark qdrant status as in_progress
    let _ = ctx
        .queue_manager
        .update_destination_status(
            &item.queue_id,
            "qdrant",
            DestinationStatus::InProgress,
        )
        .await;

    // === INGEST / UPDATE: process file content ===
    let document_content = ctx
        .document_processor
        .process_file_content(file_path, &item.collection)
        .await
        .map_err(|e| UnifiedProcessorError::ProcessingFailed(e.to_string()))?;

    info!(
        "Extracted {} chunks from {}",
        document_content.chunks.len(),
        payload.file_path
    );

    // Generate stable document_id for this file (deterministic from tenant + path)
    let file_document_id =
        crate::generate_document_id(&item.tenant_id, &payload.file_path);

    // Compute file hash and base_point BEFORE the chunk loop so point IDs use the base_point model
    let file_hash = tracked_files_schema::compute_file_hash(file_path)
        .unwrap_or_else(|_| "unknown".to_string());
    let base_point = wqm_common::hashing::compute_base_point(
        &item.tenant_id,
        &item.branch,
        relative_path,
        &file_hash,
    );

    // Embed all chunks
    let embed_result = chunk_embed::embed_chunks(
        ctx,
        item,
        &document_content,
        file_path,
        &file_document_id,
        relative_path,
        &base_point,
        &file_hash,
        payload.file_type.as_deref(),
    )
    .await?;

    let mut points = embed_result.points;
    let chunk_records = embed_result.chunk_records;
    let lsp_status = embed_result.lsp_status;
    let treesitter_status = embed_result.treesitter_status;

    // === KEYWORD/TAG EXTRACTION (Task 33) ===
    // Run extraction pipeline after chunk embeddings, before Qdrant upsert.
    // Results are injected into point payloads. Failures are non-fatal.
    if item.op == QueueOperation::Add || item.op == QueueOperation::Update {
        keyword_extract::run_keyword_extraction(
            ctx,
            item,
            file_path,
            &document_content,
            &mut points,
        )
        .await;
    }

    // === GRAPH RELATIONSHIP EXTRACTION (graph-rag Task 3) ===
    // Extract code relationships (CALLS, CONTAINS, IMPORTS, USES_TYPE) from
    // semantic chunk metadata and store in graph.db. Non-blocking: failures
    // are logged but never fail the ingestion pipeline.
    graph_ingest::ingest_graph_edges(
        ctx,
        &item.tenant_id,
        relative_path,
        &document_content.chunks,
    )
    .await;

    // Upsert to Qdrant + record in tracked_files atomically
    let file_id = store_track::upsert_and_track(
        ctx,
        item,
        pool,
        points,
        &chunk_records,
        watch_folder_id,
        relative_path,
        &base_point,
        &file_hash,
        file_path,
        &document_content,
        lsp_status,
        treesitter_status,
        payload.file_type.as_deref(),
    )
    .await?;

    // Mark qdrant destination as done (Task 6: per-destination state machine)
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::Done)
        .await;

    // === FTS5 CODE SEARCH INDEX UPDATE (Task 52) ===
    let _ = ctx
        .queue_manager
        .update_destination_status(
            &item.queue_id,
            "search",
            DestinationStatus::InProgress,
        )
        .await;
    if let Some(sdb) = &ctx.search_db {
        fts5_index::update_fts5_for_file(
            sdb,
            pool,
            file_id,
            &payload.file_path,
            &item.tenant_id,
            Some(&item.branch),
            Some(&base_point),
            Some(relative_path),
            Some(&file_hash),
        )
        .await;
    }
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
        .await;

    info!(
        "Successfully processed file item {} ({})",
        item.queue_id, payload.file_path
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_strategy_handles_file_items() {
        let strategy = FileStrategy;
        assert!(strategy.handles(&ItemType::File, &QueueOperation::Add));
        assert!(strategy.handles(&ItemType::File, &QueueOperation::Update));
        assert!(strategy.handles(&ItemType::File, &QueueOperation::Delete));
    }

    #[test]
    fn test_file_strategy_rejects_non_file_items() {
        let strategy = FileStrategy;
        assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Folder, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Tenant, &QueueOperation::Add));
        assert!(!strategy.handles(&ItemType::Url, &QueueOperation::Add));
    }

    #[test]
    fn test_file_strategy_name() {
        let strategy = FileStrategy;
        assert_eq!(strategy.name(), "file");
    }
}
