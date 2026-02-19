//! File processing strategy.
//!
//! Handles `ItemType::File` queue items: ingestion (add/update) and deletion,
//! including tracked_files management, Qdrant upsert, FTS5 indexing, LSP
//! enrichment, and keyword/tag extraction.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, error, info, warn};

use crate::context::ProcessingContext;
use crate::embedding::SparseEmbedding;
use crate::file_classification::{get_extension_for_storage, is_test_file};
use crate::fts_batch_processor::{FileChange, FtsBatchConfig, FtsBatchProcessor};
use crate::indexed_content_schema;
use crate::keyword_extraction::collection_config;
use crate::keyword_extraction::pipeline::PipelineInput;
use crate::lsp::{EnrichmentStatus, LspEnrichment};
use crate::search_db::SearchDbManager;
use crate::storage::DocumentPoint;
use crate::strategies::ProcessingStrategy;
use crate::tracked_files_schema::{self, ChunkType as TrackedChunkType, ProcessingStatus};
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    DestinationStatus, FilePayload, ItemType, QueueOperation, UnifiedQueueItem,
};
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS};
use wqm_common::hashing::compute_content_hash;

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
        let payload: FilePayload = serde_json::from_str(&item.payload_json).map_err(|e| {
            UnifiedProcessorError::InvalidPayload(format!("Failed to parse FilePayload: {}", e))
        })?;

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
        // This ensures tracked_files can be updated after Qdrant write, maintaining
        // tracked_files as authoritative inventory and preventing orphaned Qdrant data.
        //
        // For library-routed files from project folders (Task 568), the item's collection
        // is "libraries" but the watch_folder has collection="projects". Fall back to
        // looking up by "projects" when the primary lookup fails.
        let (watch_folder_id, base_path) = match watch_info {
            Some((wid, bp)) => (wid, bp),
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
                        (wid, bp)
                    }
                    None => {
                        error!(
                            "watch_folders validation failed: tenant_id={}, collection={} (also tried 'projects') -- refusing ingestion",
                            item.tenant_id, item.collection
                        );
                        return Err(UnifiedProcessorError::QueueOperation(format!(
                            "No watch_folder found for tenant_id={}, collection={} or projects. Cannot ingest without tracked_files context.",
                            item.tenant_id, item.collection
                        )));
                    }
                }
            }
            None => {
                error!(
                    "watch_folders validation failed: tenant_id={}, collection={} -- refusing ingestion to prevent orphaned data",
                    item.tenant_id, item.collection
                );
                return Err(UnifiedProcessorError::QueueOperation(format!(
                    "No watch_folder found for tenant_id={}, collection={}. Cannot ingest without tracked_files context.",
                    item.tenant_id, item.collection
                )));
            }
        };

        let relative_path =
            tracked_files_schema::compute_relative_path(&payload.file_path, &base_path)
                .unwrap_or_else(|| payload.file_path.clone());

        crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        // === DELETE OPERATION ===
        if item.op == QueueOperation::Delete {
            return Self::process_file_delete(
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
            Self::cleanup_missing_file(ctx, item, pool, &watch_folder_id, &relative_path, &payload)
                .await;
            return Err(UnifiedProcessorError::FileNotFound(
                payload.file_path.clone(),
            ));
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

                Self::execute_update_deletion(
                    ctx, item, pool, &watch_folder_id, &relative_path, &payload, &existing, &new_hash,
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

        Self::ingest_file_content(
            ctx, item, pool, file_path, &payload, &watch_folder_id, &relative_path,
        )
        .await
    }

    /// Clean up tracked records and Qdrant points for a file that no longer exists on disk.
    async fn cleanup_missing_file(
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

    /// Execute the deletion part of an update operation (reference-counted).
    ///
    /// Called after hash comparison determines the file has changed.
    /// Stores a `QueueDecision` for retry-safe execution and deletes old points
    /// only if no other watch folder references the same base_point.
    async fn execute_update_deletion(
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
                    Self::update_fts5_for_file(
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

        // Check if LSP enrichment is available for this project
        let (is_project_active, lsp_mgr_guard) = if let Some(lsp_mgr) = &ctx.lsp_manager {
            let mgr = lsp_mgr.read().await;
            let is_active = mgr.has_active_servers(&item.tenant_id).await;
            if is_active {
                debug!(
                    "LSP enrichment available for project {} on file {}",
                    item.tenant_id, payload.file_path
                );
            }
            (is_active, Some(lsp_mgr.clone()))
        } else {
            (false, None)
        };

        // Determine LSP/treesitter status for tracked_files
        let mut lsp_status = ProcessingStatus::None;
        let mut treesitter_status = ProcessingStatus::None;

        // Generate stable document_id for this file (deterministic from tenant + path)
        let file_document_id =
            crate::generate_document_id(&item.tenant_id, &payload.file_path);

        // Compute file hash and base_point BEFORE the chunk loop so point IDs use the base_point model
        let file_hash_early = tracked_files_schema::compute_file_hash(file_path)
            .unwrap_or_else(|_| "unknown".to_string());
        let base_point = wqm_common::hashing::compute_base_point(
            &item.tenant_id,
            &item.branch,
            relative_path,
            &file_hash_early,
        );

        // Process each chunk and build points + chunk metadata
        let mut points = Vec::new();
        let mut chunk_records: Vec<(
            String,
            i32,
            String,
            Option<TrackedChunkType>,
            Option<String>,
            Option<i32>,
            Option<i32>,
        )> = Vec::new();
        let embedding_start = std::time::Instant::now();

        for (chunk_idx, chunk) in document_content.chunks.iter().enumerate() {
            // Semaphore-gated embedding generation (Task 504)
            let _permit = ctx.embedding_semaphore.acquire().await.map_err(|e| {
                UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e))
            })?;
            let embedding_result = ctx
                .embedding_generator
                .generate_embedding(&chunk.content, "bge-small-en-v1.5")
                .await
                .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
            drop(_permit);

            let mut point_payload = HashMap::new();
            point_payload.insert("content".to_string(), serde_json::json!(chunk.content));
            point_payload.insert(
                "chunk_index".to_string(),
                serde_json::json!(chunk.chunk_index),
            );
            point_payload.insert(
                "file_path".to_string(),
                serde_json::json!(payload.file_path),
            );
            point_payload.insert(
                "document_id".to_string(),
                serde_json::json!(file_document_id),
            );
            point_payload.insert(
                "tenant_id".to_string(),
                serde_json::json!(item.tenant_id),
            );
            point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
            point_payload.insert("base_point".to_string(), serde_json::json!(base_point));
            point_payload.insert(
                "relative_path".to_string(),
                serde_json::json!(relative_path),
            );
            point_payload.insert(
                "absolute_path".to_string(),
                serde_json::json!(payload.file_path),
            );
            point_payload.insert(
                "file_hash".to_string(),
                serde_json::json!(file_hash_early),
            );
            point_payload.insert(
                "document_type".to_string(),
                serde_json::json!(document_content.document_type.as_str()),
            );
            if let Some(lang) = document_content.document_type.language() {
                point_payload.insert("language".to_string(), serde_json::json!(lang));
            }
            // Add file extension as metadata (e.g., "rs", "py", "md") -- lowercase for consistency
            if let Some(ext) = std::path::Path::new(&payload.file_path)
                .extension()
                .and_then(|e| e.to_str())
            {
                point_payload.insert(
                    "file_extension".to_string(),
                    serde_json::json!(ext.to_lowercase()),
                );
            }
            point_payload.insert("item_type".to_string(), serde_json::json!("file"));

            if let Some(file_type) = &payload.file_type {
                point_payload.insert(
                    "file_type".to_string(),
                    serde_json::json!(file_type.to_lowercase()),
                );
            }

            // Build tags array from static metadata for filtering/aggregation
            {
                let mut tags = Vec::new();
                if let Some(ft) = &payload.file_type {
                    tags.push(ft.to_lowercase());
                }
                if let Some(lang) = document_content.document_type.language() {
                    tags.push(lang.to_string());
                }
                if let Some(ext) = std::path::Path::new(&payload.file_path)
                    .extension()
                    .and_then(|e| e.to_str())
                {
                    tags.push(ext.to_lowercase());
                }
                if is_test_file(file_path) {
                    tags.push("test".to_string());
                }
                if !tags.is_empty() {
                    point_payload.insert("tags".to_string(), serde_json::json!(tags));
                }
            }

            // Extract chunk metadata for tracked record
            let symbol_name = chunk.metadata.get("symbol_name").cloned();
            let start_line = chunk
                .metadata
                .get("start_line")
                .and_then(|s| s.parse::<i32>().ok());
            let end_line = chunk
                .metadata
                .get("end_line")
                .and_then(|s| s.parse::<i32>().ok());
            let chunk_type_str = chunk.metadata.get("chunk_type");
            let chunk_type = chunk_type_str.and_then(|s| TrackedChunkType::from_str(s));

            // Detect tree-sitter status from chunk metadata
            if chunk.metadata.contains_key("chunk_type") {
                treesitter_status = ProcessingStatus::Done;
            }

            for (key, value) in &chunk.metadata {
                point_payload.insert(format!("chunk_{}", key), serde_json::json!(value));
            }

            // LSP enrichment (if available and file language has LSP support)
            if let Some(lsp_mgr) = &lsp_mgr_guard {
                let file_lang = file_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(crate::lsp::Language::from_extension);

                if file_lang
                    .as_ref()
                    .map_or(false, |l| l.has_lsp_support())
                {
                    let mgr = lsp_mgr.read().await;

                    let sym_name = chunk
                        .metadata
                        .get("symbol_name")
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    let sl = chunk
                        .metadata
                        .get("start_line")
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(chunk_idx as u32 * 20);

                    let el = chunk
                        .metadata
                        .get("end_line")
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(sl + 20);

                    let enrichment = mgr
                        .enrich_chunk(
                            &item.tenant_id,
                            file_path,
                            sym_name,
                            sl,
                            el,
                            is_project_active,
                        )
                        .await;

                    if enrichment.enrichment_status == EnrichmentStatus::Skipped {
                        // LSP server not ready -- mark as pending for metadata_uplift retry
                        point_payload.insert(
                            "lsp_enrichment_status".to_string(),
                            serde_json::json!("pending"),
                        );
                        // Keep lsp_status as None (not Done) so tracked_files reflects incomplete state
                    } else {
                        Self::add_lsp_enrichment_to_payload(
                            &mut point_payload,
                            &enrichment,
                        );
                        lsp_status = ProcessingStatus::Done;
                    }
                } else {
                    // Non-code file (markdown, config, etc.) -- skip LSP enrichment
                    point_payload.insert(
                        "lsp_enrichment_status".to_string(),
                        serde_json::json!("skipped"),
                    );
                    lsp_status = ProcessingStatus::Skipped;
                }
            }

            let point_id =
                wqm_common::hashing::compute_point_id(&base_point, chunk_idx as u32);
            let content_hash = tracked_files_schema::compute_content_hash(&chunk.content);

            // Use lexicon-backed BM25 for sparse vectors (Task 19: true BM25 with persisted IDF)
            let chunk_tokens: Vec<String> = chunk
                .content
                .split_whitespace()
                .map(|s| s.to_lowercase())
                .collect();
            let lexicon_sparse = ctx
                .lexicon_manager
                .generate_sparse_vector(&item.collection, &chunk_tokens)
                .await;
            // Fall back to embedding generator's sparse vector if lexicon has no corpus stats yet
            let sparse = if !lexicon_sparse.indices.is_empty() {
                lexicon_sparse
            } else {
                embedding_result.sparse.clone()
            };

            let point = DocumentPoint {
                id: point_id.clone(),
                dense_vector: embedding_result.dense.vector,
                sparse_vector: Self::sparse_embedding_to_map(&sparse),
                payload: point_payload,
            };

            points.push(point);
            chunk_records.push((
                point_id,
                chunk_idx as i32,
                content_hash,
                chunk_type,
                symbol_name,
                start_line,
                end_line,
            ));
        }

        info!(
            "Embedding generation completed: {} chunks in {}ms",
            chunk_records.len(),
            embedding_start.elapsed().as_millis()
        );

        // === KEYWORD/TAG EXTRACTION (Task 33) ===
        // Run extraction pipeline after chunk embeddings, before Qdrant upsert.
        // Results are injected into point payloads. Failures are non-fatal.
        if item.op == QueueOperation::Add || item.op == QueueOperation::Update {
            Self::run_keyword_extraction(ctx, item, file_path, &document_content, &mut points)
                .await;
        }

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
            Self::handle_qdrant_failure(
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
        let file_hash = file_hash_early;
        let file_mtime = tracked_files_schema::get_file_mtime(file_path)
            .unwrap_or_else(|_| wqm_common::timestamps::now_utc());
        let language = chunk_records
            .first()
            .and_then(|_| document_content.metadata.get("language"))
            .cloned();
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
                        &file_hash,
                        chunk_records.len() as i32,
                        chunking_method,
                        lsp_status,
                        treesitter_status,
                        Some(&base_point),
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
                        payload.file_type.as_deref(),
                        language.as_deref(),
                        &file_mtime,
                        &file_hash,
                        chunk_records.len() as i32,
                        chunking_method,
                        lsp_status,
                        treesitter_status,
                        Some(&item.collection),
                        extension.as_deref(),
                        is_test,
                        Some(&base_point),
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
            if !chunk_records.is_empty() {
                tracked_files_schema::insert_qdrant_chunks_tx(
                    &mut tx,
                    file_id,
                    &chunk_records,
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
        let file_id = tx_result?;

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
            Self::update_fts5_for_file(
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

    /// Run keyword/tag extraction pipeline and inject results into point payloads.
    async fn run_keyword_extraction(
        ctx: &ProcessingContext,
        item: &UnifiedQueueItem,
        file_path: &Path,
        document_content: &crate::DocumentContent,
        points: &mut [DocumentPoint],
    ) {
        let chunk_vectors: Vec<Vec<f32>> =
            points.iter().map(|p| p.dense_vector.clone()).collect();
        let chunk_texts: Vec<String> = document_content
            .chunks
            .iter()
            .map(|c| c.content.clone())
            .collect();
        let is_code = document_content.document_type.is_code();
        let language = document_content.document_type.language();

        // Fetch corpus size and build DF lookup from lexicon (Task 17)
        let corpus_size = ctx
            .lexicon_manager
            .corpus_size(&item.collection)
            .await;
        let full_text = chunk_texts.join("\n");

        // Build per-document DF lookup for unique terms in this document
        let unique_terms: std::collections::HashSet<String> = full_text
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| w.len() >= 2)
            .collect();
        let mut df_lookup = HashMap::new();
        for term in &unique_terms {
            let df = ctx
                .lexicon_manager
                .document_frequency(&item.collection, term)
                .await;
            if df > 0 {
                df_lookup.insert(term.clone(), df as u64);
            }
        }

        let pipeline_input = PipelineInput {
            file_path,
            full_text: &full_text,
            language,
            is_code,
            chunk_vectors: &chunk_vectors,
            chunk_texts: &chunk_texts,
            corpus_size,
            df_lookup: &df_lookup,
        };

        let extraction_start = std::time::Instant::now();
        let pipeline_config = collection_config::config_for_collection(&item.collection);
        let extraction = crate::keyword_extraction::pipeline::run_pipeline(
            &pipeline_input,
            &ctx.embedding_generator,
            &pipeline_config,
        )
        .await;

        let extraction_ms = extraction_start.elapsed().as_millis();
        info!(
            "Keyword/tag extraction completed in {}ms: {} keywords, {} tags, {} structural tags (corpus_size={})",
            extraction_ms,
            extraction.keywords.len(),
            extraction.tags.len(),
            extraction.structural_tags.len(),
            corpus_size,
        );

        // Update lexicon with this document's terms (Task 17)
        let tokens: Vec<String> = unique_terms.into_iter().collect();
        if let Err(e) = ctx
            .lexicon_manager
            .add_document(&item.collection, &tokens)
            .await
        {
            warn!(
                "Failed to update lexicon for {}: {}",
                item.collection, e
            );
        }

        // Inject extraction results into all point payloads
        let kw_phrases = extraction.keyword_phrases();
        let tag_phrases = extraction.tag_phrases();
        let struct_map = extraction.structural_tags_map();
        let basket_map = extraction.basket_map();

        for point in points.iter_mut() {
            if !kw_phrases.is_empty() {
                point
                    .payload
                    .insert("keywords".to_string(), serde_json::json!(kw_phrases));
            }
            if !tag_phrases.is_empty() {
                point.payload.insert(
                    "concept_tags".to_string(),
                    serde_json::json!(tag_phrases),
                );
            }
            if !struct_map.is_empty() {
                point.payload.insert(
                    "structural_tags".to_string(),
                    serde_json::json!(struct_map),
                );
            }
            if !basket_map.is_empty() {
                point.payload.insert(
                    "keyword_baskets".to_string(),
                    serde_json::json!(basket_map),
                );
            }
        }
    }

    /// Handle Qdrant insert failure by cleaning up stale SQLite state.
    async fn handle_qdrant_failure(
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

    /// Update the FTS5 code search index for a single file (Task 52).
    ///
    /// Reads the file from disk, compares content hash against indexed_content cache.
    /// If changed (or new), computes line diff and applies to code_lines + FTS5.
    /// Failures are logged but non-fatal -- they don't block Qdrant ingestion.
    #[allow(clippy::too_many_arguments)]
    async fn update_fts5_for_file(
        search_db: &Arc<SearchDbManager>,
        state_pool: &SqlitePool,
        file_id: i64,
        file_path: &str,
        tenant_id: &str,
        branch: Option<&str>,
        base_point: Option<&str>,
        relative_path: Option<&str>,
        file_hash: Option<&str>,
    ) {
        let fts_start = std::time::Instant::now();

        // Read file content from disk
        let new_content = match tokio::fs::read_to_string(file_path).await {
            Ok(content) => content,
            Err(e) => {
                debug!(
                    "FTS5: cannot read file for indexing (may be binary): {}: {}",
                    file_path, e
                );
                return;
            }
        };

        let new_hash = compute_content_hash(&new_content);

        // Check indexed_content cache for skip detection
        let old_content =
            match indexed_content_schema::get_indexed_content(state_pool, file_id).await {
                Ok(Some((cached_bytes, cached_hash))) => {
                    if cached_hash == new_hash {
                        debug!(
                            "FTS5: content unchanged (hash match), skipping: {}",
                            file_path
                        );
                        return;
                    }
                    // Content changed -- use cached content as diff base
                    String::from_utf8(cached_bytes).unwrap_or_default()
                }
                Ok(None) => {
                    // New file -- no old content to diff against
                    String::new()
                }
                Err(e) => {
                    warn!(
                        "FTS5: failed to read indexed_content cache for file_id={}: {}",
                        file_id, e
                    );
                    String::new()
                }
            };

        // Apply diff to code_lines via FtsBatchProcessor (single-file mode)
        let processor = FtsBatchProcessor::new(search_db, FtsBatchConfig::default());
        let change = FileChange {
            file_id,
            old_content,
            new_content: new_content.clone(),
            tenant_id: tenant_id.to_string(),
            branch: branch.map(|s| s.to_string()),
            file_path: file_path.to_string(),
            base_point: base_point.map(|s| s.to_string()),
            relative_path: relative_path.map(|s| s.to_string()),
            file_hash: file_hash.map(|s| s.to_string()),
        };

        // Use full_rewrite for new files (empty old content), diff for updates
        let fts_result = if change.old_content.is_empty() {
            processor
                .full_rewrite(
                    file_id,
                    &change.new_content,
                    tenant_id,
                    branch,
                    file_path,
                    base_point,
                    relative_path,
                    file_hash,
                )
                .await
        } else {
            // Use flush() with queue_depth=0 (single-file mode)
            let mut processor = processor;
            processor.add_change(change);
            processor.flush(0).await
        };

        match fts_result {
            Ok(stats) => {
                debug!(
                    "FTS5: updated {} (+{} ~{} -{}) for {} in {}ms",
                    file_path,
                    stats.lines_inserted,
                    stats.lines_updated,
                    stats.lines_deleted,
                    file_path,
                    fts_start.elapsed().as_millis()
                );

                // Update indexed_content cache with new content + hash
                if let Err(e) = indexed_content_schema::upsert_indexed_content(
                    state_pool,
                    file_id,
                    new_content.as_bytes(),
                    &new_hash,
                )
                .await
                {
                    warn!(
                        "FTS5: failed to update indexed_content cache for file_id={}: {}",
                        file_id, e
                    );
                }
            }
            Err(e) => {
                warn!(
                    "FTS5: failed to update code_lines for {}: {} (non-fatal)",
                    file_path, e
                );
            }
        }
    }

    /// Process file delete operation with tracked_files awareness (Task 506 + Task 519).
    async fn process_file_delete(
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

    /// Convert a `SparseEmbedding` to the `HashMap` format expected by `DocumentPoint`.
    fn sparse_embedding_to_map(
        sparse: &SparseEmbedding,
    ) -> Option<HashMap<u32, f32>> {
        crate::shared::embedding_pipeline::sparse_embedding_to_map(sparse)
    }

    /// Add LSP enrichment data to a point payload.
    pub(crate) fn add_lsp_enrichment_to_payload(
        payload: &mut HashMap<String, serde_json::Value>,
        enrichment: &LspEnrichment,
    ) {
        // Add enrichment status (lowercase for consistent metadata filtering)
        payload.insert(
            "lsp_enrichment_status".to_string(),
            serde_json::json!(enrichment.enrichment_status.as_str()),
        );

        // Skip adding empty data for non-success status
        if enrichment.enrichment_status == EnrichmentStatus::Skipped
            || enrichment.enrichment_status == EnrichmentStatus::Failed
        {
            if let Some(error) = &enrichment.error_message {
                payload.insert(
                    "lsp_enrichment_error".to_string(),
                    serde_json::json!(error),
                );
            }
            return;
        }

        // Add references (limited to avoid huge payloads)
        if !enrichment.references.is_empty() {
            let refs: Vec<_> = enrichment
                .references
                .iter()
                .take(20)
                .map(|r| {
                    serde_json::json!({
                        "file": r.file,
                        "line": r.line,
                        "column": r.column
                    })
                })
                .collect();
            payload.insert("lsp_references".to_string(), serde_json::json!(refs));
            payload.insert(
                "lsp_references_count".to_string(),
                serde_json::json!(enrichment.references.len()),
            );
        }

        // Add type info
        if let Some(type_info) = &enrichment.type_info {
            payload.insert(
                "lsp_type_signature".to_string(),
                serde_json::json!(type_info.type_signature),
            );
            payload.insert(
                "lsp_type_kind".to_string(),
                serde_json::json!(type_info.kind),
            );
            if let Some(doc) = &type_info.documentation {
                // Truncate long docs
                let truncated = if doc.len() > 500 {
                    format!("{}...", &doc[..500])
                } else {
                    doc.clone()
                };
                payload.insert(
                    "lsp_type_documentation".to_string(),
                    serde_json::json!(truncated),
                );
            }
        }

        // Add resolved imports
        if !enrichment.resolved_imports.is_empty() {
            let imports: Vec<_> = enrichment
                .resolved_imports
                .iter()
                .map(|imp| {
                    serde_json::json!({
                        "name": imp.import_name,
                        "target_file": imp.target_file,
                        "is_stdlib": imp.is_stdlib,
                        "resolved": imp.resolved
                    })
                })
                .collect();
            payload.insert("lsp_imports".to_string(), serde_json::json!(imports));
        }

        // Add definition location
        if let Some(def) = &enrichment.definition {
            payload.insert(
                "lsp_definition".to_string(),
                serde_json::json!({
                    "file": def.file,
                    "line": def.line,
                    "column": def.column
                }),
            );
        }
    }
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

    #[test]
    fn test_lsp_enrichment_status_lowercase_in_payload() {
        use crate::lsp::project_manager::{EnrichmentStatus, LspEnrichment};

        let mut payload = std::collections::HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        FileStrategy::add_lsp_enrichment_to_payload(&mut payload, &enrichment);
        let status = payload
            .get("lsp_enrichment_status")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(status, "success", "lsp_enrichment_status must be lowercase");

        let mut payload2 = std::collections::HashMap::new();
        let enrichment2 = LspEnrichment {
            enrichment_status: EnrichmentStatus::Failed,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: Some("test error".to_string()),
        };

        FileStrategy::add_lsp_enrichment_to_payload(&mut payload2, &enrichment2);
        let status2 = payload2
            .get("lsp_enrichment_status")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(
            status2, "failed",
            "lsp_enrichment_status must be lowercase"
        );
    }
}
