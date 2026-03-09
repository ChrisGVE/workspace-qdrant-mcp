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
mod ingest;
mod keyword_extract;
mod keyword_persist;
pub(crate) mod lsp_payload;
mod store_track;
mod update_preamble;

use std::path::Path;

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, error, info, warn};

use crate::context::ProcessingContext;
use crate::specs::parse_payload;
use crate::strategies::ProcessingStrategy;
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation, UnifiedQueueItem};
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

        // Per-extension size limit check (Task 14) - skip for delete operations
        if item.op != QueueOperation::Delete {
            if let Some(size) = payload.size_bytes {
                let ext = crate::file_classification::get_extension_for_storage(
                    std::path::Path::new(&payload.file_path),
                )
                .unwrap_or_default();
                if let Some(limit) = ctx.ingestion_limits.size_limit_bytes(&ext) {
                    if size > limit {
                        warn!(
                            extension = %ext,
                            size_kb = size / 1024,
                            limit_kb = limit / 1024,
                            path = %payload.file_path,
                            "Skipping oversized file: exceeds per-extension limit"
                        );
                        return Ok(());
                    }
                }
            }
        }

        let file_path = Path::new(&payload.file_path);
        let pool = ctx.queue_manager.pool();

        // Look up watch_folder for tracked_files context
        let (watch_folder_id, base_path) = resolve_watch_folder(pool, item).await?;

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
                ctx,
                item,
                pool,
                &watch_folder_id,
                &relative_path,
                &payload,
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
                    ctx,
                    item,
                    pool,
                    &watch_folder_id,
                    &relative_path,
                    &payload,
                    &existing,
                    &new_hash,
                )
                .await?;
            } else {
                // Not tracked yet -- defensive cleanup: delete by filter as fallback for update
                ctx.storage_client
                    .delete_points_by_filter(&item.collection, &payload.file_path, &item.tenant_id)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            }
        }

        // === UPLIFT OPERATION: capability upgrade re-processing ===
        // Bypasses hash comparison — the file content hasn't changed but
        // capabilities have improved (grammar now available, LSP now ready,
        // or a previous enrichment failure should be retried). Delete old
        // points so the full re-ingest produces fresh chunks/enrichment.
        if item.op == QueueOperation::Uplift {
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
                info!(
                    "Uplift: re-processing file for capability upgrade: {}",
                    relative_path
                );
                update_preamble::execute_update_deletion(
                    ctx,
                    item,
                    pool,
                    &watch_folder_id,
                    &relative_path,
                    &payload,
                    &existing,
                    &new_hash,
                )
                .await?;
            } else {
                debug!(
                    "Uplift: file not previously tracked, treating as fresh ingest: {}",
                    relative_path
                );
            }
        }

        ingest::ingest_file_content(
            ctx,
            item,
            pool,
            file_path,
            &payload,
            &watch_folder_id,
            &base_path,
            &relative_path,
        )
        .await
    }
}

/// Resolve the watch folder for a file item (with library fallback).
async fn resolve_watch_folder(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
) -> Result<(String, String), UnifiedProcessorError> {
    let watch_info =
        tracked_files_schema::lookup_watch_folder(pool, &item.tenant_id, &item.collection)
            .await
            .map_err(|e| {
                UnifiedProcessorError::QueueOperation(format!(
                    "Failed to lookup watch_folder: {}",
                    e
                ))
            })?;

    // CRITICAL: watch_folders lookup MUST succeed before ingestion.
    // For library-routed files from project folders, the item's tenant_id is a
    // derived library name (e.g. "abc123-refs") and collection is "libraries",
    // but the watch_folder has the original project tenant_id and collection="projects".
    // Fall back using source_project_id from metadata when the primary lookup fails.
    match watch_info {
        Some((wid, bp)) => Ok((wid, bp)),
        None if item.collection == COLLECTION_LIBRARIES => {
            // Extract source_project_id from metadata for format-routed files
            let source_project_id = item
                .metadata
                .as_deref()
                .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
                .and_then(|v| v.get("source_project_id")?.as_str().map(String::from));

            // Try fallback with source_project_id (original project tenant)
            let fallback_tenant = source_project_id.as_deref().unwrap_or(&item.tenant_id);
            let fallback = tracked_files_schema::lookup_watch_folder(
                pool,
                fallback_tenant,
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
                        "Library-routed file resolved via project watch_folder: library_tenant={}, source_project={}, watch_id={}",
                        item.tenant_id, fallback_tenant, wid
                    );
                    Ok((wid, bp))
                }
                None => {
                    error!(
                        "watch_folders validation failed: tenant_id={}, source_project={:?}, collection={} -- refusing ingestion",
                        item.tenant_id, source_project_id, item.collection
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
