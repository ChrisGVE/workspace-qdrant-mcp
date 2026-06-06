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
//! - `zero_byte` — graceful handling of empty (0-byte) files

mod branch_dedup;
mod chunk_embed;
mod component;
mod delete;
mod fts5_index;
mod grammar;
mod graph_ingest;
mod ingest;
mod keyword_extract;
mod keyword_persist;
pub(crate) mod lsp_payload;
mod store_track;
mod update_preamble;
mod zero_byte;

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
use wqm_common::paths::CanonicalPath;

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

        let payload: FilePayload = parse_payload(item)?;

        if !passes_ingestion_guards(ctx, item, &payload) {
            return Ok(());
        }

        let pool = ctx.queue_manager.pool();
        let (watch_folder_id, base_path) =
            resolve_watch_folder(pool, item, payload.file_path.as_str()).await?;

        // Reconstruct the absolute filesystem path by anchoring the
        // relative payload path to the watch_folder root. The relative
        // form (already validated by serde + the type system) is what we
        // pass downstream as `relative_path`.
        let base_canonical = CanonicalPath::from_user_input(&base_path).map_err(|e| {
            UnifiedProcessorError::InvalidPayload(format!(
                "watch_folder.path is not canonical for tenant_id={}: {}",
                item.tenant_id, e
            ))
        })?;
        let abs_canonical = payload.file_path.to_absolute(&base_canonical);
        let abs_file_path: String = abs_canonical.as_str().to_string();
        let file_path = Path::new(abs_file_path.as_str());
        let relative_path: &str = payload.file_path.as_str();

        crate::shared::ensure_collection(&ctx.storage_client, &item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        if item.op == QueueOperation::Delete {
            return delete::process_file_delete(
                ctx,
                item,
                pool,
                &watch_folder_id,
                relative_path,
                &abs_file_path,
            )
            .await;
        }

        if !file_path.exists() {
            // F-035: handle_missing_file now returns Err if Qdrant delete failed;
            // propagate so the queue row picks up retry metadata.
            handle_missing_file(
                ctx,
                item,
                pool,
                &watch_folder_id,
                relative_path,
                &abs_file_path,
                &payload,
            )
            .await?;
            return Ok(());
        }

        if zero_byte::is_zero_byte(file_path) {
            return zero_byte::handle_zero_byte_file(
                ctx,
                item,
                pool,
                file_path,
                &payload,
                &watch_folder_id,
                relative_path,
            )
            .await;
        }

        if item.op == QueueOperation::Update || item.op == QueueOperation::Add {
            // For Update this is the hash compare + reference-counted deletion
            // of the prior generation. For Add it closes a GC gap: Qdrant point
            // IDs are deterministic per content hash, so re-Adding an
            // already-tracked file whose content changed produces NEW point IDs
            // that do NOT overwrite the old generation — leaking the old points
            // and growing the `projects` collection without bound on re-scans.
            // Reuse the same safe, reference-counted deletion. Only do the
            // defensive filter-delete for *untracked* files on Update: for an
            // untracked Add (a genuinely new file, or a post-reembed re-ingest
            // against a freshly-recreated collection) it would be a no-op round
            // trip at best and a cross-branch over-delete risk at worst.
            let defensive_delete_untracked = item.op == QueueOperation::Update;
            if prepare_update(
                ctx,
                item,
                pool,
                file_path,
                &watch_folder_id,
                relative_path,
                &abs_file_path,
                &payload,
                defensive_delete_untracked,
            )
            .await?
                == UpdateAction::Skip
            {
                return Ok(());
            }
        }

        if item.op == QueueOperation::Uplift {
            prepare_uplift(
                ctx,
                item,
                pool,
                file_path,
                &watch_folder_id,
                relative_path,
                &abs_file_path,
                &payload,
            )
            .await?;
        }

        ingest::ingest_file_content(
            ctx,
            item,
            pool,
            file_path,
            &payload,
            &abs_file_path,
            &watch_folder_id,
            &base_path,
            relative_path,
        )
        .await
    }
}

/// Return value for `prepare_update`: indicates whether to skip or proceed with ingest.
#[derive(PartialEq)]
enum UpdateAction {
    Skip,
    Proceed,
}

/// Check allowlist and per-extension size limit. Returns `false` if the file
/// should be silently skipped (non-error).
fn passes_ingestion_guards(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &FilePayload,
) -> bool {
    if item.op == QueueOperation::Delete {
        return true;
    }

    let rel = payload.file_path.as_str();
    if !ctx.allowed_extensions.is_allowed(rel, &item.collection) {
        debug!(
            "File type not in allowlist, skipping: {} (collection={})",
            rel, item.collection
        );
        return false;
    }

    if let Some(size) = payload.size_bytes {
        let ext = crate::file_classification::get_extension_for_storage(Path::new(rel))
            .unwrap_or_default();
        if let Some(limit) = ctx.ingestion_limits.size_limit_bytes(&ext) {
            if size > limit {
                warn!(
                    extension = %ext,
                    size_kb = size / 1024,
                    limit_kb = limit / 1024,
                    path = %rel,
                    "Skipping oversized file: exceeds per-extension limit"
                );
                return false;
            }
        }
    }

    true
}

/// Handle a queue item whose file no longer exists on disk: clean up tracked
/// records and mark both destinations done so the item is dequeued cleanly.
///
/// **F-035:** if Qdrant cleanup fails, returns `Err` without marking
/// destinations done — the queue row stays for retry.
#[allow(clippy::too_many_arguments)]
async fn handle_missing_file(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &sqlx::SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
    payload: &FilePayload,
) -> crate::unified_queue_processor::UnifiedProcessorResult<()> {
    delete::cleanup_missing_file(
        ctx,
        item,
        pool,
        watch_folder_id,
        relative_path,
        abs_file_path,
    )
    .await?;
    let _ = ctx
        .queue_manager
        .update_destination_status(
            &item.queue_id,
            "qdrant",
            crate::unified_queue_schema::DestinationStatus::Done,
        )
        .await;
    let _ = ctx
        .queue_manager
        .update_destination_status(
            &item.queue_id,
            "search",
            crate::unified_queue_schema::DestinationStatus::Done,
        )
        .await;
    debug!(
        "File no longer exists, cleaned up and dequeuing: {}",
        payload.file_path.as_str()
    );
    Ok(())
}

/// Handle the Update pre-flight: hash comparison and reference-counted deletion.
///
/// Returns `UpdateAction::Skip` when the file is unchanged (hash match).
#[allow(clippy::too_many_arguments)]
async fn prepare_update(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &sqlx::SqlitePool,
    file_path: &Path,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
    payload: &FilePayload,
    // When the file is NOT tracked, also issue a defensive delete-by-filter to
    // sweep any orphaned points. Safe for Update; skipped for Add to avoid a
    // per-new-file round trip and a cross-branch over-delete on shared paths.
    defensive_delete_untracked: bool,
) -> UnifiedProcessorResult<UpdateAction> {
    let new_hash = tracked_files_schema::compute_file_hash(file_path).map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!("Failed to hash file: {}", e))
    })?;

    if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    {
        if existing.file_hash == new_hash {
            info!(
                "File unchanged (hash match), skipping update: {}",
                relative_path
            );
            return Ok(UpdateAction::Skip);
        }
        update_preamble::execute_update_deletion(
            ctx,
            item,
            pool,
            watch_folder_id,
            relative_path,
            abs_file_path,
            payload,
            &existing,
            &new_hash,
        )
        .await?;
    } else if defensive_delete_untracked {
        // Not tracked yet — defensive cleanup via filter (filter matches
        // the absolute path stored in the Qdrant payload's `file_path`
        // field; the queue payload itself is relative).
        ctx.storage_client
            .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    }
    Ok(UpdateAction::Proceed)
}

/// Handle the Uplift pre-flight: delete old points so fresh enrichment is produced.
#[allow(clippy::too_many_arguments)]
async fn prepare_uplift(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &sqlx::SqlitePool,
    file_path: &Path,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
    payload: &FilePayload,
) -> UnifiedProcessorResult<()> {
    let new_hash = tracked_files_schema::compute_file_hash(file_path).map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!("Failed to hash file: {}", e))
    })?;

    if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
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
            watch_folder_id,
            relative_path,
            abs_file_path,
            payload,
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
    Ok(())
}

/// Resolve the watch folder for a file item (with library fallback).
async fn resolve_watch_folder(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
    relative_path: &str,
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
        Some((wid, bp)) => {
            // Multi-clone disambiguation: `lookup_watch_folder` keys on
            // tenant_id alone, so for a tenant with sibling working copies it
            // can return a different (or stale) clone's path. If THIS file
            // isn't at the resolved root but another watch_folder of the same
            // tenant has it, use that one — otherwise the file resolves to a
            // base where it doesn't exist, hits `handle_missing_file`, and is
            // silently skipped (never indexed). See the multi-clone bugs in
            // 545c4cd09 (registration) and the last_scan fix.
            if let Some(better) = disambiguate_multi_path(pool, item, relative_path, &bp).await {
                return Ok(better);
            }
            Ok((wid, bp))
        }
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

/// For a tenant with more than one watch_folder (sibling working copies of the
/// same git remote), pick the watch_folder whose root actually contains
/// `relative_path` on disk.
///
/// Returns `None` — keep the caller's original resolution — when either the
/// already-resolved `current_base` holds the file (the common single-clone
/// case, settled in one stat), or no sibling has it (genuinely missing /
/// delete op). Only when the resolved base misses AND a sibling hits do we
/// re-point, which is exactly the stale/wrong-clone case.
async fn disambiguate_multi_path(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
    relative_path: &str,
    current_base: &str,
) -> Option<(String, String)> {
    // Fast path: the resolved base already holds the file — nothing to fix.
    if Path::new(current_base).join(relative_path).exists() {
        return None;
    }
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT watch_id, path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2",
    )
    .bind(&item.tenant_id)
    .bind(&item.collection)
    .fetch_all(pool)
    .await
    .ok()?;
    if rows.len() <= 1 {
        return None;
    }
    rows.into_iter()
        .find(|(_, bp)| Path::new(bp).join(relative_path).exists())
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
