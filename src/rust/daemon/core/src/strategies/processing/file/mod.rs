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
use crate::patterns::global_ignore;
use crate::patterns::ignore_gate::IgnoreGate;
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

        // Dequeue-time ignore gate: ignore rules may have changed between
        // enqueue and processing (a global.wqmignore edit, a new project
        // .wqmignore). Without this re-check, a stale Add/Update burns the
        // full parse+embed cost on a now-excluded path and the result has to
        // be reconciled away afterwards. Deletes are exempt — they are how
        // excluded files leave the index.
        if item.op != QueueOperation::Delete && is_ignored_at_dequeue(&base_path, file_path) {
            info!(
                "Skipping now-ignored file (dequeue gate): {}",
                relative_path
            );
            // Mark both destinations done so the row finalizes cleanly
            // (mirrors handle_missing_file; see PR #86 on stranded statuses).
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
            return Ok(());
        }

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
                &base_path,
                relative_path,
                &abs_file_path,
                &payload,
                defensive_delete_untracked,
            )
            .await?
                == UpdateAction::Skip
            {
                return resolve_skip_destinations(
                    ctx,
                    item,
                    pool,
                    &watch_folder_id,
                    relative_path,
                    &abs_file_path,
                    &payload,
                )
                .await;
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

/// Dequeue-time re-check of the full ignore decision (project cascade +
/// `global.wqmignore`) for a queued file path.
///
/// Builds the same [`IgnoreGate`] the scan paths use, anchored at the watch
/// folder root, and replays ancestor directory rules — the queued path is
/// reached directly, not via a pruning walk. Building the gate costs a few
/// file reads; the embedding work it can save costs seconds to minutes.
fn is_ignored_at_dequeue(base_path: &str, file_path: &Path) -> bool {
    let root = Path::new(base_path);
    let gate = IgnoreGate::for_dir(
        file_path.parent().unwrap_or(root),
        Some(root),
        global_ignore::resolve_global_ignore_path().as_deref(),
    );
    gate.is_ignored_with_ancestors(root, file_path)
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

/// Resolve both destination sinks for an unchanged-hash Skip so the queue
/// row can finalize.
///
/// A bare `return Ok(())` on Skip strands state-machine items (decision_json
/// set): with no sink resolution, `finalize_after_success` keeps the row
/// `in_progress` and it re-leases forever while logging "Successfully
/// processed" — the recurring-poison shape from PR #86, this time at runtime
/// (observed live: an Update whose first attempt committed Qdrant +
/// tracked_files, then failed the FTS5 batch with `database is locked`; every
/// retry hash-matched, skipped, and never finalized, while search.db kept
/// serving the previous file generation).
///
/// The hash match proves the Qdrant generation is committed (tracked_files is
/// written in the same transaction as the upsert, see store_track), so the
/// qdrant sink resolves on that proof. The search sink gets NO such proof —
/// the FTS5 batch is asynchronous and may have failed after tracked_files was
/// already updated — so delegate to the retry-skip path, which re-dispatches
/// the FTS5 work: a cheap no-op when search is current (indexed_content hash
/// match), a healing rewrite when it is stale.
async fn resolve_skip_destinations(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &sqlx::SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    abs_file_path: &str,
    payload: &FilePayload,
) -> UnifiedProcessorResult<()> {
    let _ = ctx
        .queue_manager
        .update_destination_status(
            &item.queue_id,
            "qdrant",
            crate::unified_queue_schema::DestinationStatus::Done,
        )
        .await;
    ingest::handle_retry_skip(
        ctx,
        item,
        pool,
        watch_folder_id,
        relative_path,
        abs_file_path,
        payload,
    )
    .await
}

/// Handle the Update pre-flight: hash comparison and reference-counted deletion.
///
/// Returns `UpdateAction::Skip` when the file is unchanged — same content
/// hash AND same chunking-configuration fingerprint. A hash match with a
/// STALE fingerprint (the language's registry `semantic_patterns` or the
/// chunker logic changed since the row was written) falls through to the
/// normal update path so extractor upgrades reach already-indexed files;
/// without this, a registry fix never propagated to unchanged files (the
/// `.proto` semantic_patterns rollout shipped and every existing `.proto`
/// stayed on its old text chunks through a full re-embed).
#[allow(clippy::too_many_arguments)]
async fn prepare_update(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &sqlx::SqlitePool,
    file_path: &Path,
    watch_folder_id: &str,
    base_path: &str,
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
            // Same detection inputs as run_ingest_pipeline, so the value
            // compares against what store_track wrote (gitattributes
            // overrides are cached per base_path — this is cheap).
            let overrides = component::get_gitattributes(ctx, base_path).await;
            let detected = crate::tree_sitter::detect_language_with_overrides(
                file_path,
                relative_path,
                &overrides,
            );
            let current_fp = crate::tree_sitter::chunker::chunking_fingerprint(detected);
            if crate::tree_sitter::chunker::stored_fingerprint_is_current(
                existing.chunker_version.as_deref(),
                &current_fp,
            ) {
                info!(
                    "File unchanged (hash + chunker fingerprint match), skipping update: {}",
                    relative_path
                );
                return Ok(UpdateAction::Skip);
            }
            info!(
                "Chunking configuration changed for {} (stored {:?} → current {}), re-chunking unchanged file",
                relative_path, existing.chunker_version, current_fp
            );
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

    #[test]
    fn dequeue_gate_skips_path_excluded_by_project_wqmignore() {
        // Simulates the race the gate closes: the item was enqueued first,
        // the exclusion landed afterwards. At dequeue time the gate must see
        // the CURRENT rules and skip — including dir-only rules that only
        // match an ancestor directory of the queued file.
        let proj = tempfile::tempdir().unwrap();
        std::fs::write(
            proj.path().join(".wqmignore"),
            "doc-backend/proto/src/generated/\n",
        )
        .unwrap();
        let base = proj.path().to_string_lossy().to_string();
        let ignored = proj
            .path()
            .join("doc-backend/proto/src/generated/doc/OnCall.java");
        let kept = proj.path().join("doc-backend/src/main/java/App.java");
        assert!(is_ignored_at_dequeue(&base, &ignored));
        assert!(!is_ignored_at_dequeue(&base, &kept));
    }

    #[test]
    fn dequeue_gate_keeps_everything_without_ignore_files() {
        let proj = tempfile::tempdir().unwrap();
        let base = proj.path().to_string_lossy().to_string();
        let f = proj.path().join("src/lib.rs");
        assert!(!is_ignored_at_dequeue(&base, &f));
    }

    mod skip_resolution {
        use std::sync::Arc;

        use async_trait::async_trait;
        use sqlx::SqlitePool;
        use tokio::sync::Semaphore;

        use crate::allowed_extensions::AllowedExtensions;
        use crate::context::ProcessingContext;
        use crate::document_processor::DocumentProcessor;
        use crate::embedding::{
            DenseEmbedding, DenseProvider, EmbeddingConfig, EmbeddingError, EmbeddingGenerator,
        };
        use crate::lexicon::LexiconManager;
        use crate::queue_operations::QueueManager;
        use crate::specs::parse_payload;
        use crate::storage::StorageClient;
        use crate::unified_queue_schema::{FilePayload, ItemType, QueueOperation, QueueStatus};

        use super::super::resolve_skip_destinations;

        /// No-op dense provider: `resolve_skip_destinations` never embeds, but
        /// `ProcessingContext::new` requires an `EmbeddingGenerator`. Twin of
        /// the stub in `strategies::processing::tenant::tests`.
        #[derive(Debug)]
        struct NoopDenseProvider;

        #[async_trait]
        impl DenseProvider for NoopDenseProvider {
            async fn embed(&self, texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
                Ok(texts
                    .iter()
                    .map(|t| DenseEmbedding {
                        vector: vec![0.0; 1],
                        model_name: "noop".to_string(),
                        sequence_length: t.len(),
                    })
                    .collect())
            }

            fn output_dim(&self) -> usize {
                1
            }

            fn provider_label(&self) -> &str {
                "noop"
            }

            fn metrics_label(&self) -> &'static str {
                "fastembed"
            }

            async fn probe(&self) -> Result<(), EmbeddingError> {
                Ok(())
            }
        }

        /// Minimal context: lazy StorageClient (never contacted), no LSP, and
        /// crucially `search_db: None` so the search sink resolves to Done
        /// without touching FTS5 — the queue-state contract is what's under test.
        fn build_test_context(pool: SqlitePool) -> ProcessingContext {
            let queue_manager = Arc::new(QueueManager::new(pool.clone()));
            let dense_provider = Arc::new(NoopDenseProvider);
            let embedding_generator = Arc::new(
                EmbeddingGenerator::new(EmbeddingConfig::default(), dense_provider)
                    .expect("EmbeddingGenerator::new should succeed with NoopDenseProvider"),
            );
            ProcessingContext::new(
                pool.clone(),
                queue_manager,
                Arc::new(StorageClient::new()),
                embedding_generator,
                Arc::new(DocumentProcessor::new()),
                Arc::new(Semaphore::new(1)),
                Arc::new(LexiconManager::new(pool, 1.2)),
                None,
                None,
                Arc::new(AllowedExtensions::default()),
            )
        }

        /// Recurring-poison regression (PR #86 family, runtime variant): an
        /// Update that stored a decision (state-machine item) and then hits the
        /// unchanged-hash Skip on retry MUST resolve both destination sinks so
        /// `finalize_after_success` returns Done — not loop `in_progress`
        /// forever re-leasing every cycle.
        #[tokio::test]
        async fn unchanged_hash_skip_resolves_sinks_so_item_finalizes() {
            let temp_dir = tempfile::tempdir().unwrap();
            let db_path = temp_dir.path().join("skip_finalize.db");
            let pool = SqlitePool::connect(&format!("sqlite://{}?mode=rwc", db_path.display()))
                .await
                .expect("create sqlite pool");
            let manager = QueueManager::new(pool.clone());
            manager.init_unified_queue().await.unwrap();
            // enqueue_unified validates tenants against watch_folders.
            sqlx::query(crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
                .execute(&pool)
                .await
                .unwrap();

            // Shape the row exactly like production: enqueue, then store the
            // update decision (flips sinks to pending = state-machine item).
            let (queue_id, _) = manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Update,
                    "tenant-skip",
                    "projects",
                    r#"{"file_path":"src/lib.rs","file_type":"code"}"#,
                    Some("main"),
                    None,
                )
                .await
                .unwrap();
            manager
                .store_queue_decision(
                    &queue_id,
                    &wqm_common::queue_types::QueueDecision {
                        delete_old: true,
                        old_base_point: Some("old-bp".to_string()),
                        new_base_point: "new-bp".to_string(),
                        old_file_hash: Some("old-hash".to_string()),
                        new_file_hash: "new-hash".to_string(),
                    },
                )
                .await
                .unwrap();

            // Lease it like the processor would, then run the Skip resolution.
            let items = manager
                .dequeue_unified(1, "test-worker", None, None, None, None, None, None)
                .await
                .unwrap();
            let item = items
                .into_iter()
                .find(|i| i.queue_id == queue_id)
                .expect("dequeued the seeded item");
            let payload: FilePayload = parse_payload(&item).unwrap();

            let ctx = build_test_context(pool.clone());
            resolve_skip_destinations(
                &ctx,
                &item,
                &pool,
                "wf-test",
                "src/lib.rs",
                "/abs/src/lib.rs",
                &payload,
            )
            .await
            .unwrap();

            let (qs, ss): (String, String) = sqlx::query_as(
                "SELECT qdrant_status, search_status FROM unified_queue WHERE queue_id = ?1",
            )
            .bind(&queue_id)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert_eq!(qs, "done", "qdrant sink must resolve on the hash proof");
            assert_eq!(
                ss, "done",
                "search sink must resolve (no search_db configured)"
            );

            let overall = manager.finalize_after_success(&queue_id).await.unwrap();
            assert_eq!(
                overall,
                QueueStatus::Done,
                "row must finalize, not re-lease"
            );
        }

        /// Extractor-upgrade gate: with the content hash unchanged, the skip
        /// decision now also depends on the chunker fingerprint —
        ///  - stale stored fingerprint → Proceed (re-chunk the file);
        ///  - current stored fingerprint → Skip;
        ///  - NULL (legacy row) → Skip (grandfathered).
        /// This is the propagation path the `.proto` semantic_patterns rollout
        /// was missing: a registry upgrade must reach unchanged files on the
        /// next visit instead of being filtered by the hash match.
        #[tokio::test]
        async fn unchanged_hash_gate_honors_chunker_fingerprint() {
            use super::super::{prepare_update, UpdateAction};

            // Real file on disk so compute_file_hash matches the seeded row.
            let project = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(project.path().join("src")).unwrap();
            let file_abs = project.path().join("src/lib.rs");
            std::fs::write(&file_abs, "pub fn answer() -> u32 { 42 }\n").unwrap();
            let real_hash = crate::tracked_files_schema::compute_file_hash(&file_abs).unwrap();
            let base_path = project.path().to_string_lossy().to_string();
            let abs_str = file_abs.to_string_lossy().to_string();

            let db_dir = tempfile::tempdir().unwrap();
            let db_path = db_dir.path().join("gate.db");
            let pool = SqlitePool::connect(&format!("sqlite://{}?mode=rwc", db_path.display()))
                .await
                .expect("create sqlite pool");
            sqlx::query(crate::tracked_files_schema::CREATE_TRACKED_FILES_V37_SQL)
                .execute(&pool)
                .await
                .unwrap();
            sqlx::query(crate::tracked_files_schema::CREATE_QDRANT_CHUNKS_SQL)
                .execute(&pool)
                .await
                .unwrap();
            // tracked_files carries a FOREIGN KEY to watch_folders and sqlx
            // enables foreign_keys by default — the parent table AND the
            // 'wf-gate' row must exist for the seed insert below.
            sqlx::query(crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
                .execute(&pool)
                .await
                .unwrap();
            sqlx::query(
                "INSERT INTO watch_folders \
                   (watch_id, path, collection, tenant_id, created_at, updated_at) \
                 VALUES ('wf-gate', ?1, 'projects', 'tenant-gate', \
                         '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            )
            .bind(&base_path)
            .execute(&pool)
            .await
            .unwrap();
            let manager = QueueManager::new(pool.clone());
            manager.init_unified_queue().await.unwrap();

            // Seed: same hash as on disk, STALE fingerprint.
            let mut tx = pool.begin().await.unwrap();
            crate::tracked_files_schema::insert_tracked_file_tx(
                &mut tx,
                "wf-gate",
                "src/lib.rs",
                Some("main"),
                Some("code"),
                Some("rust"),
                "2026-01-01T00:00:00Z",
                &real_hash,
                3,
                Some("tree_sitter"),
                Some("0:rust:stale0000000"),
                crate::tracked_files_schema::ProcessingStatus::Done,
                crate::tracked_files_schema::ProcessingStatus::Done,
                None,
                None,
                false,
                Some("bp-gate"),
                None,
            )
            .await
            .unwrap();
            tx.commit().await.unwrap();

            let ctx = build_test_context(pool.clone());
            // Hand-built leased item: prepare_update only reads identity
            // fields; the queue row itself is not required (decision storage
            // is best-effort).
            let item: crate::unified_queue_schema::UnifiedQueueItem = serde_json::from_str(
                r#"{
                    "queue_id": "q-gate",
                    "idempotency_key": "i-gate",
                    "item_type": "file",
                    "op": "update",
                    "tenant_id": "tenant-gate",
                    "collection": "projects",
                    "status": "in_progress",
                    "branch": "main",
                    "payload_json": "{}",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z"
                }"#,
            )
            .unwrap();
            let payload: FilePayload =
                serde_json::from_str(r#"{"file_path":"src/lib.rs"}"#).unwrap();

            let run_gate = |stamp: Option<String>| {
                let ctx = &ctx;
                let pool = &pool;
                let file_abs = &file_abs;
                let base_path = &base_path;
                let abs_str = &abs_str;
                let item = &item;
                let payload = &payload;
                async move {
                    if let Some(fp) = stamp {
                        sqlx::query("UPDATE tracked_files SET chunker_version = ?1")
                            .bind(fp)
                            .execute(pool)
                            .await
                            .unwrap();
                    } else {
                        sqlx::query("UPDATE tracked_files SET chunker_version = NULL")
                            .execute(pool)
                            .await
                            .unwrap();
                    }
                    prepare_update(
                        ctx, item, pool, file_abs, "wf-gate", base_path, "src/lib.rs", abs_str,
                        payload, false,
                    )
                    .await
                    .unwrap()
                }
            };

            // Stale fingerprint → re-process despite the hash match.
            let action = run_gate(Some("0:rust:stale0000000".to_string())).await;
            assert!(
                action == UpdateAction::Proceed,
                "stale chunker fingerprint must re-process the unchanged file"
            );

            // Current fingerprint → skip.
            let current = crate::tree_sitter::chunker::chunking_fingerprint(Some("rust"));
            let action = run_gate(Some(current)).await;
            assert!(
                action == UpdateAction::Skip,
                "hash + current fingerprint must skip"
            );

            // NULL (legacy row) → grandfathered skip.
            let action = run_gate(None).await;
            assert!(
                action == UpdateAction::Skip,
                "NULL fingerprint is grandfathered and must skip"
            );
        }
    }
}
