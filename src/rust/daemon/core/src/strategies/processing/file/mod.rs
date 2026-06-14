//! File processing strategy.
//!
//! Handles `ItemType::File` queue items: ingestion (add/update) and deletion,
//! including tracked_files management, Qdrant upsert, FTS5 indexing, LSP
//! enrichment, and keyword/tag extraction.
//!
//! Split into focused submodules:
//! - `chunk_embed` — per-chunk embedding, payload construction, LSP enrichment
//! - `dedup` — content-hash deduplication for cross-branch file ingestion
//! - `delete` — delete operation, missing-file cleanup, Qdrant failure handling
//! - `dependency_ingest` — dependency manifest parsing and storage for grouping
//! - `fts5_index` — FTS5 code search index updates
//! - `keyword_extract` — keyword/tag extraction pipeline
//! - `lsp_payload` — LSP enrichment payload serialization
//! - `parse` — document parse + identifier phase (extract.document span)
//! - `store_track` — Qdrant upsert + tracked_files/qdrant_chunks transaction
//! - `update_preamble` — hash comparison + reference-counted old point deletion
//! - `zero_byte` — graceful handling of empty (0-byte) files

mod chunk_embed;
mod component;
mod dedup;
mod delete;
mod dependency_ingest;
mod discovery_trigger;
mod fts5_index;
mod grammar;
mod graph_ingest;
mod ingest;
mod keyword_extract;
mod keyword_persist;
pub(crate) mod lsp_payload;
mod narrative_phase;
mod parse;
mod store_track;
mod update_preamble;
mod zero_byte;

use std::path::Path;

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, error, info, warn};

use crate::config::IngestionLimitsConfig;
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

        // SkipNotAllowed can short-circuit before resolving the watch folder:
        // a non-allowlisted extension is not indexable data and records nothing.
        // SkipOversized is handled after the watch folder is resolved (below).
        let guard = ingestion_guard(ctx, item, &payload);
        if let IngestionGuard::SkipNotAllowed = guard {
            return Ok(());
        }

        let pool = ctx.queue_manager.pool();
        let (watch_folder_id, base_path) = resolve_watch_folder(pool, item).await?;

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

        // An oversized allowlisted file (e.g. a 24 MB JSON) is user data we
        // decline to index. Record it as a 0-chunk tracked file so it is
        // visible and reconcile does not re-enqueue it forever (#113).
        if let IngestionGuard::SkipOversized = guard {
            return record_oversized_skip(
                ctx,
                item,
                pool,
                &watch_folder_id,
                relative_path,
                file_path,
            )
            .await;
        }

        crate::shared::ensure_collection(
            &ctx.storage_client,
            &item.collection,
            ctx.embedding_generator.dense_dim() as u64,
        )
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

        // Security: verify resolved path stays within project root.
        let base_path_ref = Path::new(&base_path);
        if !wqm_common::paths::is_within_boundary(file_path, base_path_ref) {
            warn!(
                "Symlink boundary escape detected: {} resolves outside project root {}",
                abs_file_path, base_path
            );
            return Err(UnifiedProcessorError::InvalidPayload(format!(
                "resolved path escapes project boundary: {} (root: {})",
                abs_file_path, base_path
            )));
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

        // Size-gate fallback (#121): some enqueue paths (branch switch, startup
        // recovery, reconciliation, the registration scan, …) leave
        // `size_bytes` unset, which silently bypasses the per-extension limit in
        // `ingestion_guard`. Stat the file here so an oversized data dump (e.g. a
        // 17 MB tokenizer.json) is recorded as a 0-chunk skip rather than chunked
        // into thousands of embeddings and flooding the embedder.
        if payload.size_bytes.is_none() {
            if let Ok(size) = std::fs::metadata(file_path).map(|m| m.len()) {
                let ext = crate::file_classification::get_extension_for_storage(file_path)
                    .unwrap_or_default();
                if let Some(limit) = extension_limit_exceeded(&ctx.ingestion_limits, &ext, size) {
                    warn!(
                        extension = %ext,
                        size_kb = size / 1024,
                        limit_kb = limit / 1024,
                        path = %relative_path,
                        "Skipping oversized file (size from stat; payload had no \
                         size_bytes): exceeds per-extension limit"
                    );
                    return record_oversized_skip(
                        ctx,
                        item,
                        pool,
                        &watch_folder_id,
                        relative_path,
                        file_path,
                    )
                    .await;
                }
            }
        }

        if item.op == QueueOperation::Update {
            if prepare_update(
                ctx,
                item,
                pool,
                file_path,
                &watch_folder_id,
                relative_path,
                &abs_file_path,
                &payload,
            )
            .await?
                == UpdateAction::Skip
            {
                if item.decision_json.is_some() {
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
                }
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

        // Content-hash dedup: if identical content already exists under a
        // different branch, skip embedding and just add this branch.
        // Only applies to Add operations (Update already handles hash comparison
        // in prepare_update, and Uplift intentionally re-processes). Forced
        // re-ingests (needs_reconcile repairs, #110) bypass dedup: its
        // branch-already-present early return assumes stored state is intact,
        // which is exactly what the repair is rebuilding.
        if item.op == QueueOperation::Add && !force_reingest(item) {
            if let Some(()) = dedup::try_dedup(
                ctx,
                item,
                pool,
                file_path,
                &watch_folder_id,
                relative_path,
                &abs_file_path,
                &item.branch,
            )
            .await?
            {
                return Ok(());
            }
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

/// Outcome of the pre-ingestion guards for a file item.
enum IngestionGuard {
    /// File passes all guards — proceed with ingestion.
    Proceed,
    /// Extension is not in the collection's allowlist (binary, image, etc.).
    /// Not indexable content and not user data — skip silently, record nothing.
    SkipNotAllowed,
    /// An allowlisted text file that exceeds its per-extension size limit
    /// (e.g. a 24 MB JSON data dump). This IS user data we deliberately decline
    /// to index, so it must be recorded as a 0-chunk tracked file rather than
    /// vanish — otherwise reconcile re-enqueues it on every run (#113).
    SkipOversized,
}

/// Check allowlist and per-extension size limit for a file item.
fn ingestion_guard(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &FilePayload,
) -> IngestionGuard {
    if item.op == QueueOperation::Delete {
        return IngestionGuard::Proceed;
    }

    let rel = payload.file_path.as_str();
    if !ctx.allowed_extensions.is_allowed(rel, &item.collection) {
        debug!(
            "File type not in allowlist, skipping: {} (collection={})",
            rel, item.collection
        );
        return IngestionGuard::SkipNotAllowed;
    }

    if let Some(size) = payload.size_bytes {
        let ext = crate::file_classification::get_extension_for_storage(Path::new(rel))
            .unwrap_or_default();
        if let Some(limit) = extension_limit_exceeded(&ctx.ingestion_limits, &ext, size) {
            warn!(
                extension = %ext,
                size_kb = size / 1024,
                limit_kb = limit / 1024,
                path = %rel,
                "Skipping oversized file: exceeds per-extension limit"
            );
            return IngestionGuard::SkipOversized;
        }
    }

    IngestionGuard::Proceed
}

/// If `size` (bytes) exceeds the configured per-extension ingestion limit for
/// `ext`, return that limit (bytes); otherwise `None`. Extensions without a
/// configured limit are unbounded. Accepts the extension with or without a
/// leading dot.
///
/// Shared by the fast-path gate (`ingestion_guard`, using the payload's
/// `size_bytes`) and the stat fallback in `process_file_item` (which covers
/// enqueue paths that leave `size_bytes` unset). See #121.
fn extension_limit_exceeded(limits: &IngestionLimitsConfig, ext: &str, size: u64) -> Option<u64> {
    limits.size_limit_bytes(ext).filter(|&limit| size > limit)
}

/// Record an oversized (size-restricted) file as a 0-chunk skipped tracked
/// file, then mark the queue item done. The `warn!` describing why was already
/// emitted by `ingestion_guard`.
///
/// Recording it (rather than silently returning) keeps the file visible and,
/// crucially, stops ignore-reconciliation from re-enqueueing it on every run —
/// reconcile treats any path absent from `tracked_files` as "missing" (#113).
///
/// If the path was previously ingested with chunks (it grew past the gate, or
/// was indexed before its enqueue path set `size_bytes`, #121), its Qdrant
/// points and FTS5 lines are now orphaned. Purge them before recording the skip
/// so reconcile-driven re-processing of already-ingested oversized files
/// (filesystem_reconcile self-heal) actually removes the stale data rather than
/// just flipping the tracked row to 0 chunks. Qdrant failures propagate so the
/// queue item retries; FTS cleanup is non-fatal.
async fn record_oversized_skip(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    file_path: &Path,
) -> UnifiedProcessorResult<()> {
    let existing = tracked_files_schema::lookup_tracked_file(
        pool,
        watch_folder_id,
        relative_path,
        Some(item.branch.as_str()),
    )
    .await
    .ok()
    .flatten();
    if let Some(ref ex) = existing {
        if ex.chunk_count > 0 {
            let abs_file_path = file_path.to_string_lossy();
            ctx.storage_client
                .delete_points_by_filter(&item.collection, &abs_file_path, &item.tenant_id)
                .await
                .map_err(|e| {
                    UnifiedProcessorError::Storage(format!(
                        "oversized cleanup: Qdrant delete failed for {} ({} stale chunks): {}",
                        relative_path, ex.chunk_count, e
                    ))
                })?;
            if let Some(sdb) = &ctx.search_db {
                let processor = crate::fts_batch_processor::FtsBatchProcessor::new(
                    sdb,
                    crate::fts_batch_processor::FtsBatchConfig::default(),
                );
                if let Err(e) = processor.delete_file(ex.file_id).await {
                    warn!(
                        "oversized cleanup: FTS5 delete failed for file_id={}: {} (non-fatal)",
                        ex.file_id, e
                    );
                }
            }
            info!(
                "oversized cleanup: purged {} stale chunks for now-skipped file {}",
                ex.chunk_count, relative_path
            );
        }
    }

    let file_hash = tracked_files_schema::compute_file_hash(file_path)
        .unwrap_or_else(|_| "unknown".to_string());
    let file_mtime = tracked_files_schema::get_file_mtime(file_path)
        .unwrap_or_else(|_| wqm_common::timestamps::now_utc());
    let extension = crate::file_classification::get_extension_for_storage(file_path);
    let is_test = crate::file_classification::is_test_file(file_path);
    let base_point =
        wqm_common::hashing::compute_base_point(&item.tenant_id, relative_path, &file_hash);

    zero_byte::record_tracked_file(
        pool,
        item,
        watch_folder_id,
        relative_path,
        &file_mtime,
        &file_hash,
        extension.as_deref(),
        is_test,
        &base_point,
        None, // file_type: classification not needed for a skipped file
    )
    .await?;

    zero_byte::mark_destinations_done(ctx, item).await;
    // Make the size-gate skip observable per tenant (#118) — otherwise skips are
    // invisible and a flood of oversized files (the #113/#121 failure mode) looks
    // like silence.
    crate::monitoring::METRICS.inc_files_size_skipped(&item.tenant_id);
    Ok(())
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
    info!(
        "File no longer exists, cleaned up and dequeuing: {}",
        payload.file_path.as_str()
    );
    Ok(())
}

/// True when the queue item's metadata carries `"force_reingest": true` —
/// set by reconcile-driven enqueues (`needs_reconcile` repairs, #110) whose
/// whole purpose is rebuilding stored state for content that did not change.
pub(super) fn force_reingest(item: &UnifiedQueueItem) -> bool {
    item.metadata
        .as_deref()
        .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
        .is_some_and(|v| v.get("force_reingest").and_then(|f| f.as_bool()) == Some(true))
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
            // Reconcile-driven repairs re-ingest even when content is
            // unchanged: the flag means stored state (e.g. FTS rows, #110)
            // is broken while the file itself never changed. Points carry
            // content-derived ids, so the upsert is idempotent and the
            // deletion preamble is unnecessary.
            if force_reingest(item) {
                info!(
                    "File unchanged but force_reingest set (needs_reconcile repair), \
                     re-ingesting: {}",
                    relative_path
                );
                return Ok(UpdateAction::Proceed);
            }
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
    } else {
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

    fn item_with_metadata(metadata: Option<&str>) -> UnifiedQueueItem {
        UnifiedQueueItem {
            queue_id: "q1".to_string(),
            idempotency_key: "k1".to_string(),
            item_type: ItemType::File,
            op: QueueOperation::Update,
            tenant_id: "t1".to_string(),
            collection: "projects".to_string(),
            status: crate::unified_queue_schema::QueueStatus::Pending,
            branch: "main".to_string(),
            payload_json: "{}".to_string(),
            metadata: metadata.map(str::to_string),
            created_at: String::new(),
            updated_at: String::new(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            error_message: None,
            last_error_at: None,
            file_path: None,
            size_bytes: None,
            qdrant_status: None,
            search_status: None,
            decision_json: None,
        }
    }

    #[test]
    fn extension_limit_exceeded_respects_per_extension_cap() {
        let limits = IngestionLimitsConfig::default(); // json => 500 KB
        let over = 600 * 1024;
        let under = 400 * 1024;
        // Over the cap returns the limit; under returns None.
        assert_eq!(
            extension_limit_exceeded(&limits, "json", over),
            Some(500 * 1024)
        );
        assert_eq!(extension_limit_exceeded(&limits, "json", under), None);
        // Unconfigured extension is unbounded.
        assert_eq!(extension_limit_exceeded(&limits, "rs", over), None);
        // Leading dot is tolerated.
        assert_eq!(
            extension_limit_exceeded(&limits, ".json", over),
            Some(500 * 1024)
        );
        // Exactly at the limit is not "exceeded".
        assert_eq!(extension_limit_exceeded(&limits, "json", 500 * 1024), None);
    }

    #[test]
    fn force_reingest_true_only_for_explicit_flag() {
        assert!(force_reingest(&item_with_metadata(Some(
            r#"{"source":"needs_reconcile","force_reingest":true}"#
        ))));
        assert!(!force_reingest(&item_with_metadata(Some(
            r#"{"force_reingest":false}"#
        ))));
        assert!(!force_reingest(&item_with_metadata(Some(
            r#"{"source":"mcp_rules_tool"}"#
        ))));
        assert!(!force_reingest(&item_with_metadata(Some("not json"))));
        assert!(!force_reingest(&item_with_metadata(None)));
    }
}
