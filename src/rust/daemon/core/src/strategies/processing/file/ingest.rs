//! File content ingestion pipeline.
//!
//! Shared by both add and update paths (after update preamble completes).
//! Orchestrates: document parsing → embedding → keyword extraction → graph
//! extraction → component detection → Qdrant upsert → FTS5 indexing.

use std::path::Path;
use std::time::Instant;

use sqlx::SqlitePool;
use tracing::{debug, info};

use crate::context::ProcessingContext;
use crate::processing_timings::{self, PhaseTiming};
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    DestinationStatus, FilePayload, QueueOperation, UnifiedQueueItem,
};

use super::chunk_embed;
use super::fts5_index;
use super::graph_ingest;
use super::keyword_extract;
use super::keyword_persist;
use super::store_track;

/// Ingest file content: embedding, Qdrant upsert, tracked_files, FTS5.
///
/// Shared by both add and update paths (after update preamble completes).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn ingest_file_content(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    payload: &FilePayload,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
) -> UnifiedProcessorResult<()> {
    // === PER-DESTINATION RETRY SKIP (Task 6) ===
    let qdrant_already_done = item.qdrant_status == Some(DestinationStatus::Done);
    if qdrant_already_done {
        return handle_retry_skip(ctx, item, pool, watch_folder_id, relative_path, payload).await;
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

    // Run the main ingest pipeline
    run_ingest_pipeline(
        ctx, item, pool, file_path, payload, watch_folder_id, base_path, relative_path,
    )
    .await
}

/// Handle retry-skip path: Qdrant already done, only update FTS5.
#[allow(clippy::too_many_arguments)]
async fn handle_retry_skip(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    watch_folder_id: &str,
    relative_path: &str,
    payload: &FilePayload,
) -> UnifiedProcessorResult<()> {
    info!(
        "Qdrant already done for {} (retry), skipping to search DB update",
        item.queue_id
    );

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

    Ok(())
}

/// Run the main file ingestion pipeline (parse → embed → extract → upsert → FTS5).
#[allow(clippy::too_many_arguments)]
async fn run_ingest_pipeline(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    payload: &FilePayload,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
) -> UnifiedProcessorResult<()> {
    let mut timings: Vec<PhaseTiming> = Vec::new();

    // Phase 1: Parse document
    let t0 = Instant::now();
    let document_content = ctx
        .document_processor
        .process_file_content(file_path, &item.collection)
        .await
        .map_err(|e| UnifiedProcessorError::ProcessingFailed(e.to_string()))?;
    timings.push(PhaseTiming { phase: "parse", duration_ms: t0.elapsed().as_millis() as u64 });

    info!(
        "Extracted {} chunks from {}",
        document_content.chunks.len(),
        payload.file_path
    );

    let file_document_id =
        crate::generate_document_id(&item.tenant_id, &payload.file_path);

    let file_hash = tracked_files_schema::compute_file_hash(file_path)
        .unwrap_or_else(|_| "unknown".to_string());
    let base_point = wqm_common::hashing::compute_base_point(
        &item.tenant_id,
        &item.branch,
        relative_path,
        &file_hash,
    );

    // Phase 2: Embed chunks
    let t0 = Instant::now();
    let embed_result = chunk_embed::embed_chunks(
        ctx, item, &document_content, file_path, &file_document_id,
        relative_path, &base_point, &file_hash, payload.file_type.as_deref(),
    )
    .await?;
    timings.push(PhaseTiming { phase: "embed", duration_ms: t0.elapsed().as_millis() as u64 });

    let mut points = embed_result.points;
    let chunk_records = embed_result.chunk_records;
    let lsp_status = embed_result.lsp_status;
    let treesitter_status = embed_result.treesitter_status;

    // Phase 3: Keyword/tag extraction
    if item.op == QueueOperation::Add || item.op == QueueOperation::Update {
        let t0 = Instant::now();
        let extraction = keyword_extract::run_keyword_extraction(
            ctx, item, file_path, &document_content, &mut points,
        )
        .await;
        timings.push(PhaseTiming { phase: "extract", duration_ms: t0.elapsed().as_millis() as u64 });

        if let Some(ref extraction) = extraction {
            keyword_persist::persist_extraction(
                pool, &file_document_id, &item.tenant_id, &item.collection, extraction,
            )
            .await;
        }
    }

    // Phase 4: Graph relationship extraction
    let t0 = Instant::now();
    graph_ingest::ingest_graph_edges(
        ctx, &item.tenant_id, relative_path, &document_content.chunks,
    )
    .await;
    timings.push(PhaseTiming { phase: "graph", duration_ms: t0.elapsed().as_millis() as u64 });

    // Phase 5: Component detection + injection
    inject_component(ctx, pool, watch_folder_id, base_path, relative_path, &mut points).await;

    // Phase 6: Qdrant upsert + tracked_files
    let t0 = Instant::now();
    let file_id = store_track::upsert_and_track(
        ctx, item, pool, points, &chunk_records, watch_folder_id, relative_path,
        &base_point, &file_hash, file_path, &document_content,
        lsp_status, treesitter_status, payload.file_type.as_deref(), None,
    )
    .await?;
    timings.push(PhaseTiming { phase: "upsert", duration_ms: t0.elapsed().as_millis() as u64 });

    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::Done)
        .await;

    // Phase 7: FTS5 search index
    update_search_index(ctx, item, pool, file_id, payload, &base_point, relative_path, &file_hash, &mut timings).await;

    // Record per-phase timings
    processing_timings::record_timings(
        pool, &item.queue_id, item.item_type.as_str(), item.op.as_str(),
        &item.tenant_id, &item.collection, &timings,
    )
    .await;

    info!(
        "Successfully processed file item {} ({})",
        item.queue_id, payload.file_path
    );

    Ok(())
}

/// Detect and inject component_id into point payloads.
async fn inject_component(
    ctx: &ProcessingContext,
    pool: &SqlitePool,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
    points: &mut [crate::storage::DocumentPoint],
) {
    if is_workspace_definition_file(relative_path) {
        let mut cache = ctx.component_cache.write().await;
        cache.remove(watch_folder_id);
        debug!(
            "Invalidated component cache for {} (workspace file changed: {})",
            watch_folder_id, relative_path
        );
    }

    let component = resolve_component(ctx, pool, watch_folder_id, base_path, relative_path).await;

    if let Some(ref comp) = component {
        for point in points.iter_mut() {
            point
                .payload
                .insert("component_id".to_string(), serde_json::json!(comp));
        }
    }
}

/// Update FTS5 search index for a file.
#[allow(clippy::too_many_arguments)]
async fn update_search_index(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_id: i64,
    payload: &FilePayload,
    base_point: &str,
    relative_path: &str,
    file_hash: &str,
    timings: &mut Vec<PhaseTiming>,
) {
    let _ = ctx
        .queue_manager
        .update_destination_status(
            &item.queue_id,
            "search",
            DestinationStatus::InProgress,
        )
        .await;
    let t0 = Instant::now();
    if let Some(sdb) = &ctx.search_db {
        fts5_index::update_fts5_for_file(
            sdb, pool, file_id, &payload.file_path, &item.tenant_id,
            Some(&item.branch), Some(base_point), Some(relative_path), Some(file_hash),
        )
        .await;
    }
    timings.push(PhaseTiming { phase: "fts5", duration_ms: t0.elapsed().as_millis() as u64 });
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
        .await;
}

/// Check if a file is a workspace definition file that triggers component re-detection.
fn is_workspace_definition_file(relative_path: &str) -> bool {
    let filename = relative_path.rsplit('/').next().unwrap_or(relative_path);
    filename == "Cargo.toml" || filename == "package.json"
}

/// Resolve the component for a file, using the per-watch-folder cache.
///
/// On cache miss: detects components from the project's workspace files,
/// persists them to `project_components`, and caches the result.
async fn resolve_component(
    ctx: &ProcessingContext,
    pool: &SqlitePool,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
) -> Option<String> {
    use crate::component_detection;

    // Fast path: check cache
    {
        let cache = ctx.component_cache.read().await;
        if let Some(components) = cache.get(watch_folder_id) {
            return component_detection::assign_component(relative_path, components)
                .map(|c| c.id.clone());
        }
    }

    // Slow path: detect from filesystem, persist, and cache
    let project_path = Path::new(base_path);
    let components = component_detection::detect_components(project_path);

    if !components.is_empty() {
        if let Err(e) =
            component_detection::persist_components(pool, watch_folder_id, &components).await
        {
            debug!("Failed to persist components for {}: {}", watch_folder_id, e);
        }
    }

    let result = component_detection::assign_component(relative_path, &components)
        .map(|c| c.id.clone());

    // Cache even if empty (avoids re-detecting for projects with no workspace)
    {
        let mut cache = ctx.component_cache.write().await;
        cache.insert(watch_folder_id.to_string(), components);
    }

    result
}
