//! File content ingestion pipeline — shared by add and update paths.
//! Orchestrates: document parsing → embedding → keyword extraction → graph
//! extraction → component detection → Qdrant upsert → FTS5 indexing.

use std::path::Path;
use std::time::Instant;

use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::processing_timings::{self, PhaseTiming};
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    DestinationStatus, FilePayload, QueueOperation, UnifiedQueueItem,
};

use super::chunk_embed;
use super::component;
use super::fts5_index;
use super::grammar;
use super::graph_ingest;
use super::keyword_extract;
use super::keyword_persist;
use super::store_track;

/// Ingest file content: embedding, Qdrant upsert, tracked_files, FTS5.
///
/// Shared by both add and update paths (after update preamble completes).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn ingest_file_content(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    payload: &FilePayload,
    abs_file_path: &str,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
) -> UnifiedProcessorResult<()> {
    // === PER-DESTINATION RETRY SKIP (Task 6) ===
    let qdrant_already_done = item.qdrant_status == Some(DestinationStatus::Done);
    if qdrant_already_done {
        return handle_retry_skip(
            ctx,
            item,
            pool,
            watch_folder_id,
            relative_path,
            abs_file_path,
            payload,
        )
        .await;
    }

    // === CROSS-BRANCH DEDUP FAST-PATH ===
    // If another branch already indexed an identical file (same watch +
    // path + file_hash), copy its Qdrant points under the new base_point
    // instead of re-running parse + embed. See branch_dedup.rs and
    // docs/specs/21-cross-branch-dedup.md. The probe is one SQL hit on
    // the idx_tracked_files_dedup index — cheap when there's no match.
    match super::branch_dedup::try_branch_dedup(
        ctx,
        item,
        payload,
        file_path,
        abs_file_path,
        relative_path,
        watch_folder_id,
    )
    .await
    {
        Ok(Some(_hit)) => return Ok(()),
        Ok(None) => { /* novel content — fall through to full ingest */ }
        Err(e) => {
            // Dedup failure is not fatal — log and proceed with the full
            // ingest path. The data ends up correct either way.
            warn!(
                "branch_dedup probe failed for {} ({}): {} — falling back to full ingest",
                relative_path, abs_file_path, e
            );
        }
    }

    // Mark qdrant status as in_progress
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::InProgress)
        .await;

    // Run the main ingest pipeline
    run_ingest_pipeline(
        ctx,
        item,
        pool,
        file_path,
        payload,
        abs_file_path,
        watch_folder_id,
        base_path,
        relative_path,
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
    abs_file_path: &str,
    _payload: &FilePayload,
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
                .update_destination_status(&item.queue_id, "search", DestinationStatus::InProgress)
                .await;
            match fts5_index::update_fts5_for_file_or_enqueue(
                sdb,
                pool,
                existing.file_id,
                abs_file_path,
                &item.tenant_id,
                Some(&item.branch),
                existing.base_point.as_deref(),
                Some(relative_path),
                Some(existing.file_hash.as_str()),
                &item.queue_id,
            )
            .await
            {
                Ok(fts5_index::Fts5Outcome::Enqueued) => {
                    // Batch writer owns search_status from here. Leave the
                    // row at `in_progress` so check_and_finalize keeps the
                    // queue item alive until the actor flips it to Done.
                }
                Ok(fts5_index::Fts5Outcome::Inline | fts5_index::Fts5Outcome::Skipped) => {
                    let _ = ctx
                        .queue_manager
                        .update_destination_status(
                            &item.queue_id,
                            "search",
                            DestinationStatus::Done,
                        )
                        .await;
                }
                Err(e) => {
                    warn!(
                        "FTS5 retry failed for {} — search_status set to Failed: {}",
                        relative_path, e
                    );
                    let _ = ctx
                        .queue_manager
                        .update_destination_status(
                            &item.queue_id,
                            "search",
                            DestinationStatus::Failed,
                        )
                        .await;
                }
            }
        } else {
            let _ = ctx
                .queue_manager
                .update_destination_status(&item.queue_id, "search", DestinationStatus::Done)
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
#[allow(clippy::too_many_arguments)]
async fn run_ingest_pipeline(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    payload: &FilePayload,
    abs_file_path: &str,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
) -> UnifiedProcessorResult<()> {
    let mut timings: Vec<PhaseTiming> = Vec::new();

    let overrides = component::get_gitattributes(ctx, base_path).await;
    let detected_language =
        crate::tree_sitter::detect_language_with_overrides(file_path, relative_path, &overrides);

    let provider =
        grammar::ensure_grammar_available(ctx, file_path, relative_path, &overrides).await;
    let (document_content, file_document_id, file_hash, base_point) = parse_document(
        ctx,
        item,
        payload,
        file_path,
        abs_file_path,
        relative_path,
        provider,
        &mut timings,
    )
    .await?;

    let (points, chunk_records, lsp_status, treesitter_status) = run_middle_phases(
        ctx,
        item,
        pool,
        file_path,
        &document_content,
        &file_document_id,
        watch_folder_id,
        base_path,
        relative_path,
        &base_point,
        &file_hash,
        payload.file_type.as_deref(),
        &mut timings,
    )
    .await?;

    let file_id = upsert_and_mark_done(
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
        payload,
        &mut timings,
    )
    .await?;

    finish_pipeline(
        ctx,
        item,
        pool,
        file_id,
        payload,
        abs_file_path,
        &base_point,
        relative_path,
        &file_hash,
        detected_language,
        &mut timings,
    )
    .await;

    Ok(())
}

/// FTS5 indexing + timing record + success log (phases 7 and final).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn finish_pipeline(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_id: i64,
    payload: &FilePayload,
    abs_file_path: &str,
    base_point: &str,
    relative_path: &str,
    file_hash: &str,
    detected_language: Option<&'static str>,
    timings: &mut Vec<PhaseTiming>,
) {
    update_search_index(
        ctx,
        item,
        pool,
        file_id,
        payload,
        abs_file_path,
        base_point,
        relative_path,
        file_hash,
        timings,
    )
    .await;
    record_pipeline_timings(pool, item, detected_language, timings).await;
    info!(
        "Successfully processed file item {} ({})",
        item.queue_id,
        payload.file_path.as_str()
    );
}

/// Phases 2–5: embed chunks, extract keywords/graph, inject component.
///
/// Returns `(points, chunk_records, lsp_status, treesitter_status)`.
#[allow(clippy::too_many_arguments)]
async fn run_middle_phases(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    document_content: &crate::DocumentContent,
    file_document_id: &str,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_type: Option<&str>,
    timings: &mut Vec<PhaseTiming>,
) -> UnifiedProcessorResult<(
    Vec<crate::storage::DocumentPoint>,
    Vec<chunk_embed::ChunkRecord>,
    crate::tracked_files_schema::ProcessingStatus,
    crate::tracked_files_schema::ProcessingStatus,
)> {
    // Phase 2: embed chunks
    let t0 = Instant::now();
    let embed_result = chunk_embed::embed_chunks(
        ctx,
        item,
        document_content,
        file_path,
        file_document_id,
        relative_path,
        base_point,
        file_hash,
        file_type,
    )
    .await?;
    timings.push(PhaseTiming {
        phase: "embed",
        duration_ms: t0.elapsed().as_millis() as u64,
    });

    let mut points = embed_result.points;
    let chunk_records = embed_result.chunk_records;
    let lsp_status = embed_result.lsp_status;
    let treesitter_status = embed_result.treesitter_status;

    // Phases 3–4: keyword extraction + graph edges
    run_keyword_and_graph_phases(
        ctx,
        item,
        pool,
        file_path,
        document_content,
        file_document_id,
        relative_path,
        &mut points,
        timings,
    )
    .await;

    // Phase 5: component detection + injection
    component::inject_component(
        ctx,
        pool,
        watch_folder_id,
        base_path,
        relative_path,
        &mut points,
    )
    .await;

    Ok((points, chunk_records, lsp_status, treesitter_status))
}

/// Parse the document and compute file identifiers (phase 0 + 1).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn parse_document(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &FilePayload,
    file_path: &Path,
    abs_file_path: &str,
    relative_path: &str,
    provider: Option<std::sync::Arc<dyn crate::tree_sitter::parser::LanguageProvider>>,
    timings: &mut Vec<PhaseTiming>,
) -> UnifiedProcessorResult<(crate::DocumentContent, String, String, String)> {
    let t0 = Instant::now();
    let document_content = ctx
        .document_processor
        .process_file_content_with_provider(file_path, &item.collection, provider)
        .await
        .map_err(|e| UnifiedProcessorError::ProcessingFailed(e.to_string()))?;
    timings.push(PhaseTiming {
        phase: "parse",
        duration_ms: t0.elapsed().as_millis() as u64,
    });
    info!(
        "Extracted {} chunks from {}",
        document_content.chunks.len(),
        payload.file_path.as_str()
    );

    let file_document_id = crate::generate_document_id(&item.tenant_id, abs_file_path);
    let file_hash = tracked_files_schema::compute_file_hash(file_path)
        .unwrap_or_else(|_| "unknown".to_string());
    let base_point = wqm_common::hashing::compute_base_point(
        &item.tenant_id,
        &item.branch,
        relative_path,
        &file_hash,
    );

    Ok((document_content, file_document_id, file_hash, base_point))
}

/// Run keyword extraction (phase 3) and graph edge ingestion (phase 4).
#[allow(clippy::too_many_arguments)]
async fn run_keyword_and_graph_phases(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_path: &Path,
    document_content: &crate::DocumentContent,
    file_document_id: &str,
    relative_path: &str,
    points: &mut [crate::storage::DocumentPoint],
    timings: &mut Vec<PhaseTiming>,
) {
    if matches!(
        item.op,
        QueueOperation::Add | QueueOperation::Update | QueueOperation::Uplift
    ) {
        let t0 = Instant::now();
        let extraction =
            keyword_extract::run_keyword_extraction(ctx, item, file_path, document_content, points)
                .await;
        timings.push(PhaseTiming {
            phase: "extract",
            duration_ms: t0.elapsed().as_millis() as u64,
        });
        if let Some(ref extraction) = extraction {
            keyword_persist::persist_extraction(
                pool,
                file_document_id,
                &item.tenant_id,
                &item.collection,
                extraction,
            )
            .await;
        }
    }

    let t0 = Instant::now();
    let abs_file_path = file_path.to_string_lossy();
    graph_ingest::ingest_graph_edges(
        ctx,
        &item.tenant_id,
        relative_path,
        &abs_file_path,
        &document_content.chunks,
    )
    .await;
    timings.push(PhaseTiming {
        phase: "graph",
        duration_ms: t0.elapsed().as_millis() as u64,
    });
}

/// Upsert into Qdrant + tracked_files, then mark qdrant destination done (phase 6).
#[allow(clippy::too_many_arguments)]
async fn upsert_and_mark_done(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    points: Vec<crate::storage::DocumentPoint>,
    chunk_records: &[chunk_embed::ChunkRecord],
    watch_folder_id: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_path: &Path,
    document_content: &crate::DocumentContent,
    lsp_status: crate::tracked_files_schema::ProcessingStatus,
    treesitter_status: crate::tracked_files_schema::ProcessingStatus,
    payload: &FilePayload,
    timings: &mut Vec<PhaseTiming>,
) -> UnifiedProcessorResult<i64> {
    let t0 = Instant::now();
    let file_id = store_track::upsert_and_track(
        ctx,
        item,
        pool,
        points,
        chunk_records,
        watch_folder_id,
        relative_path,
        base_point,
        file_hash,
        file_path,
        document_content,
        lsp_status,
        treesitter_status,
        payload.file_type.as_deref(),
        None,
    )
    .await?;
    timings.push(PhaseTiming {
        phase: "upsert",
        duration_ms: t0.elapsed().as_millis() as u64,
    });

    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "qdrant", DestinationStatus::Done)
        .await;

    Ok(file_id)
}

/// Update FTS5 search index for a file (phase 7).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn update_search_index(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_id: i64,
    _payload: &FilePayload,
    abs_file_path: &str,
    base_point: &str,
    relative_path: &str,
    file_hash: &str,
    timings: &mut Vec<PhaseTiming>,
) {
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "search", DestinationStatus::InProgress)
        .await;
    let t0 = Instant::now();
    // Tri-state: Enqueued = batch actor owns the next flip, Done = set
    // it now (inline write succeeded or no search_db configured),
    // Failed = set it now (inline write errored).
    let final_status: Option<DestinationStatus> = if let Some(sdb) = &ctx.search_db {
        match fts5_index::update_fts5_for_file_or_enqueue(
            sdb,
            pool,
            file_id,
            abs_file_path,
            &item.tenant_id,
            Some(&item.branch),
            Some(base_point),
            Some(relative_path),
            Some(file_hash),
            &item.queue_id,
        )
        .await
        {
            Ok(fts5_index::Fts5Outcome::Enqueued) => None,
            Ok(fts5_index::Fts5Outcome::Inline | fts5_index::Fts5Outcome::Skipped) => {
                Some(DestinationStatus::Done)
            }
            Err(e) => {
                warn!(
                    "FTS5 indexing failed for {} — search_status set to Failed: {}",
                    relative_path, e
                );
                Some(DestinationStatus::Failed)
            }
        }
    } else {
        Some(DestinationStatus::Done)
    };
    timings.push(PhaseTiming {
        phase: "fts5",
        duration_ms: t0.elapsed().as_millis() as u64,
    });
    if let Some(status) = final_status {
        let _ = ctx
            .queue_manager
            .update_destination_status(&item.queue_id, "search", status)
            .await;
    }
}

/// Record pipeline phase timings to the database.
async fn record_pipeline_timings(
    pool: &SqlitePool,
    item: &UnifiedQueueItem,
    detected_language: Option<&'static str>,
    timings: &[PhaseTiming],
) {
    processing_timings::record_timings(
        pool,
        &item.queue_id,
        item.item_type.as_str(),
        item.op.as_str(),
        &item.tenant_id,
        &item.collection,
        detected_language,
        timings,
    )
    .await;
}
