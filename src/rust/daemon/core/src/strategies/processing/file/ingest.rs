//! File content ingestion pipeline — shared by add and update paths.
//! Orchestrates: document parsing -> embedding -> Tier 2 taxonomy tagging ->
//! keyword extraction -> graph extraction -> component detection -> Qdrant
//! upsert -> FTS5 indexing -> dependency extraction.

use std::path::Path;
use std::time::Instant;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::processing_timings::{self, PhaseTiming};
use crate::tagging::aggregate_document_embedding;
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{
    DestinationStatus, FilePayload, QueueOperation, UnifiedQueueItem,
};

use super::chunk_embed;
use super::component;
use super::dependency_ingest;
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
    // Detect the current branch from the filesystem via the TTL cache.
    // Falls back to item.branch when git detection fails (non-git dir, detached HEAD).
    let detected_branch = ctx
        .branch_cache
        .get_branch(Path::new(base_path), &item.branch);

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
            &detected_branch,
        )
        .await;
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
        &detected_branch,
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
    detected_branch: &str,
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
            Some(detected_branch),
        )
        .await
        {
            let _ = ctx
                .queue_manager
                .update_destination_status(&item.queue_id, "search", DestinationStatus::InProgress)
                .await;
            let fts_status = match fts5_index::update_fts5_for_file(
                sdb,
                pool,
                existing.file_id,
                abs_file_path,
                &item.tenant_id,
                Some(detected_branch),
                existing.base_point.as_deref(),
                Some(relative_path),
                Some(existing.file_hash.as_str()),
            )
            .await
            {
                Ok(_) => DestinationStatus::Done,
                Err(e) => {
                    warn!(
                        "FTS5 retry failed for {} — search_status set to Failed: {}",
                        relative_path, e
                    );
                    DestinationStatus::Failed
                }
            };
            let _ = ctx
                .queue_manager
                .update_destination_status(&item.queue_id, "search", fts_status)
                .await;
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

/// Run the main file ingestion pipeline (parse -> embed -> extract -> upsert -> FTS5 -> deps).
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
    detected_branch: &str,
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
        detected_branch,
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
        detected_branch,
    )
    .await?;

    finish_pipeline(
        ctx,
        item,
        pool,
        file_id,
        payload,
        file_path,
        abs_file_path,
        &base_point,
        relative_path,
        &file_hash,
        detected_language,
        &mut timings,
        detected_branch,
    )
    .await;

    Ok(())
}

/// FTS5 indexing + dependency extraction + timing record + success log.
#[allow(clippy::too_many_arguments)]
async fn finish_pipeline(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    pool: &SqlitePool,
    file_id: i64,
    payload: &FilePayload,
    file_path: &Path,
    abs_file_path: &str,
    base_point: &str,
    relative_path: &str,
    file_hash: &str,
    detected_language: Option<&'static str>,
    timings: &mut Vec<PhaseTiming>,
    detected_branch: &str,
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
        detected_branch,
    )
    .await;

    // Phase 8: dependency manifest extraction (non-blocking side effect).
    // If this file is a dependency manifest (Cargo.toml, package.json, etc.),
    // parse and cache the project's dependencies for Jaccard grouping.
    dependency_ingest::maybe_store_dependencies(pool, &item.tenant_id, file_path, abs_file_path)
        .await;

    record_pipeline_timings(pool, item, detected_language, timings).await;
    info!(
        "Successfully processed file item {} ({})",
        item.queue_id,
        payload.file_path.as_str()
    );
}

/// Phases 2-5: embed chunks, Tier 2 tagging, extract keywords/graph, inject component.
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
    detected_branch: &str,
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
        detected_branch,
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

    // Phase 2b: Tier 2 taxonomy tagging (after embedding, before keyword extraction)
    run_tier2_tagging(ctx, &mut points, timings).await;

    // Phases 3-4: keyword extraction + graph edges
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

/// Tier 2 taxonomy-based tagging: compute aggregate embedding from chunk
/// dense vectors, classify against taxonomy terms, inject matching labels
/// into all point payloads.
///
/// Performance target: <30ms per file (cosine sim against ~180 terms is O(n*d)).
async fn run_tier2_tagging(
    ctx: &ProcessingContext,
    points: &mut [crate::storage::DocumentPoint],
    timings: &mut Vec<PhaseTiming>,
) {
    let tagger = match &ctx.tier2_tagger {
        Some(t) => t,
        None => return,
    };

    if points.is_empty() {
        return;
    }

    let t0 = Instant::now();

    // Collect chunk dense vectors for aggregation
    let chunk_embeddings: Vec<Vec<f32>> = points.iter().map(|p| p.dense_vector.clone()).collect();

    // Compute the aggregate (mean) embedding for the whole document
    let agg_embedding = match aggregate_document_embedding(&chunk_embeddings) {
        Some(emb) => emb,
        None => {
            debug!("Tier 2: no aggregate embedding (empty chunks), skipping");
            return;
        }
    };

    // Classify against taxonomy
    let matches = tagger.classify(&agg_embedding);

    if matches.is_empty() {
        timings.push(PhaseTiming {
            phase: "tier2",
            duration_ms: t0.elapsed().as_millis() as u64,
        });
        debug!("Tier 2: no taxonomy matches above threshold");
        return;
    }

    // Collect taxonomy tag labels (category names with tax: prefix)
    let taxonomy_tags: Vec<String> = matches
        .iter()
        .map(|m| format!("tax:{}", m.category))
        .collect();

    // Inject taxonomy_tags into every point payload and append to existing tags
    for point in points.iter_mut() {
        // Add dedicated taxonomy_tags field
        point.payload.insert(
            "taxonomy_tags".to_string(),
            serde_json::json!(taxonomy_tags),
        );

        // Also append to the general "tags" array for unified search filtering
        if let Some(existing_tags) = point.payload.get_mut("tags") {
            if let Some(arr) = existing_tags.as_array_mut() {
                for tag in &taxonomy_tags {
                    let tag_val = serde_json::json!(tag);
                    if !arr.contains(&tag_val) {
                        arr.push(tag_val);
                    }
                }
            }
        } else {
            point
                .payload
                .insert("tags".to_string(), serde_json::json!(taxonomy_tags));
        }
    }

    let elapsed = t0.elapsed();
    timings.push(PhaseTiming {
        phase: "tier2",
        duration_ms: elapsed.as_millis() as u64,
    });

    info!(
        "Tier 2 tagging: {} matches in {}ms (categories: {})",
        matches.len(),
        elapsed.as_millis(),
        taxonomy_tags.join(", ")
    );
}

/// Parse the document and compute file identifiers (phase 0 + 1).
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
    let base_point =
        wqm_common::hashing::compute_base_point(&item.tenant_id, relative_path, &file_hash);

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
    graph_ingest::ingest_graph_edges(
        ctx,
        &item.tenant_id,
        relative_path,
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
    detected_branch: &str,
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
        detected_branch,
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
    detected_branch: &str,
) {
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "search", DestinationStatus::InProgress)
        .await;
    let t0 = Instant::now();
    let mut fts_ok = true;
    if let Some(sdb) = &ctx.search_db {
        match fts5_index::update_fts5_for_file(
            sdb,
            pool,
            file_id,
            abs_file_path,
            &item.tenant_id,
            Some(detected_branch),
            Some(base_point),
            Some(relative_path),
            Some(file_hash),
        )
        .await
        {
            Ok(_) => {}
            Err(e) => {
                warn!(
                    "FTS5 indexing failed for {} — search_status set to Failed: {}",
                    relative_path, e
                );
                fts_ok = false;
            }
        }
    }
    timings.push(PhaseTiming {
        phase: "fts5",
        duration_ms: t0.elapsed().as_millis() as u64,
    });
    let search_status = if fts_ok {
        DestinationStatus::Done
    } else {
        DestinationStatus::Failed
    };
    let _ = ctx
        .queue_manager
        .update_destination_status(&item.queue_id, "search", search_status)
        .await;
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
