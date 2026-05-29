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

    // === BRANCH DISCOVERY CHECK ===
    // If this is the first file event for an unknown branch, run discovery
    // to populate existing content before processing individual files.
    super::discovery_trigger::check_and_run_discovery(
        ctx,
        pool,
        watch_folder_id,
        &item.tenant_id,
        base_path,
        &detected_branch,
    )
    .await;

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

    // === CONTENT-HASH DEDUP CHECK ===
    // If identical content exists at same path for another branch, skip embedding.
    if let Some(()) = super::dedup::try_dedup(
        ctx,
        item,
        pool,
        file_path,
        watch_folder_id,
        relative_path,
        abs_file_path,
        &detected_branch,
    )
    .await?
    {
        return Ok(());
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

    // Phase 2c: IMPLEMENTS_CONCEPT edges (after tagging, before graph extraction)
    let taxonomy_tags: Vec<String> = points
        .first()
        .and_then(|p| p.payload.get("taxonomy_tags"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    if !taxonomy_tags.is_empty() {
        create_concept_edges(ctx, &item.tenant_id, file_path, &taxonomy_tags).await;
    }

    // Phase 2d: COVERS_TOPIC edges for narrative files (markdown/text only)
    if !taxonomy_tags.is_empty() && is_narrative_file(file_path) {
        create_covers_topic_edges(
            ctx,
            &item.tenant_id,
            file_path,
            &taxonomy_tags,
            &document_content.raw_text,
        )
        .await;
    }

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
        detected_branch,
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

/// Create IMPLEMENTS_CONCEPT edges from code symbols to ConceptNodes.
///
/// For each Tier 2 taxonomy match, ensures a global ConceptNode exists
/// and creates edges from file-level code symbols to it.
async fn create_concept_edges(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &std::path::Path,
    taxonomy_tags: &[String],
) {
    use crate::graph::{
        compute_node_id_for_type, EdgeType, GraphEdge, GraphNode, NodeIdFields, NodeType,
    };

    let Some(ref graph_store) = ctx.graph_store else {
        return;
    };

    if taxonomy_tags.is_empty() {
        return;
    }

    let mut concept_nodes = Vec::new();
    let mut edges = Vec::new();
    let file_path_str: String = file_path.to_string_lossy().into_owned();

    for tag in taxonomy_tags {
        let concept_label = tag.strip_prefix("tax:").unwrap_or(tag);
        let concept_fields =
            NodeIdFields::new("__global__", "", concept_label, NodeType::ConceptNode);
        let concept_id = compute_node_id_for_type(&concept_fields);

        let mut node = GraphNode::new("__global__", "", concept_label, NodeType::ConceptNode);
        node.node_id = concept_id.clone();
        concept_nodes.push(node);

        let file_node_id = crate::graph::compute_node_id(
            tenant_id,
            &file_path_str,
            &file_path_str,
            NodeType::File,
        );

        edges.push(GraphEdge::new(
            tenant_id,
            &file_node_id,
            &concept_id,
            EdgeType::ImplementsConcept,
            &file_path_str,
        ));
    }

    if let Err(e) = graph_store.upsert_nodes(&concept_nodes).await {
        tracing::warn!("Failed to upsert ConceptNodes: {e}");
    }
    if let Err(e) = graph_store.insert_edges(&edges).await {
        tracing::warn!("Failed to insert IMPLEMENTS_CONCEPT edges: {e}");
    }
}

/// Check whether a file is a narrative document (markdown or plain text).
///
/// Only these file types get COVERS_TOPIC edges with depth metadata,
/// because depth estimation requires prose content structure.
fn is_narrative_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map_or(false, |ext| {
            let lower = ext.to_ascii_lowercase();
            lower == "md" || lower == "markdown" || lower == "txt" || lower == "rst"
        })
}

/// Create COVERS_TOPIC edges from a narrative file's node to ConceptNodes.
///
/// For each taxonomy tag, ensures a global ConceptNode exists and creates
/// a COVERS_TOPIC edge with depth metadata estimated from the file content.
/// Only called for narrative files (markdown, text, rst).
async fn create_covers_topic_edges(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &std::path::Path,
    taxonomy_tags: &[String],
    content: &str,
) {
    use crate::graph::{
        compute_node_id_for_type, EdgeType, GraphEdge, GraphNode, NodeIdFields, NodeType,
    };
    use crate::narrative::depth::estimate_depth;

    let Some(ref graph_store) = ctx.graph_store else {
        return;
    };

    if taxonomy_tags.is_empty() {
        return;
    }

    // Estimate depth for the full file (heading_level=0, no subsections flag
    // since we treat the whole file as a single narrative unit here).
    let depth_level = estimate_depth(content, 0, false);

    let mut concept_nodes = Vec::new();
    let mut edges = Vec::new();
    let file_path_str: String = file_path.to_string_lossy().into_owned();

    let file_node_id =
        crate::graph::compute_node_id(tenant_id, &file_path_str, &file_path_str, NodeType::File);

    for tag in taxonomy_tags {
        let concept_label = tag.strip_prefix("tax:").unwrap_or(tag);
        let concept_fields =
            NodeIdFields::new("__global__", "", concept_label, NodeType::ConceptNode);
        let concept_id = compute_node_id_for_type(&concept_fields);

        let mut node = GraphNode::new("__global__", "", concept_label, NodeType::ConceptNode);
        node.node_id = concept_id.clone();
        concept_nodes.push(node);

        edges.push(
            GraphEdge::new(
                tenant_id,
                &file_node_id,
                &concept_id,
                EdgeType::CoversTopic,
                &file_path_str,
            )
            .with_depth(depth_level),
        );
    }

    if let Err(e) = graph_store.upsert_nodes(&concept_nodes).await {
        tracing::warn!("Failed to upsert ConceptNodes for COVERS_TOPIC: {e}");
    }
    if let Err(e) = graph_store.insert_edges(&edges).await {
        tracing::warn!("Failed to insert COVERS_TOPIC edges: {e}");
    }

    debug!(
        "Created {} COVERS_TOPIC edges with depth={} for {}",
        edges.len(),
        depth_level,
        file_path_str,
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
    detected_branch: &str,
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
        Some(detected_branch),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::graph::{
        compute_edge_id, compute_node_id, compute_node_id_for_type, DepthLevel, EdgeType,
        GraphEdge, NodeIdFields, NodeType,
    };

    // ── is_narrative_file ──────────────────────────────────────────

    #[test]
    fn narrative_file_markdown() {
        assert!(is_narrative_file(Path::new("docs/README.md")));
        assert!(is_narrative_file(Path::new("notes.markdown")));
    }

    #[test]
    fn narrative_file_txt() {
        assert!(is_narrative_file(Path::new("todo.txt")));
        assert!(is_narrative_file(Path::new("/abs/path/NOTES.TXT")));
    }

    #[test]
    fn narrative_file_rst() {
        assert!(is_narrative_file(Path::new("docs/index.rst")));
    }

    #[test]
    fn narrative_file_code_is_not_narrative() {
        assert!(!is_narrative_file(Path::new("main.rs")));
        assert!(!is_narrative_file(Path::new("lib.py")));
        assert!(!is_narrative_file(Path::new("index.ts")));
        assert!(!is_narrative_file(Path::new("Cargo.toml")));
    }

    #[test]
    fn narrative_file_no_extension() {
        assert!(!is_narrative_file(Path::new("Makefile")));
        assert!(!is_narrative_file(Path::new("LICENSE")));
    }

    // ── COVERS_TOPIC edge construction ─────────────────────────────

    #[test]
    fn covers_topic_edge_has_depth_metadata() {
        let tenant = "t1";
        let file_path = "docs/guide.md";
        let concept_label = "machine-learning";

        let concept_fields =
            NodeIdFields::new("__global__", "", concept_label, NodeType::ConceptNode);
        let concept_id = compute_node_id_for_type(&concept_fields);

        let file_node_id = compute_node_id(tenant, file_path, file_path, NodeType::File);

        let edge = GraphEdge::new(
            tenant,
            &file_node_id,
            &concept_id,
            EdgeType::CoversTopic,
            file_path,
        )
        .with_depth(DepthLevel::Intermediate);

        assert_eq!(edge.edge_type, EdgeType::CoversTopic);
        assert_eq!(edge.depth_level(), Some(DepthLevel::Intermediate));
        assert_eq!(edge.source_node_id, file_node_id);
        assert_eq!(edge.target_node_id, concept_id);
        assert!(edge.metadata_json.is_some());
    }

    #[test]
    fn covers_topic_edge_id_differs_from_implements_concept() {
        let tenant = "t1";
        let file_path = "docs/guide.md";
        let concept_label = "testing";

        let concept_fields =
            NodeIdFields::new("__global__", "", concept_label, NodeType::ConceptNode);
        let concept_id = compute_node_id_for_type(&concept_fields);
        let file_node_id = compute_node_id(tenant, file_path, file_path, NodeType::File);

        let covers_id = compute_edge_id(&file_node_id, &concept_id, EdgeType::CoversTopic);
        let implements_id =
            compute_edge_id(&file_node_id, &concept_id, EdgeType::ImplementsConcept);

        assert_ne!(
            covers_id, implements_id,
            "COVERS_TOPIC and IMPLEMENTS_CONCEPT edges must have distinct IDs"
        );
    }

    #[test]
    fn covers_topic_depth_varies_with_content() {
        use crate::narrative::depth::estimate_depth;

        // Short content -> Reference
        let short = "API ref";
        let short_depth = estimate_depth(short, 0, false);
        assert_eq!(short_depth, DepthLevel::Reference);

        // Long technical content -> Rigorous
        let long_technical = "word ".repeat(2500);
        let long_depth = estimate_depth(&long_technical, 0, false);
        assert_eq!(long_depth, DepthLevel::Rigorous);
    }

    #[test]
    fn non_narrative_file_skipped_by_guard() {
        // Verify the is_narrative_file guard prevents COVERS_TOPIC for code files
        let rust_file = PathBuf::from("src/lib.rs");
        let tags = vec!["tax:systems-programming".to_string()];

        assert!(!is_narrative_file(&rust_file));
        // The guard `!taxonomy_tags.is_empty() && is_narrative_file(file_path)`
        // would be false, so no COVERS_TOPIC edges are created for .rs files.
        assert!(!tags.is_empty() && !is_narrative_file(&rust_file));
    }

    #[test]
    fn markdown_file_with_tags_passes_guard() {
        let md_file = PathBuf::from("docs/architecture.md");
        let tags = vec!["tax:architecture".to_string()];

        assert!(is_narrative_file(&md_file));
        assert!(!tags.is_empty() && is_narrative_file(&md_file));
    }

    #[test]
    fn covers_topic_multiple_tags_create_multiple_edges() {
        let tenant = "t1";
        let file_path = "guide.md";
        let tags = vec![
            "tax:testing".to_string(),
            "tax:ci-cd".to_string(),
            "tax:devops".to_string(),
        ];

        let file_node_id = compute_node_id(tenant, file_path, file_path, NodeType::File);

        let edges: Vec<GraphEdge> = tags
            .iter()
            .map(|tag| {
                let label = tag.strip_prefix("tax:").unwrap_or(tag);
                let fields = NodeIdFields::new("__global__", "", label, NodeType::ConceptNode);
                let concept_id = compute_node_id_for_type(&fields);
                GraphEdge::new(
                    tenant,
                    &file_node_id,
                    &concept_id,
                    EdgeType::CoversTopic,
                    file_path,
                )
                .with_depth(DepthLevel::Introductory)
            })
            .collect();

        assert_eq!(edges.len(), 3);
        // Each edge targets a different concept node
        let target_ids: std::collections::HashSet<&str> =
            edges.iter().map(|e| e.target_node_id.as_str()).collect();
        assert_eq!(target_ids.len(), 3, "each tag produces a unique target");
        // All edges have depth metadata
        for edge in &edges {
            assert_eq!(edge.depth_level(), Some(DepthLevel::Introductory));
        }
    }
}
