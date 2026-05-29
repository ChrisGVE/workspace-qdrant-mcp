//! File content ingestion pipeline — shared by add and update paths.
//! Orchestrates: document parsing -> embedding -> Tier 2 taxonomy tagging ->
//! keyword extraction -> graph extraction -> component detection -> Qdrant
//! upsert -> FTS5 indexing -> dependency extraction.

use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::Instant;

use futures::future::FutureExt;

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
use super::narrative_phase;
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
        detected_language,
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
    detected_language: Option<&'static str>,
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

    // Phase 2c: build symbol-granular IMPLEMENTS_CONCEPT nodes/edges. These are
    // NOT written here; they are merged into the Phase 4 graph `reingest_file`
    // transaction so the single delete-then-insert covers code + concept edges
    // (using relative_path as source_file → cleaned up on re-ingestion).
    let (concept_nodes, concept_edges) = match &ctx.tier2_tagger {
        Some(tagger) => build_implements_concept_edges(
            tagger,
            &ctx.concept_config,
            &item.tenant_id,
            relative_path,
            detected_branch,
            &points,
            &chunk_records,
        ),
        None => (Vec::new(), Vec::new()),
    };

    // Phase 4b: narrative extraction. Failure-isolated — any error yields an
    // empty result and a warning, never aborting ingestion (AC-A9). Its output
    // is threaded into the SAME graph reingest transaction as the code and
    // concept layers (a separate write would delete those edges).
    // Library-collection files emit LibrarySection (not DocumentSection) nodes,
    // scoped by library name (== tenant_id for the libraries collection).
    let library_name = if item.collection == wqm_common::constants::COLLECTION_LIBRARIES {
        Some(item.tenant_id.as_str())
    } else {
        None
    };
    let t_narr = Instant::now();
    let narrative_result = match AssertUnwindSafe(narrative_phase::run(
        ctx,
        &item.tenant_id,
        file_path,
        relative_path,
        &document_content.raw_text,
        detected_language,
        detected_branch,
        library_name,
        &points,
        &chunk_records,
    ))
    .catch_unwind()
    .await
    {
        Ok(result) => result,
        Err(_) => {
            warn!(
                "narrative_extraction_failed: panic during narrative extraction for {} (tenant {}); ingestion continues",
                relative_path, item.tenant_id
            );
            crate::narrative::NarrativeExtractionResult::default()
        }
    };
    timings.push(PhaseTiming {
        phase: "narrative",
        duration_ms: t_narr.elapsed().as_millis() as u64,
    });
    let (narrative_nodes, narrative_edges) = (narrative_result.nodes, narrative_result.edges);

    // Phases 3-4: keyword extraction + graph edges (concept + narrative
    // nodes/edges merged into the single graph reingest transaction).
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
        concept_nodes,
        concept_edges,
        narrative_nodes,
        narrative_edges,
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

/// Build symbol-granular IMPLEMENTS_CONCEPT edges with confidence weights.
///
/// For each code symbol chunk (function/method/class/struct/trait/…), classify
/// that symbol's OWN dense vector against the Tier-2 taxonomy and emit an edge
/// from the symbol node to each matching global ConceptNode, weighted by the
/// classification cosine. No extra embedding is performed — existing chunk
/// vectors are reused.
///
/// Returns `(concept_nodes, edges)` to be merged into the file's graph
/// `reingest_file` transaction (see [`graph_ingest::ingest_graph_edges`]), so
/// the single delete-then-insert covers code and concept edges together and
/// concept edges are cleaned up on re-ingestion.
///
/// Node IDs MUST match the graph extractor exactly: source = `compute_node_id(
/// tenant, relative_path, symbol_name, node_type)` where `node_type` is derived
/// from the chunk's `chunk_type` display string (matching
/// `extractor::node_type_from_display_name`). Using the lossy tracked
/// `ChunkType` would orphan async functions, constants, type aliases and macros.
fn build_implements_concept_edges(
    tagger: &crate::tagging::Tier2Tagger,
    concept_config: &crate::config::ConceptConfig,
    tenant_id: &str,
    relative_path: &str,
    detected_branch: &str,
    points: &[crate::storage::DocumentPoint],
    chunk_records: &[chunk_embed::ChunkRecord],
) -> (Vec<crate::graph::GraphNode>, Vec<crate::graph::GraphEdge>) {
    use crate::graph::{
        compute_node_id, compute_node_id_for_type, EdgeType, GraphEdge, GraphNode, NodeIdFields,
        NodeType,
    };
    use std::collections::HashMap;

    let min_confidence = concept_config.min_confidence;
    let max_per_unit = concept_config.max_per_unit;

    // Group chunk vectors by their symbol node id. Split sub-chunks share a
    // symbol_name + chunk_type, so they collapse onto a single node id; their
    // vectors are mean-aggregated below (no extra embedding).
    let mut groups: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    for (point, record) in points.iter().zip(chunk_records.iter()) {
        // Symbol name must be present and non-empty.
        let Some(symbol_name) = record.symbol_name.as_deref().filter(|s| !s.is_empty()) else {
            continue;
        };
        // Derive node_type from the display string carried in the payload
        // (build_chunk_payload prefixes chunk metadata with `chunk_`). This
        // matches the graph extractor exactly; the lossy tracked ChunkType
        // would orphan async functions / constants / type aliases / macros.
        let Some(chunk_type_str) = point
            .payload
            .get("chunk_chunk_type")
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        let Some(node_type) = crate::graph::extractor::node_type_from_display_name(chunk_type_str)
        else {
            continue; // preamble / text / unknown → not a symbol node
        };
        if point.dense_vector.is_empty() {
            continue;
        }

        let node_id = compute_node_id(tenant_id, relative_path, symbol_name, node_type);
        groups
            .entry(node_id)
            .or_default()
            .push(point.dense_vector.clone());
    }

    if groups.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut concept_nodes: HashMap<String, GraphNode> = HashMap::new();
    let mut edges = Vec::new();

    for (symbol_node_id, vectors) in &groups {
        let Some(symbol_embedding) = aggregate_document_embedding(vectors) else {
            continue;
        };

        // Classify this symbol's vector; apply the concept threshold + cap.
        let mut matches = tagger.classify(&symbol_embedding);
        matches.retain(|m| m.score >= min_confidence);
        matches.truncate(max_per_unit);

        for m in matches {
            let concept_label = m.category.as_str();
            let concept_fields =
                NodeIdFields::new("__global__", "", concept_label, NodeType::ConceptNode);
            let concept_id = compute_node_id_for_type(&concept_fields);

            concept_nodes.entry(concept_id.clone()).or_insert_with(|| {
                let mut node =
                    GraphNode::new("__global__", "", concept_label, NodeType::ConceptNode);
                node.node_id = concept_id.clone();
                node
            });

            let mut edge = GraphEdge::new(
                tenant_id,
                symbol_node_id.clone(),
                concept_id.clone(),
                EdgeType::ImplementsConcept,
                relative_path,
            )
            .with_branch(detected_branch);
            edge.weight = m.score;
            edges.push(edge);
        }
    }

    if !edges.is_empty() {
        debug!(
            "IMPLEMENTS_CONCEPT: {} symbol→concept edges across {} symbols, {} concepts for {}",
            edges.len(),
            groups.len(),
            concept_nodes.len(),
            relative_path,
        );
    }

    (concept_nodes.into_values().collect(), edges)
}

/// Check whether a file is a narrative document (markdown or plain text).
///
/// Only used by tests now; production narrative-file detection lives in
/// [`narrative_phase`].
#[cfg(test)]
fn is_narrative_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map_or(false, |ext| {
            let lower = ext.to_ascii_lowercase();
            lower == "md" || lower == "markdown" || lower == "txt" || lower == "rst"
        })
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
    concept_nodes: Vec<crate::graph::GraphNode>,
    concept_edges: Vec<crate::graph::GraphEdge>,
    narrative_nodes: Vec<crate::graph::GraphNode>,
    narrative_edges: Vec<crate::graph::GraphEdge>,
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
        concept_nodes,
        concept_edges,
        narrative_nodes,
        narrative_edges,
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

    // ── INT-E1: symbol-granular IMPLEMENTS_CONCEPT edges ───────────

    use crate::storage::DocumentPoint;
    use crate::tagging::{TaxonomyEntry, Tier2Config, Tier2Tagger};
    use std::collections::HashMap as StdHashMap;

    fn mk_point(chunk_type: &str, vec: Vec<f32>) -> DocumentPoint {
        let mut payload: StdHashMap<String, serde_json::Value> = StdHashMap::new();
        payload.insert(
            "chunk_chunk_type".to_string(),
            serde_json::json!(chunk_type),
        );
        DocumentPoint {
            id: "point".to_string(),
            dense_vector: vec,
            sparse_vector: None,
            payload,
        }
    }

    fn mk_record(symbol: &str) -> chunk_embed::ChunkRecord {
        chunk_embed::ChunkRecord {
            point_id: "point".to_string(),
            chunk_index: 0,
            content_hash: "hash".to_string(),
            chunk_type: None,
            symbol_name: Some(symbol.to_string()),
            start_line: Some(1),
            end_line: Some(2),
        }
    }

    /// Tagger with one term in category `machine-learning`, embedded as the
    /// unit vector `[1, 0, 0]` so cosine similarity is exactly computable.
    fn mk_tagger() -> Tier2Tagger {
        let entries = vec![TaxonomyEntry {
            term: "machine learning".to_string(),
            category: "machine-learning".to_string(),
        }];
        let embeddings = vec![vec![1.0_f32, 0.0, 0.0]];
        Tier2Tagger::from_precomputed(entries, embeddings, Tier2Config::default())
    }

    #[test]
    fn implements_concept_edge_source_is_symbol_not_file() {
        let tagger = mk_tagger();
        let cfg = crate::config::ConceptConfig::default();
        let tenant = "t1";
        let rel = "src/model.rs";

        // One matching symbol (cosine 1.0) and one non-matching (cosine 0.0).
        let points = vec![
            mk_point("function", vec![1.0, 0.0, 0.0]),
            mk_point("function", vec![0.0, 1.0, 0.0]),
        ];
        let records = vec![mk_record("train_model"), mk_record("unrelated")];

        let (nodes, edges) =
            build_implements_concept_edges(&tagger, &cfg, tenant, rel, "main", &points, &records);

        // Exactly one edge: only the matching symbol clears the threshold.
        assert_eq!(edges.len(), 1, "sub-threshold symbol must be absent");
        let edge = &edges[0];

        // Source is the symbol node, NOT the File node.
        let symbol_id = compute_node_id(tenant, rel, "train_model", NodeType::Function);
        let file_id = compute_node_id(tenant, rel, rel, NodeType::File);
        assert_eq!(edge.source_node_id, symbol_id);
        assert_ne!(edge.source_node_id, file_id);
        assert_eq!(edge.edge_type, EdgeType::ImplementsConcept);

        // Target is the global ConceptNode for the matched category.
        let concept_id = compute_node_id_for_type(&NodeIdFields::new(
            "__global__",
            "",
            "machine-learning",
            NodeType::ConceptNode,
        ));
        assert_eq!(edge.target_node_id, concept_id);

        // Weight is the symbol-chunk cosine (≈ 1.0), not a document mean.
        assert!(
            (edge.weight - 1.0).abs() < 1e-6,
            "weight should equal symbol cosine, got {}",
            edge.weight
        );

        // Branch propagated; ConceptNode emitted.
        assert_eq!(edge.branch.as_deref(), Some("main"));
        assert!(nodes.iter().any(|n| n.node_id == concept_id));
    }

    #[test]
    fn implements_concept_handles_async_function_display_string() {
        // Regression: tracked ChunkType has no AsyncFunction, so deriving the
        // node type from the display string (not the lossy tracked type) is
        // required or async-fn symbols would orphan.
        let tagger = mk_tagger();
        let cfg = crate::config::ConceptConfig::default();
        let tenant = "t1";
        let rel = "src/fetch.rs";

        let points = vec![mk_point("async_function", vec![1.0, 0.0, 0.0])];
        let records = vec![mk_record("fetch_data")];

        let (_nodes, edges) =
            build_implements_concept_edges(&tagger, &cfg, tenant, rel, "main", &points, &records);

        assert_eq!(edges.len(), 1);
        let expected = compute_node_id(tenant, rel, "fetch_data", NodeType::AsyncFunction);
        assert_eq!(edges[0].source_node_id, expected);
    }

    #[test]
    fn implements_concept_dedups_split_subchunks_per_symbol() {
        // Two points sharing symbol_name + chunk_type (e.g. an oversized symbol
        // split into sub-chunks) collapse onto one node → one edge per concept.
        let tagger = mk_tagger();
        let cfg = crate::config::ConceptConfig::default();
        let points = vec![
            mk_point("function", vec![1.0, 0.0, 0.0]),
            mk_point("function", vec![1.0, 0.0, 0.0]),
        ];
        let records = vec![mk_record("big_fn"), mk_record("big_fn")];

        let (_nodes, edges) = build_implements_concept_edges(
            &tagger, &cfg, "t1", "src/a.rs", "main", &points, &records,
        );
        assert_eq!(edges.len(), 1, "split sub-chunks must dedup to one edge");
    }

    #[test]
    fn implements_concept_skips_non_symbol_chunks() {
        let tagger = mk_tagger();
        let cfg = crate::config::ConceptConfig::default();
        // Text/preamble chunks and records without a symbol name yield nothing.
        let points = vec![
            mk_point("text", vec![1.0, 0.0, 0.0]),
            mk_point("preamble", vec![1.0, 0.0, 0.0]),
        ];
        let mut empty_symbol = mk_record("");
        empty_symbol.symbol_name = None;
        let records = vec![mk_record("ignored_text"), empty_symbol];

        let (nodes, edges) = build_implements_concept_edges(
            &tagger, &cfg, "t1", "src/a.rs", "main", &points, &records,
        );
        assert!(edges.is_empty());
        assert!(nodes.is_empty());
    }

    #[test]
    fn implements_concept_respects_max_per_unit_cap() {
        // Two taxonomy terms in distinct categories, both matching strongly.
        let entries = vec![
            TaxonomyEntry {
                term: "a".to_string(),
                category: "cat-a".to_string(),
            },
            TaxonomyEntry {
                term: "b".to_string(),
                category: "cat-b".to_string(),
            },
        ];
        let embeddings = vec![vec![1.0_f32, 0.0, 0.0], vec![0.9839_f32, 0.1789, 0.0]];
        let tagger = Tier2Tagger::from_precomputed(entries, embeddings, Tier2Config::default());
        let cfg = crate::config::ConceptConfig {
            min_confidence: 0.35,
            max_per_unit: 1,
        };

        let points = vec![mk_point("function", vec![1.0, 0.0, 0.0])];
        let records = vec![mk_record("f")];
        let (_n, edges) = build_implements_concept_edges(
            &tagger, &cfg, "t1", "src/a.rs", "main", &points, &records,
        );
        assert_eq!(
            edges.len(),
            1,
            "max_per_unit cap must bound edges per symbol"
        );
    }
}
