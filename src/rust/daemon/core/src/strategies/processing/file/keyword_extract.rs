//! Keyword and tag extraction pipeline.
//!
//! Runs the 8-stage keyword/tag extraction pipeline after chunk embeddings
//! are generated. Results are injected into point payloads before Qdrant upsert.

use std::path::Path;

use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::keyword_extraction::collection_config;
use crate::keyword_extraction::cooccurrence_graph;
use crate::keyword_extraction::pipeline::{ExtractionResult, PipelineInput};
use crate::storage::DocumentPoint;
use crate::unified_queue_schema::UnifiedQueueItem;

/// Run keyword/tag extraction pipeline and inject results into point payloads.
///
/// Returns the `ExtractionResult` for SQLite persistence by the caller.
pub(super) async fn run_keyword_extraction(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    file_path: &Path,
    document_content: &crate::DocumentContent,
    points: &mut [DocumentPoint],
) -> Option<ExtractionResult> {
    let chunk_vectors: Vec<Vec<f32>> = points.iter().map(|p| p.dense_vector.clone()).collect();
    let chunk_texts: Vec<String> = document_content
        .chunks
        .iter()
        .map(|c| c.content.clone())
        .collect();
    // Per-chunk metadata (`chunk_type`, `symbol_name`, `parent_symbol`, …)
    // is preserved across the SemanticChunk → TextChunk conversion in
    // `convert_semantic_chunks_to_text_chunks`. Feed it to the pipeline so
    // tree-sitter symbols can contribute concept-level candidate tags.
    let chunk_metadata: Vec<std::collections::HashMap<String, String>> = document_content
        .chunks
        .iter()
        .map(|c| c.metadata.clone())
        .collect();
    let is_code = document_content.document_type.is_code();
    let language = document_content.document_type.language();
    let full_text = chunk_texts.join("\n");

    let (corpus_size, df_lookup, unique_terms) =
        fetch_lexicon_data(ctx, &item.collection, &full_text).await;

    let centrality_scores = fetch_centrality_scores(ctx, item, is_code).await;

    let pipeline_input = PipelineInput {
        file_path,
        full_text: &full_text,
        language,
        is_code,
        chunk_vectors: &chunk_vectors,
        chunk_texts: &chunk_texts,
        chunk_metadata: Some(&chunk_metadata),
        corpus_size,
        df_lookup: &df_lookup,
        centrality_scores: centrality_scores.as_ref(),
    };

    let extraction_start = std::time::Instant::now();
    let pipeline_config = collection_config::config_for_collection(&item.collection);
    let keyword_gen = ctx.keyword_generator();
    let extraction = crate::keyword_extraction::pipeline::run_pipeline(
        &pipeline_input,
        keyword_gen,
        &pipeline_config,
    )
    .await;

    info!(
        "Keyword/tag extraction completed in {}ms: {} keywords, {} tags, {} structural tags (corpus_size={})",
        extraction_start.elapsed().as_millis(),
        extraction.keywords.len(),
        extraction.tags.len(),
        extraction.structural_tags.len(),
        corpus_size,
    );

    update_lexicon_and_graph(
        ctx,
        item,
        is_code,
        language,
        &full_text,
        &unique_terms,
        &pipeline_config,
    )
    .await;
    inject_extraction_results(&extraction, points);

    Some(extraction)
}

async fn fetch_lexicon_data(
    ctx: &ProcessingContext,
    collection: &str,
    full_text: &str,
) -> (
    u64,
    std::collections::HashMap<String, u64>,
    std::collections::HashSet<String>,
) {
    let corpus_size = ctx.lexicon_manager.corpus_size(collection).await;

    let unique_terms: std::collections::HashSet<String> = full_text
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| w.len() >= 2)
        .collect();

    let df_lookup = ctx
        .lexicon_manager
        .document_frequencies_batch(collection, &unique_terms)
        .await;

    (corpus_size, df_lookup, unique_terms)
}

async fn fetch_centrality_scores(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    is_code: bool,
) -> Option<std::collections::HashMap<String, f64>> {
    if !is_code {
        return None;
    }
    let mut cache = ctx.cooccurrence_cache.lock().await;
    match cache
        .get_or_compute(&ctx.pool, &item.tenant_id, &item.collection)
        .await
    {
        Ok(scores) if !scores.is_empty() => Some(scores),
        Ok(_) => None,
        Err(e) => {
            warn!("Failed to fetch centrality scores: {}", e);
            None
        }
    }
}

async fn update_lexicon_and_graph(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    is_code: bool,
    language: Option<&str>,
    full_text: &str,
    unique_terms: &std::collections::HashSet<String>,
    pipeline_config: &crate::keyword_extraction::pipeline::PipelineConfig,
) {
    let tokens: Vec<String> = unique_terms.iter().cloned().collect();
    if let Err(e) = ctx
        .lexicon_manager
        .add_document(&item.collection, &tokens)
        .await
    {
        warn!("Failed to update lexicon for {}: {}", item.collection, e);
    }

    if is_code {
        if let Some(lang) = language {
            let symbols =
                cooccurrence_graph::extract_symbols(full_text, lang, &pipeline_config.lsp);
            if symbols.len() >= 2 {
                if let Err(e) = cooccurrence_graph::update_graph(
                    &ctx.pool,
                    &item.tenant_id,
                    &item.collection,
                    &symbols,
                )
                .await
                {
                    warn!("Failed to update co-occurrence graph: {}", e);
                }
            }
        }
    }
}

fn inject_extraction_results(extraction: &ExtractionResult, points: &mut [DocumentPoint]) {
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
            point
                .payload
                .insert("concept_tags".to_string(), serde_json::json!(tag_phrases));
        }
        if !struct_map.is_empty() {
            point
                .payload
                .insert("structural_tags".to_string(), serde_json::json!(struct_map));
        }
        if !basket_map.is_empty() {
            point
                .payload
                .insert("keyword_baskets".to_string(), serde_json::json!(basket_map));
        }
    }
}
