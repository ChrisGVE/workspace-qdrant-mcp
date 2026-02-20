//! Keyword and tag extraction pipeline.
//!
//! Runs the 8-stage keyword/tag extraction pipeline after chunk embeddings
//! are generated. Results are injected into point payloads before Qdrant upsert.

use std::collections::HashMap;
use std::path::Path;

use tracing::{info, warn};

use crate::context::ProcessingContext;
use crate::keyword_extraction::collection_config;
use crate::keyword_extraction::pipeline::PipelineInput;
use crate::storage::DocumentPoint;
use crate::unified_queue_schema::UnifiedQueueItem;

/// Run keyword/tag extraction pipeline and inject results into point payloads.
pub(super) async fn run_keyword_extraction(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    file_path: &Path,
    document_content: &crate::DocumentContent,
    points: &mut [DocumentPoint],
) {
    let chunk_vectors: Vec<Vec<f32>> =
        points.iter().map(|p| p.dense_vector.clone()).collect();
    let chunk_texts: Vec<String> = document_content
        .chunks
        .iter()
        .map(|c| c.content.clone())
        .collect();
    let is_code = document_content.document_type.is_code();
    let language = document_content.document_type.language();

    // Fetch corpus size and build DF lookup from lexicon (Task 17)
    let corpus_size = ctx
        .lexicon_manager
        .corpus_size(&item.collection)
        .await;
    let full_text = chunk_texts.join("\n");

    // Build per-document DF lookup for unique terms in this document
    let unique_terms: std::collections::HashSet<String> = full_text
        .split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| w.len() >= 2)
        .collect();
    let mut df_lookup = HashMap::new();
    for term in &unique_terms {
        let df = ctx
            .lexicon_manager
            .document_frequency(&item.collection, term)
            .await;
        if df > 0 {
            df_lookup.insert(term.clone(), df as u64);
        }
    }

    let pipeline_input = PipelineInput {
        file_path,
        full_text: &full_text,
        language,
        is_code,
        chunk_vectors: &chunk_vectors,
        chunk_texts: &chunk_texts,
        corpus_size,
        df_lookup: &df_lookup,
    };

    let extraction_start = std::time::Instant::now();
    let pipeline_config = collection_config::config_for_collection(&item.collection);
    let extraction = crate::keyword_extraction::pipeline::run_pipeline(
        &pipeline_input,
        &ctx.embedding_generator,
        &pipeline_config,
    )
    .await;

    let extraction_ms = extraction_start.elapsed().as_millis();
    info!(
        "Keyword/tag extraction completed in {}ms: {} keywords, {} tags, {} structural tags (corpus_size={})",
        extraction_ms,
        extraction.keywords.len(),
        extraction.tags.len(),
        extraction.structural_tags.len(),
        corpus_size,
    );

    // Update lexicon with this document's terms (Task 17)
    let tokens: Vec<String> = unique_terms.into_iter().collect();
    if let Err(e) = ctx
        .lexicon_manager
        .add_document(&item.collection, &tokens)
        .await
    {
        warn!(
            "Failed to update lexicon for {}: {}",
            item.collection, e
        );
    }

    // Inject extraction results into all point payloads
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
            point.payload.insert(
                "concept_tags".to_string(),
                serde_json::json!(tag_phrases),
            );
        }
        if !struct_map.is_empty() {
            point.payload.insert(
                "structural_tags".to_string(),
                serde_json::json!(struct_map),
            );
        }
        if !basket_map.is_empty() {
            point.payload.insert(
                "keyword_baskets".to_string(),
                serde_json::json!(basket_map),
            );
        }
    }
}
