//! Per-chunk embedding, payload construction, and LSP enrichment.
//!
//! Processes each chunk from the document processor: generates dense/sparse
//! embeddings, builds Qdrant payload metadata, applies LSP enrichment when
//! available, and returns assembled `DocumentPoint`s with chunk tracking records.

mod payload;
mod types;

pub(super) use types::{ChunkRecord, EmbedResult};

use std::path::Path;

use tracing::{debug, info};

use crate::context::ProcessingContext;
use crate::lsp::EnrichmentStatus;
use crate::tracked_files_schema::{self, ChunkType as TrackedChunkType, ProcessingStatus};
use crate::unified_queue_processor::UnifiedProcessorError;
use crate::unified_queue_schema::UnifiedQueueItem;
use crate::DocumentContent;

use super::lsp_payload;
use payload::{build_chunk_payload, sparse_embedding_to_map};

/// Embed all chunks from a document, building Qdrant points and chunk records.
///
/// For each chunk: generates dense + sparse embeddings, constructs the full
/// payload map, applies LSP enrichment when available, and tracks chunk metadata
/// for SQLite `qdrant_chunks`.
#[allow(clippy::too_many_arguments)]
pub(super) async fn embed_chunks(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    document_content: &DocumentContent,
    file_path: &Path,
    file_document_id: &str,
    relative_path: &str,
    base_point: &str,
    file_hash: &str,
    file_type: Option<&str>,
) -> Result<EmbedResult, UnifiedProcessorError> {
    // Check if LSP enrichment is available for this project
    let (is_project_active, lsp_mgr_guard) = if let Some(lsp_mgr) = &ctx.lsp_manager {
        let mgr = lsp_mgr.read().await;
        let is_active = mgr.has_active_servers(&item.tenant_id).await;
        if is_active {
            debug!(
                "LSP enrichment available for project {} on file {}",
                item.tenant_id,
                file_path.display()
            );
        }
        (is_active, Some(lsp_mgr.clone()))
    } else {
        (false, None)
    };

    let mut lsp_status = ProcessingStatus::None;
    let mut treesitter_status = ProcessingStatus::None;
    let mut points = Vec::new();
    let mut chunk_records = Vec::new();
    let embedding_start = std::time::Instant::now();

    for (chunk_idx, chunk) in document_content.chunks.iter().enumerate() {
        // Semaphore-gated embedding generation (Task 504)
        let _permit = ctx.embedding_semaphore.acquire().await.map_err(|e| {
            UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e))
        })?;
        let embedding_result = ctx
            .embedding_generator
            .generate_embedding(&chunk.content, "bge-small-en-v1.5")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
        drop(_permit);

        let mut point_payload = build_chunk_payload(
            &chunk.content,
            chunk.chunk_index,
            item,
            document_content,
            file_path,
            file_document_id,
            relative_path,
            base_point,
            file_hash,
            file_type,
            &chunk.metadata,
        );

        // Extract chunk metadata for tracked record
        let symbol_name = chunk.metadata.get("symbol_name").cloned();
        let start_line = chunk
            .metadata
            .get("start_line")
            .and_then(|s| s.parse::<i32>().ok());
        let end_line = chunk
            .metadata
            .get("end_line")
            .and_then(|s| s.parse::<i32>().ok());
        let chunk_type_str = chunk.metadata.get("chunk_type");
        let chunk_type = chunk_type_str.and_then(|s| TrackedChunkType::from_str(s));

        // Detect tree-sitter status from chunk metadata
        if chunk.metadata.contains_key("chunk_type") {
            treesitter_status = ProcessingStatus::Done;
        }

        // LSP enrichment (if available and file language has LSP support)
        if let Some(lsp_mgr) = &lsp_mgr_guard {
            let file_lang = file_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(crate::lsp::Language::from_extension);

            if file_lang
                .as_ref()
                .map_or(false, |l| l.has_lsp_support())
            {
                let mgr = lsp_mgr.read().await;

                let sym_name = chunk
                    .metadata
                    .get("symbol_name")
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");

                let sl = chunk
                    .metadata
                    .get("start_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(chunk_idx as u32 * 20);

                let el = chunk
                    .metadata
                    .get("end_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(sl + 20);

                let enrichment = mgr
                    .enrich_chunk(
                        &item.tenant_id,
                        file_path,
                        sym_name,
                        sl,
                        el,
                        is_project_active,
                    )
                    .await;

                if enrichment.enrichment_status == EnrichmentStatus::Skipped {
                    // LSP server not ready -- mark as pending for metadata_uplift retry
                    point_payload.insert(
                        "lsp_enrichment_status".to_string(),
                        serde_json::json!("pending"),
                    );
                    // Keep lsp_status as None (not Done) so tracked_files reflects incomplete state
                } else {
                    lsp_payload::add_lsp_enrichment_to_payload(
                        &mut point_payload,
                        &enrichment,
                    );
                    lsp_status = ProcessingStatus::Done;
                }
            } else {
                // Non-code file (markdown, config, etc.) -- skip LSP enrichment
                point_payload.insert(
                    "lsp_enrichment_status".to_string(),
                    serde_json::json!("skipped"),
                );
                lsp_status = ProcessingStatus::Skipped;
            }
        }

        let point_id =
            wqm_common::hashing::compute_point_id(base_point, chunk_idx as u32);
        let content_hash = tracked_files_schema::compute_content_hash(&chunk.content);

        // Generate sparse vector based on configured mode (bm25 or splade)
        let sparse = if ctx.embedding_generator.sparse_vector_mode() == "splade" {
            // SPLADE++ learned sparse vectors (Task 38)
            match ctx.embedding_generator.generate_splade_sparse_vector(&chunk.content).await {
                Ok(s) => s,
                Err(e) => {
                    debug!("SPLADE++ fallback to BM25: {}", e);
                    embedding_result.sparse.clone()
                }
            }
        } else {
            // BM25 with lexicon-backed IDF (Task 19)
            let chunk_tokens: Vec<String> = chunk
                .content
                .split_whitespace()
                .map(|s| s.to_lowercase())
                .collect();
            let lexicon_sparse = ctx
                .lexicon_manager
                .generate_sparse_vector(&item.collection, &chunk_tokens)
                .await;
            // Fall back to embedding generator's ephemeral BM25 if lexicon has no corpus stats
            if !lexicon_sparse.indices.is_empty() {
                lexicon_sparse
            } else {
                embedding_result.sparse.clone()
            }
        };

        let point = crate::storage::DocumentPoint {
            id: point_id.clone(),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: sparse_embedding_to_map(&sparse),
            payload: point_payload,
        };

        points.push(point);
        chunk_records.push(ChunkRecord {
            point_id,
            chunk_index: chunk_idx as i32,
            content_hash,
            chunk_type,
            symbol_name,
            start_line,
            end_line,
        });
    }

    info!(
        "Embedding generation completed: {} chunks in {}ms",
        chunk_records.len(),
        embedding_start.elapsed().as_millis()
    );

    Ok(EmbedResult {
        points,
        chunk_records,
        lsp_status,
        treesitter_status,
    })
}

#[cfg(test)]
mod tests;
