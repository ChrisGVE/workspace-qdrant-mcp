//! Per-chunk embedding, payload construction, and LSP enrichment.
//!
//! Processes each chunk from the document processor: generates dense/sparse
//! embeddings, builds Qdrant payload metadata, applies LSP enrichment when
//! available, and returns assembled `DocumentPoint`s with chunk tracking records.

mod payload;
mod types;

pub(super) use types::{ChunkRecord, EmbedResult};

use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};

use crate::context::ProcessingContext;
use crate::lsp::EnrichmentStatus;
use crate::tracing_gate::{tier_enabled, TraceTier};
use crate::tracked_files_schema::{self, ChunkType as TrackedChunkType, ProcessingStatus};
use crate::unified_queue_processor::UnifiedProcessorError;
use crate::unified_queue_schema::UnifiedQueueItem;
use crate::DocumentContent;

use super::lsp_payload;
use payload::{build_chunk_payload, sparse_embedding_to_map};

/// Result of processing a single chunk, to be assembled into `EmbedResult`.
struct ChunkOutput {
    point: crate::storage::DocumentPoint,
    record: ChunkRecord,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
}

/// Embed all chunks from a document, building Qdrant points and chunk records.
///
/// Dense embeddings are batched into a single provider call (the provider
/// handles sub-chunking by `remote_batch_size` internally). Per-chunk work
/// (sparse vectors, LSP enrichment, payload construction) runs sequentially
/// after the batch returns.
///
/// `branch` is the detected current branch (from `BranchCache`), used for
/// the `"branches"` Qdrant payload field.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    name = "embed.generate",
    level = "debug",
    skip_all,
    fields(
        wqm.engine = tracing::field::Empty,
        model = tracing::field::Empty,
        batch.size = tracing::field::Empty,
    )
)]
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
    branch: &str,
) -> Result<EmbedResult, UnifiedProcessorError> {
    if document_content.chunks.is_empty() {
        debug!(
            "No chunks to embed for {} — returning empty EmbedResult",
            file_path.display()
        );
        return Ok(EmbedResult {
            points: Vec::new(),
            chunk_records: Vec::new(),
            lsp_status: ProcessingStatus::Skipped,
            treesitter_status: ProcessingStatus::Skipped,
        });
    }

    let (is_project_active, lsp_mgr_guard) = check_lsp_availability(ctx, item, file_path).await;

    let embedding_start = std::time::Instant::now();
    let idf_epoch = ctx.lexicon_manager.corpus_size(&item.collection).await;

    // Split any chunk that exceeds the active provider's input budget so a
    // single overlong chunk never triggers an HTTP 400 from the remote
    // embedder. The budget is bytes: byte-level BPE guarantees the token
    // count never exceeds the byte count, so a byte cap bounds tokens with
    // no tokenizer dependency. No-op for providers without a caller-side
    // limit (FastEmbed).
    let max_input_bytes = ctx.embedding_generator.max_input_bytes();
    let chunks = split_oversized_chunks(&document_content.chunks, max_input_bytes);
    if chunks.len() != document_content.chunks.len() {
        info!(
            "Split {} oversized chunk(s) into {} sub-chunks (budget {} bytes) for {}",
            chunks.len() - document_content.chunks.len(),
            chunks.len(),
            max_input_bytes,
            file_path.display()
        );
    }

    // Batch-embed all chunk texts in one provider call.
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

    if tier_enabled(TraceTier::Hot) {
        let span = tracing::Span::current();
        span.record("wqm.engine", ctx.embedding_generator.metrics_label());
        span.record("model", "default");
        span.record("batch.size", chunk_texts.len());
    }

    let _permit = ctx
        .embedding_semaphore
        .acquire()
        .await
        .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;

    let batch_results = ctx
        .embedding_generator
        .generate_embeddings_batch(&chunk_texts, "default")
        .await
        .map_err(UnifiedProcessorError::from)?;

    drop(_permit);

    // Per-chunk: build payload, sparse vectors, LSP enrichment, assemble point.
    let mut lsp_status = ProcessingStatus::None;
    let mut treesitter_status = ProcessingStatus::None;
    let mut points = Vec::new();
    let mut chunk_records = Vec::new();

    for (chunk_idx, (chunk, embedding_result)) in chunks.iter().zip(batch_results).enumerate() {
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
            None, // library_ctx
            branch,
        );

        let (symbol_name, start_line, end_line, chunk_type) = extract_chunk_metadata(chunk);

        let ts_status = if chunk.metadata.contains_key("chunk_type") {
            ProcessingStatus::Done
        } else {
            ProcessingStatus::None
        };

        let chunk_lsp_status = if let Some(lsp_mgr) = &lsp_mgr_guard {
            apply_lsp_enrichment(
                item,
                file_path,
                chunk_idx,
                &chunk.metadata,
                is_project_active,
                lsp_mgr,
                &mut point_payload,
            )
            .await
        } else {
            ProcessingStatus::None
        };

        let output = assemble_chunk_output(
            ctx,
            item,
            chunk_idx,
            &chunk.content,
            embedding_result,
            point_payload,
            chunk_type,
            symbol_name,
            start_line,
            end_line,
            chunk_lsp_status,
            ts_status,
            base_point,
            idf_epoch,
        )
        .await;

        if output.lsp_status != ProcessingStatus::None {
            lsp_status = output.lsp_status;
        }
        if output.treesitter_status != ProcessingStatus::None {
            treesitter_status = output.treesitter_status;
        }
        points.push(output.point);
        chunk_records.push(output.record);
    }

    let embed_ms = embedding_start.elapsed().as_millis();

    // Route the stage-3 batch embed latency to the metrics switchboard: telemetry
    // (Prometheus) plus the always-on queue-health control lane that drives the
    // EWMA verdict. embed_ms and source size are co-measured for one batch. The
    // handle is resolved inline per batch — `metrics_label()` is a `&'static str`,
    // so this is one OnceCell load + a two-field struct, off the per-chunk loop.
    // (This is the LIVE production embedding path; the arch doc's original anchor
    // `IngestionEngine::stage3_embed_chunks` is never constructed in the daemon.)
    if let Some(sw) = crate::switchboard::switchboard() {
        let source_bytes = chunk_texts.iter().map(|s| s.len()).sum::<usize>();
        let handle = sw.handle(
            crate::switchboard::MetricId::EmbedderLatency,
            ctx.embedding_generator.metrics_label(),
        );
        sw.emit_record(
            handle,
            crate::switchboard::EmbedLatencyRec {
                embed_ms,
                source_bytes,
            },
        );
    }

    info!(
        "Embedding generation completed: {} chunks in {}ms",
        chunk_records.len(),
        embed_ms
    );

    Ok(EmbedResult {
        points,
        chunk_records,
        lsp_status,
        treesitter_status,
    })
}

/// Check whether LSP enrichment is available for the project, returning
/// `(is_project_active, lsp_manager_guard)`.
async fn check_lsp_availability(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    file_path: &Path,
) -> (
    bool,
    Option<std::sync::Arc<tokio::sync::RwLock<crate::lsp::LanguageServerManager>>>,
) {
    if let Some(lsp_mgr) = &ctx.lsp_manager {
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
    }
}

/// Generate the sparse vector for a chunk, returning the vector and whether
/// lexicon-backed BM25 was used (for idf_epoch tagging).
async fn generate_sparse(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    chunk_content: &str,
    fallback_sparse: &crate::embedding::SparseEmbedding,
) -> (crate::embedding::SparseEmbedding, bool) {
    if ctx.embedding_generator.sparse_vector_mode() == "splade" {
        // SPLADE++ learned sparse vectors (Task 38)
        match ctx
            .embedding_generator
            .generate_splade_sparse_vector(chunk_content)
            .await
        {
            Ok(s) => (s, false),
            Err(e) => {
                debug!("SPLADE++ fallback to BM25: {}", e);
                (fallback_sparse.clone(), false)
            }
        }
    } else {
        // BM25 with lexicon-backed IDF (Task 19)
        let chunk_tokens: Vec<String> = chunk_content
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let lexicon_sparse = ctx
            .lexicon_manager
            .generate_sparse_vector(&item.collection, &chunk_tokens)
            .await;
        if !lexicon_sparse.indices.is_empty() {
            (lexicon_sparse, true)
        } else {
            (fallback_sparse.clone(), false)
        }
    }
}

/// Apply LSP enrichment to `point_payload` for a single chunk, returning the
/// updated LSP processing status.
async fn apply_lsp_enrichment(
    item: &UnifiedQueueItem,
    file_path: &Path,
    chunk_idx: usize,
    chunk_metadata: &HashMap<String, String>,
    is_project_active: bool,
    lsp_mgr_guard: &std::sync::Arc<tokio::sync::RwLock<crate::lsp::LanguageServerManager>>,
    point_payload: &mut HashMap<String, serde_json::Value>,
) -> ProcessingStatus {
    let file_lang = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(crate::lsp::Language::from_extension);

    if file_lang.as_ref().map_or(false, |l| l.has_lsp_support()) {
        let mgr = lsp_mgr_guard.read().await;

        let sym_name = chunk_metadata
            .get("symbol_name")
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        let sl = chunk_metadata
            .get("start_line")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(chunk_idx as u32 * 20);

        let el = chunk_metadata
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
            // Keep lsp_status as None so tracked_files reflects incomplete state
            ProcessingStatus::None
        } else {
            lsp_payload::add_lsp_enrichment_to_payload(point_payload, &enrichment);
            ProcessingStatus::Done
        }
    } else {
        // Non-code file (markdown, config, etc.) -- skip LSP enrichment
        point_payload.insert(
            "lsp_enrichment_status".to_string(),
            serde_json::json!("skipped"),
        );
        ProcessingStatus::Skipped
    }
}

/// Extract chunk metadata fields for the tracking record.
fn extract_chunk_metadata(
    chunk: &crate::TextChunk,
) -> (
    Option<String>,
    Option<i32>,
    Option<i32>,
    Option<TrackedChunkType>,
) {
    let symbol_name = chunk.metadata.get("symbol_name").cloned();
    let start_line = chunk
        .metadata
        .get("start_line")
        .and_then(|s| s.parse::<i32>().ok());
    let end_line = chunk
        .metadata
        .get("end_line")
        .and_then(|s| s.parse::<i32>().ok());
    let chunk_type = chunk
        .metadata
        .get("chunk_type")
        .and_then(|s| TrackedChunkType::from_str(s));
    (symbol_name, start_line, end_line, chunk_type)
}

#[allow(clippy::too_many_arguments)]
async fn assemble_chunk_output(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    chunk_idx: usize,
    content: &str,
    embedding_result: crate::embedding::EmbeddingResult,
    mut point_payload: std::collections::HashMap<String, serde_json::Value>,
    chunk_type: Option<TrackedChunkType>,
    symbol_name: Option<String>,
    start_line: Option<i32>,
    end_line: Option<i32>,
    lsp_status: ProcessingStatus,
    treesitter_status: ProcessingStatus,
    base_point: &str,
    idf_epoch: u64,
) -> ChunkOutput {
    let point_id = wqm_common::hashing::compute_point_id(base_point, chunk_idx as u32);
    let content_hash = tracked_files_schema::compute_content_hash(content);

    let (sparse, used_lexicon_bm25) =
        generate_sparse(ctx, item, content, &embedding_result.sparse).await;

    if used_lexicon_bm25 && idf_epoch > 0 {
        point_payload.insert("idf_epoch".to_string(), serde_json::json!(idf_epoch));
    }

    let point = crate::storage::DocumentPoint {
        id: point_id.clone(),
        dense_vector: embedding_result.dense.vector,
        sparse_vector: sparse_embedding_to_map(&sparse),
        payload: point_payload,
    };

    ChunkOutput {
        point,
        record: ChunkRecord {
            point_id,
            chunk_index: chunk_idx as i32,
            content_hash,
            chunk_type,
            symbol_name,
            start_line,
            end_line,
        },
        lsp_status,
        treesitter_status,
    }
}

/// Split any chunk whose content exceeds `max_bytes` (UTF-8 byte length) into
/// sub-chunks that each fit the budget, renumbering `chunk_index` sequentially
/// so payloads and point IDs stay consistent. Sub-chunks inherit the parent's
/// metadata (line range, symbol) plus a `split_part = "i/n"` marker.
/// `usize::MAX` (no provider limit) returns the chunks unchanged.
///
/// The budget is bytes, not chars: byte-level BPE (cl100k/o200k) never emits
/// more tokens than bytes, so a byte cap is a hard upper bound on tokens.
fn split_oversized_chunks(chunks: &[crate::TextChunk], max_bytes: usize) -> Vec<crate::TextChunk> {
    if max_bytes == usize::MAX {
        return chunks.to_vec();
    }

    let mut out: Vec<crate::TextChunk> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        if chunk.content.len() <= max_bytes {
            out.push(chunk.clone());
            continue;
        }

        let pieces = split_text_on_budget(&chunk.content, max_bytes);
        let n = pieces.len();
        let mut char_cursor = chunk.start_char;
        for (i, piece) in pieces.into_iter().enumerate() {
            let piece_chars = piece.chars().count();
            let mut metadata = chunk.metadata.clone();
            metadata.insert("split_part".to_string(), format!("{}/{}", i + 1, n));
            out.push(crate::TextChunk {
                content: piece,
                chunk_index: 0, // renumbered below
                start_char: char_cursor,
                end_char: char_cursor + piece_chars,
                metadata,
            });
            char_cursor += piece_chars;
        }
    }

    for (i, chunk) in out.iter_mut().enumerate() {
        chunk.chunk_index = i;
    }
    out
}

/// Split `content` into pieces of at most `max_bytes` UTF-8 bytes, preferring
/// line boundaries. A single line longer than `max_bytes` is hard-split on
/// UTF-8 char boundaries (never mid-codepoint). Always returns at least one
/// piece.
fn split_text_on_budget(content: &str, max_bytes: usize) -> Vec<String> {
    debug_assert!(max_bytes > 0);
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();

    // `split_inclusive` keeps the trailing '\n' with each line so the
    // reassembled content is byte-identical to the original (no data loss).
    for line in content.split_inclusive('\n') {
        let line_bytes = line.len();

        if line_bytes > max_bytes {
            if !current.is_empty() {
                pieces.push(std::mem::take(&mut current));
            }
            pieces.extend(hard_split_bytes(line, max_bytes));
            continue;
        }

        if current.len() + line_bytes > max_bytes && !current.is_empty() {
            pieces.push(std::mem::take(&mut current));
        }
        current.push_str(line);
    }

    if !current.is_empty() {
        pieces.push(current);
    }
    if pieces.is_empty() {
        pieces.push(String::new());
    }
    pieces
}

/// Hard-split a string into pieces of at most `max_bytes` UTF-8 bytes, breaking
/// only on char boundaries so a multi-byte codepoint is never split. A single
/// char wider than `max_bytes` (only possible for an absurdly small budget) is
/// emitted on its own.
fn hard_split_bytes(s: &str, max_bytes: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut buf = String::new();
    for ch in s.chars() {
        if !buf.is_empty() && buf.len() + ch.len_utf8() > max_bytes {
            pieces.push(std::mem::take(&mut buf));
        }
        buf.push(ch);
    }
    if !buf.is_empty() {
        pieces.push(buf);
    }
    pieces
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use std::collections::HashMap;

    fn chunk(content: &str, idx: usize) -> crate::TextChunk {
        crate::TextChunk {
            content: content.to_string(),
            chunk_index: idx,
            start_char: 0,
            end_char: content.chars().count(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn no_limit_returns_chunks_unchanged() {
        let chunks = vec![chunk("a".repeat(100_000).as_str(), 0)];
        let out = split_oversized_chunks(&chunks, usize::MAX);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content.chars().count(), 100_000);
    }

    #[test]
    fn small_chunks_pass_through() {
        let chunks = vec![chunk("hello", 0), chunk("world", 1)];
        let out = split_oversized_chunks(&chunks, 100);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content, "hello");
    }

    #[test]
    fn oversized_chunk_is_split_under_budget_without_loss() {
        let body = (0..50)
            .map(|i| format!("line number {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let original = body.clone();
        let chunks = vec![chunk(&body, 0)];

        let out = split_oversized_chunks(&chunks, 40);

        assert!(out.len() > 1, "expected the oversized chunk to be split");
        for (i, c) in out.iter().enumerate() {
            assert!(
                c.content.len() <= 40,
                "piece {i} exceeds budget: {} bytes",
                c.content.len()
            );
            assert_eq!(c.chunk_index, i, "chunk_index must be renumbered");
            assert!(c.metadata.contains_key("split_part"));
        }
        // Reassembly is lossless.
        let rejoined: String = out.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(rejoined, original);
    }

    #[test]
    fn single_long_line_is_hard_split_on_char_boundaries() {
        // Multi-byte chars must never be split mid-codepoint. "あ" is 3 bytes,
        // so a 30-byte budget holds 10 chars per piece.
        let line = "あ".repeat(100); // no newlines, 100 chars = 300 bytes
        let pieces = split_text_on_budget(&line, 30);
        assert!(pieces.len() >= 4);
        for p in &pieces {
            assert!(
                p.len() <= 30,
                "piece exceeds byte budget: {} bytes",
                p.len()
            );
            // Every piece is valid UTF-8 (no mid-codepoint split).
            assert_eq!(p.chars().count() * 3, p.len());
        }
        let rejoined: String = pieces.concat();
        assert_eq!(rejoined, line);
    }
}

#[cfg(test)]
mod tests;
