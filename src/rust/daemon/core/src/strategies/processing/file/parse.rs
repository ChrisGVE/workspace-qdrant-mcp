//! Document parse + identifier phase of file ingestion (phase 0 + 1).
//!
//! Extracted from `ingest.rs` so the `extract.document` / `chunk.tree_sitter`
//! tracing spans (PRD B2) wrap the parse + chunking call without bloating the
//! ingest orchestration module.

use std::path::Path;
use std::time::Instant;

use tracing::{info, Instrument};

use crate::context::ProcessingContext;
use crate::monitoring::labels::cardinality::bounded_file_type;
use crate::processing_timings::PhaseTiming;
use crate::tracing_gate::{tier_enabled, TraceTier};
use crate::tracked_files_schema;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{FilePayload, UnifiedQueueItem};

/// Parse the document and compute file identifiers (phase 0 + 1).
///
/// The `extract.document` span is the consumer-side root of the extract step;
/// the chunking call (`process_file_content_with_provider`, which runs the
/// tree-sitter / fallback chunker) is wrapped in a child `chunk.tree_sitter`
/// span. Both are debug-level and skip_all; their bounded attributes are only
/// computed when the Hot trace tier is active so they cost ~nothing when off.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    name = "extract.document",
    level = "debug",
    skip_all,
    fields(
        file.size_bytes = tracing::field::Empty,
        extractor = tracing::field::Empty,
    )
)]
pub(super) async fn parse_document(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
    payload: &FilePayload,
    file_path: &Path,
    abs_file_path: &str,
    relative_path: &str,
    provider: Option<std::sync::Arc<dyn crate::tree_sitter::parser::LanguageProvider>>,
    timings: &mut Vec<PhaseTiming>,
) -> UnifiedProcessorResult<(crate::DocumentContent, String, String, String)> {
    if tier_enabled(TraceTier::Hot) {
        let span = tracing::Span::current();
        if let Ok(meta) = std::fs::metadata(file_path) {
            span.record("file.size_bytes", meta.len());
        }
        // The extractor is selected by file type inside the document processor;
        // record the bounded file-type label as a stable, low-cardinality proxy.
        span.record("extractor", bounded_file_type(file_path));
    }

    let t0 = Instant::now();

    // chunk.tree_sitter: the chunking call (tree-sitter when a grammar is
    // available, otherwise the text fallback) nested under extract.document.
    let chunk_span = tracing::debug_span!(
        "chunk.tree_sitter",
        chunk.count = tracing::field::Empty,
        grammar = tracing::field::Empty,
    );
    let document_content = ctx
        .document_processor
        .process_file_content_with_provider(file_path, &item.collection, provider)
        .instrument(chunk_span.clone())
        .await
        .map_err(|e| UnifiedProcessorError::ProcessingFailed(e.to_string()))?;
    if tier_enabled(TraceTier::Hot) {
        chunk_span.record("chunk.count", document_content.chunks.len());
        chunk_span.record("grammar", bounded_file_type(file_path));
    }

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
