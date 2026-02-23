//! Graph relationship extraction and storage during file ingestion.
//!
//! Non-blocking: graph errors are logged but never fail the ingestion pipeline.

use tracing::{debug, warn};

use crate::context::ProcessingContext;
use crate::graph::extractor::{extract_edges_from_text_chunks, ExtractionResult};
use crate::TextChunk;

/// Extract graph relationships from text chunks and store them atomically.
///
/// Performs:
/// 1. Delete old edges for this file (cleanup from previous ingestion)
/// 2. Extract new nodes/edges from chunk metadata
/// 3. Upsert nodes + insert edges in a single write-lock hold
///
/// All graph errors are logged and swallowed — graph failures must never
/// block the main ingestion pipeline.
pub(super) async fn ingest_graph_edges(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &str,
    chunks: &[TextChunk],
) {
    let Some(ref graph_store) = ctx.graph_store else {
        return; // Graph not initialized — skip silently
    };

    let ExtractionResult { nodes, edges } =
        extract_edges_from_text_chunks(chunks, tenant_id, file_path);

    if nodes.is_empty() && edges.is_empty() {
        return;
    }

    debug!(
        "Graph: extracting {} nodes, {} edges for {}",
        nodes.len(),
        edges.len(),
        file_path
    );

    if let Err(e) = graph_store
        .reingest_file(tenant_id, file_path, &nodes, &edges)
        .await
    {
        warn!(
            "Graph ingestion failed for {} (tenant {}): {}",
            file_path, tenant_id, e
        );
    }
}

/// Delete graph edges for a file (called during file deletion).
///
/// Non-blocking: errors are logged but don't fail the deletion pipeline.
pub(super) async fn delete_graph_edges(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &str,
) {
    let Some(ref graph_store) = ctx.graph_store else {
        return;
    };

    // Use reingest_file with empty nodes/edges to just delete old edges
    if let Err(e) = graph_store
        .reingest_file(tenant_id, file_path, &[], &[])
        .await
    {
        warn!(
            "Graph edge deletion failed for {} (tenant {}): {}",
            file_path, tenant_id, e
        );
    }
}

