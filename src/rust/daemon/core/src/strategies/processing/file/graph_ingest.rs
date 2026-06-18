//! Graph relationship extraction and storage during file ingestion.
//!
//! Non-blocking: graph errors are logged but never fail the ingestion pipeline.

use std::collections::HashMap;

use tracing::{debug, warn};

use crate::context::ProcessingContext;
use crate::graph::extractor::{extract_edges_from_text_chunks, ExtractionResult};
use crate::TextChunk;

use super::chunk_embed::ChunkRecord;

/// Extract graph relationships from text chunks and store them atomically.
///
/// Performs:
/// 1. Delete old edges for this file (cleanup from previous ingestion)
/// 2. Extract new nodes/edges from chunk metadata
/// 3. Annotate extracted chunk-symbol nodes with their Qdrant point IDs
///    (DATA-03: each chunk node that has a corresponding embedding carries
///    `qdrant_point_id` and `point_id_state = "linked"`)
/// 4. Merge caller-supplied concept nodes/edges (IMPLEMENTS_CONCEPT) and
///    narrative nodes/edges so they share the file's single delete-then-insert
///    transaction and are cleaned up on re-ingestion (their `source_file` is
///    this file's relative path).
/// 5. Upsert nodes + insert edges in a single write-lock hold
///
/// All graph errors are logged and swallowed — graph failures must never
/// block the main ingestion pipeline.
#[allow(clippy::too_many_arguments)]
pub(super) async fn ingest_graph_edges(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &str,
    chunks: &[TextChunk],
    branch: Option<&str>,
    chunk_records: &[ChunkRecord],
    concept_nodes: Vec<crate::graph::GraphNode>,
    concept_edges: Vec<crate::graph::GraphEdge>,
    narrative_nodes: Vec<crate::graph::GraphNode>,
    narrative_edges: Vec<crate::graph::GraphEdge>,
) {
    let Some(ref graph_store) = ctx.graph_store else {
        return; // Graph not initialized — skip silently
    };

    // Time the full processing-layer graph extraction + store cycle (PRD D5:
    // graph_extract_duration_seconds{layer=processing}).
    let extract_start = std::time::Instant::now();

    let ExtractionResult {
        mut nodes,
        mut edges,
    } = extract_edges_from_text_chunks(chunks, tenant_id, file_path, branch);

    // Link chunk-derived symbol nodes to their Qdrant point IDs.
    //
    // `chunk_records` maps chunk index → (symbol_name, point_id). Each symbol
    // node in the graph corresponds to a non-preamble, non-text chunk whose
    // `symbol_name` matches the node's `symbol_name`. Nodes are matched by
    // symbol_name; when multiple chunks share a name (e.g. split sub-chunks)
    // the first non-empty point_id wins (sub-chunks produce the same point_id
    // formula so any one is authoritative).
    annotate_point_ids(&mut nodes, chunk_records, tenant_id, file_path);

    // Merge concept nodes/edges into the same reingest transaction. ConceptNodes
    // are global (upserted, never deleted); concept edges carry source_file =
    // this file's relative path so the reingest DELETE cleans up stale ones.
    nodes.extend(concept_nodes);
    edges.extend(concept_edges);

    // Merge narrative nodes/edges (DocumentSection / CodeComment / Docstring +
    // EXPLAINS / DESCRIBES / REFERENCES_DOC / COVERS_TOPIC) into the SAME
    // transaction. A separate reingest_file call would delete the code and
    // concept edges just prepared above.
    nodes.extend(narrative_nodes);
    edges.extend(narrative_edges);

    if nodes.is_empty() && edges.is_empty() {
        return;
    }

    debug!(
        "Graph: extracting {} nodes, {} edges for {}",
        nodes.len(),
        edges.len(),
        file_path
    );

    match graph_store
        .reingest_file(tenant_id, file_path, &nodes, &edges)
        .await
    {
        Ok(()) => {
            crate::graph::metrics::record_graph_upsert(
                tenant_id,
                nodes.len() as u64,
                edges.len() as u64,
            );
        }
        Err(e) => {
            warn!(
                "Graph ingestion failed for {} (tenant {}): {}",
                file_path, tenant_id, e
            );
            crate::graph::metrics::record_graph_ingest_error(tenant_id);
        }
    }

    crate::graph::metrics::record_graph_extract_duration(
        tenant_id,
        crate::graph::metrics::LAYER_PROCESSING,
        extract_start.elapsed().as_secs_f64(),
    );
}

/// Annotate extracted graph nodes with Qdrant point IDs from chunk records.
///
/// Each chunk record carries the `point_id` computed by
/// `wqm_common::hashing::compute_point_id(base_point, chunk_index)` and the
/// `symbol_name` for code-symbol chunks. Graph nodes for code symbols are
/// matched by `symbol_name`; the first non-empty match wins. Non-symbol nodes
/// (file, concept, stub) remain unlinked (`point_id_state = "none"`).
fn annotate_point_ids(
    nodes: &mut Vec<crate::graph::GraphNode>,
    chunk_records: &[ChunkRecord],
    tenant_id: &str,
    file_path: &str,
) {
    if chunk_records.is_empty() {
        return;
    }

    // Build a map from symbol_name → point_id. When multiple records share a
    // name (split sub-chunks), all produce the same point_id formula, so we
    // just keep the first non-empty one.
    let mut by_symbol: HashMap<&str, &str> = HashMap::new();
    for rec in chunk_records {
        if let Some(ref sym) = rec.symbol_name {
            if !sym.is_empty() && !rec.point_id.is_empty() {
                by_symbol
                    .entry(sym.as_str())
                    .or_insert(rec.point_id.as_str());
            }
        }
    }

    if by_symbol.is_empty() {
        return;
    }

    let linked_count = nodes
        .iter_mut()
        .filter(|n| {
            // Only annotate nodes that belong to this file and tenant (not stubs,
            // not concept nodes, not narrative nodes from other sources).
            n.tenant_id == tenant_id && n.file_path == file_path && !n.symbol_name.is_empty()
        })
        .filter_map(|n| by_symbol.get(n.symbol_name.as_str()).map(|pid| (n, *pid)))
        .fold(0usize, |acc, (node, point_id)| {
            node.qdrant_point_id = Some(point_id.to_string());
            node.point_id_state = "linked".to_string();
            acc + 1
        });

    if linked_count > 0 {
        debug!(
            "Graph: linked {} chunk nodes to Qdrant point IDs for {} in tenant {}",
            linked_count, file_path, tenant_id
        );
    }
}

/// Delete graph edges for a file (called during file deletion).
///
/// Non-blocking: errors are logged but don't fail the deletion pipeline.
pub(super) async fn delete_graph_edges(ctx: &ProcessingContext, tenant_id: &str, file_path: &str) {
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
