//! Graph relationship extraction and storage during file ingestion.
//!
//! Non-blocking: graph errors are logged but never fail the ingestion pipeline.
//!
//! Tree-sitter is the always-on baseline edge source. When an LSP server is
//! already warm for the file, an additive precision pass resolves `CALLS` edges
//! via call hierarchy: tree-sitter emits a name-only stub callee (empty
//! file_path → an id that never matches the callee's real node), whereas LSP
//! knows the callee's definition site, so we add an edge to its real node_id.
//! The pass is gated on server readiness and is a no-op on a cold index, so it
//! never adds latency to the common path.

use std::path::Path;

use tracing::{debug, warn};

use crate::context::ProcessingContext;
use crate::graph::extractor::{
    extract_edges_from_text_chunks, node_type_from_display_name, ExtractionResult,
};
use crate::graph::compute_node_id;
use crate::lsp::{resolved_call_edges, symbol_column_in_line};
use crate::TextChunk;

/// Extract graph relationships from text chunks and store them atomically.
///
/// Performs:
/// 1. Delete old edges for this file (cleanup from previous ingestion)
/// 2. Extract new nodes/edges from chunk metadata (tree-sitter baseline)
/// 3. LSP precision pass for `CALLS` edges when a server is ready (additive)
/// 4. Upsert nodes + insert edges in a single write-lock hold
///
/// All graph errors are logged and swallowed — graph failures must never
/// block the main ingestion pipeline.
pub(super) async fn ingest_graph_edges(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &str,
    abs_file_path: &str,
    chunks: &[TextChunk],
) {
    let Some(ref graph_store) = ctx.graph_store else {
        return; // Graph not initialized — skip silently
    };

    let mut extraction = extract_edges_from_text_chunks(chunks, tenant_id, file_path);

    // Additive LSP precision pass (best-effort; no-op when no server is ready).
    resolve_calls_via_lsp(ctx, tenant_id, file_path, abs_file_path, chunks, &mut extraction).await;

    let ExtractionResult { nodes, edges } = extraction;

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
            // Throughput metric: count freshly-written edges by type so the
            // Grafana "Code Graph" dashboard can show ingest rate per edge type.
            let mut by_type: std::collections::HashMap<&str, u64> =
                std::collections::HashMap::new();
            for edge in &edges {
                *by_type.entry(edge.edge_type.as_str()).or_default() += 1;
            }
            for (edge_type, count) in by_type {
                crate::monitoring::metrics_core::METRICS
                    .graph_edges_ingested_total
                    .with_label_values(&[tenant_id, edge_type])
                    .inc_by(count);
            }
        }
        Err(e) => {
            warn!(
                "Graph ingestion failed for {} (tenant {}): {}",
                file_path, tenant_id, e
            );
        }
    }
}

/// LSP precision pass: resolve `CALLS` edges to real callee nodes.
///
/// For each function/method chunk, asks the (already-warm) LSP server for the
/// symbol's outgoing calls and adds an edge to each resolved callee's real
/// node_id. Gated on `is_server_ready_for_file`, so it is a no-op when no
/// server is running for the tenant (cold index / LSP disabled). Callee node
/// type defaults to `Function` (best-effort; free functions are the common
/// case — a mismatched method target is simply an unmatched extra node, no
/// worse than the tree-sitter stub it complements).
async fn resolve_calls_via_lsp(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &str,
    abs_file_path: &str,
    chunks: &[TextChunk],
    extraction: &mut ExtractionResult,
) {
    let Some(ref lsp_arc) = ctx.lsp_manager else {
        return;
    };
    let abs_path = Path::new(abs_file_path);
    let mgr = lsp_arc.read().await;
    if !mgr.is_server_ready_for_file(tenant_id, abs_path).await {
        return; // Server not warm for this file — tree-sitter edges stand.
    }

    // Derive the project root by removing the relative suffix from the absolute
    // path; used to relativize LSP-returned callee paths back to graph keys.
    let norm_abs = abs_file_path.replace('\\', "/");
    let norm_rel = file_path.replace('\\', "/");
    let Some(project_root) = norm_abs
        .strip_suffix(&norm_rel)
        .map(|r| r.trim_end_matches('/').to_string())
    else {
        return; // Can't derive root (path layout unexpected) — skip safely.
    };

    for chunk in chunks {
        let meta = &chunk.metadata;
        let Some(chunk_type) = meta.get("chunk_type") else {
            continue;
        };
        // Only callable definitions have outgoing calls.
        let Some(node_type) = node_type_from_display_name(chunk_type) else {
            continue;
        };
        if !matches!(
            chunk_type.as_str(),
            "function" | "async_function" | "method"
        ) {
            continue;
        }
        let Some(symbol) = meta.get("symbol_name").filter(|s| !s.is_empty()) else {
            continue;
        };
        let Some(line) = meta.get("start_line").and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };

        // Column of the symbol on its definition line (UTF-16, LSP encoding).
        let first_line = chunk.content.lines().next().unwrap_or("");
        let column = symbol_column_in_line(first_line, symbol);

        let calls = mgr
            .resolved_outgoing_calls(abs_path, line, column)
            .await
            .unwrap_or_default();
        if calls.is_empty() {
            continue;
        }

        let caller_id = compute_node_id(tenant_id, file_path, symbol, node_type);
        let (nodes, edges) =
            resolved_call_edges(tenant_id, &caller_id, file_path, &project_root, &calls);
        debug!(
            "Graph LSP pass: {} resolved call edge(s) from {}",
            edges.len(),
            symbol
        );
        extraction.nodes.extend(nodes);
        extraction.edges.extend(edges);
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
