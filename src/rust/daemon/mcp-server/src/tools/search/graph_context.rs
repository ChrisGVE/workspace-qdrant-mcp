//! Graph context enrichment for search results.
//!
//! Mirrors `expandGraphContext` in `search-graph-context.ts` and
//! `expandAndFuseWithGraph` in `search-graph-expansion.ts`.
//!
//! When `includeGraphContext=true`, fetches 1-hop callers/callees from the
//! daemon's GraphService for each search result that contains a code symbol.
//!
//! ## Node ID computation
//! SHA256(tenant_id|file_path|symbol_name|symbol_type)[..16] as hex — matches
//! TS `computeNodeId` in `search-graph-context.ts:43-52` and Rust daemon.

use sha2::{Digest, Sha256};

use super::types::{GraphContext, GraphContextNode, SearchResult};
use crate::proto::{QueryRelatedRequest, QueryRelatedResponse};

// ---------------------------------------------------------------------------
// Constants (mirrors search-graph-context.ts)
// ---------------------------------------------------------------------------

/// Chunk types eligible for graph context.
///
/// Mirrors `CODE_CHUNK_TYPES` in `search-graph-context.ts:20-34`.
const CODE_CHUNK_TYPES: &[&str] = &[
    "function",
    "async_function",
    "method",
    "class",
    "struct",
    "trait",
    "interface",
    "enum",
    "impl",
    "module",
    "constant",
    "type_alias",
    "macro",
];

// ---------------------------------------------------------------------------
// Dependency trait (injectable for tests)
// ---------------------------------------------------------------------------

/// Trait for graph query_related — injectable in tests.
pub trait GraphQueryDaemon: Send + Sync {
    fn query_related(
        &mut self,
        request: QueryRelatedRequest,
    ) -> impl std::future::Future<Output = Result<QueryRelatedResponse, tonic::Status>> + Send;
}

// ---------------------------------------------------------------------------
// Node ID computation
// ---------------------------------------------------------------------------

/// Compute node_id matching Rust daemon's `compute_node_id`.
///
/// SHA256(tenant_id|file_path|symbol_name|symbol_type)[..16] as hex.
/// Mirrors `computeNodeId` in `search-graph-context.ts:43-52`. Shared with the
/// graph-fusion pass (`graph_fusion.rs`, GitHub #80).
pub(super) fn compute_node_id(
    tenant_id: &str,
    file_path: &str,
    symbol_name: &str,
    symbol_type: &str,
) -> String {
    let input = format!("{tenant_id}|{file_path}|{symbol_name}|{symbol_type}");
    let hash = Sha256::digest(input.as_bytes());
    let hex = hex_encode(&hash);
    hex[..32].to_string()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Enrichment target collection
// ---------------------------------------------------------------------------

fn collect_enrich_targets(results: &mut Vec<SearchResult>) -> Vec<usize> {
    let mut eligible: Vec<usize> = Vec::new();
    for (i, r) in results.iter().enumerate() {
        let symbol_name = r.metadata.get("chunk_symbol_name").and_then(|v| v.as_str());
        let chunk_type = r.metadata.get("chunk_chunk_type").and_then(|v| v.as_str());
        let tenant_id = r.metadata.get("tenant_id").and_then(|v| v.as_str());
        let file_path = r
            .metadata
            .get("relative_path")
            .or_else(|| r.metadata.get("file_path"))
            .and_then(|v| v.as_str());

        if let (Some(sym), Some(ct), Some(tid), Some(fp)) =
            (symbol_name, chunk_type, tenant_id, file_path)
        {
            if CODE_CHUNK_TYPES.contains(&ct) {
                let _ = (sym, tid, fp); // ensure they're used
                eligible.push(i);
            }
        }
    }
    eligible
}

// ---------------------------------------------------------------------------
// Public function
// ---------------------------------------------------------------------------

/// Enrich search results with 1-hop graph context for code symbols.
///
/// Mirrors `expandGraphContext` in `search-graph-context.ts:155-163`.
/// Queries are fired concurrently with a 200ms per-query timeout.
/// Failures are silently swallowed.
pub async fn expand_graph_context<D>(daemon: &mut D, results: &mut Vec<SearchResult>)
where
    D: GraphQueryDaemon,
{
    let eligible = collect_enrich_targets(results);
    if eligible.is_empty() {
        return;
    }

    // Build (node_id, tenant_id, file_path, symbol_name) tuples.
    let targets: Vec<(usize, String, String, String, String)> = eligible
        .iter()
        .filter_map(|&i| {
            let r = &results[i];
            let symbol_name = r
                .metadata
                .get("chunk_symbol_name")
                .and_then(|v| v.as_str())?
                .to_string();
            let chunk_type = r
                .metadata
                .get("chunk_chunk_type")
                .and_then(|v| v.as_str())?
                .to_string();
            let tenant_id = r
                .metadata
                .get("tenant_id")
                .and_then(|v| v.as_str())?
                .to_string();
            let file_path = r
                .metadata
                .get("relative_path")
                .or_else(|| r.metadata.get("file_path"))
                .and_then(|v| v.as_str())?
                .to_string();
            let node_id = compute_node_id(&tenant_id, &file_path, &symbol_name, &chunk_type);
            Some((i, node_id, tenant_id, file_path, symbol_name))
        })
        .collect();

    // Query graph for each target (sequential — daemon is &mut).
    for (idx, node_id, tenant_id, file_path, symbol_name) in targets {
        let req = QueryRelatedRequest {
            tenant_id: tenant_id.clone(),
            node_id: node_id.clone(),
            max_hops: 1,
            edge_types: vec![],
            branch: None,
        };

        // 200ms per-query timeout — mirrors GRAPH_QUERY_TIMEOUT_MS in TS.
        let response = match tokio::time::timeout(
            std::time::Duration::from_millis(200),
            daemon.query_related(req),
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            _ => continue, // timeout or error — swallow silently
        };

        if response.nodes.is_empty() {
            continue;
        }

        let mut callers: Vec<GraphContextNode> = Vec::new();
        let mut callees: Vec<GraphContextNode> = Vec::new();

        for node in &response.nodes {
            if node.node_id == node_id {
                continue;
            }
            let ctx_node = GraphContextNode {
                symbol: node.symbol_name.clone(),
                file_path: node.file_path.clone(),
                line: None,
            };
            // Mirrors TS: CALLS_REVERSE | CONTAINS → callers; else → callees
            if node.edge_type == "CALLS_REVERSE" || node.edge_type == "CONTAINS" {
                callers.push(ctx_node);
            } else {
                callees.push(ctx_node);
            }
        }

        results[idx].graph_context = Some(GraphContext {
            symbol: symbol_name,
            file_path,
            callers,
            callees,
        });
    }
}
