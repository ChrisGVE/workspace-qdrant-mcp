//! Graph-augmented RAG expansion + score fusion (GitHub #80).
//!
//! Mirrors `expandAndFuseWithGraph` in `search-graph-expansion.ts`. Runs BETWEEN
//! RRF fusion+sort and diversity re-ranking when `includeGraphContext=true`:
//!
//! 1. Vector search (RRF) — results arrive pre-fused and sorted.
//! 2. Graph expansion — up to 2 hops, max 5 per result, max 50 total.
//! 3. Score fusion — `alpha * vector_score` for originals, `(1-alpha) * proximity`
//!    for graph-expanded nodes.
//! 4. Convergence bonus — `+0.1` for nodes found by BOTH vector and graph.
//!
//! Distinct from `graph_context::expand_graph_context` (the post-slice per-result
//! caller/callee enrichment, `expandGraphContext`), which also runs when
//! `includeGraphContext=true` but does NOT change the result set or scores.

use std::collections::HashSet;

use serde_json::Value;

use super::graph_context::{compute_node_id, GraphQueryDaemon};
use crate::proto::QueryRelatedRequest;
use crate::qdrant::fusion::{SearchType, TaggedResult};

// ── Constants (mirror search-graph-expansion.ts) ─────────────────────────────

const ALPHA: f64 = 0.7;
const CONVERGENCE_BONUS: f64 = 0.1;
const MAX_EXPANDED_PER_RESULT: usize = 5;
const MAX_EXPANDED_TOTAL: usize = 50;
const GRAPH_EXPANSION_TIMEOUT_MS: u64 = 500;
/// TS `candidates.slice(0, 20)` — only the top 20 results seed expansion.
const MAX_CANDIDATES: usize = 20;
/// Edge types traversed during fusion (TS `edge_types`).
const FUSION_EDGE_TYPES: &[&str] = &["CALLS", "USES_TYPE", "CONTAINS"];
const FUSION_MAX_HOPS: u32 = 2;

/// Chunk types eligible for fusion expansion.
///
/// Mirrors `CODE_CHUNK_TYPES` in `search-graph-expansion.ts:25-36` — note this
/// is the NARROW 10-entry set, deliberately distinct from the 13-entry set in
/// `graph_context::CODE_CHUNK_TYPES` (no `constant`/`type_alias`/`macro`).
const FUSION_CODE_CHUNK_TYPES: &[&str] = &[
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
];

// ── Candidate collection ─────────────────────────────────────────────────────

struct ExpandCandidate {
    tenant_id: String,
    node_id: String,
}

/// Build the expansion candidate list from the fused results (TS `collectCandidates`).
fn collect_candidates(results: &[TaggedResult]) -> Vec<ExpandCandidate> {
    let mut candidates = Vec::new();
    for r in results {
        let symbol_name = r.payload.get("chunk_symbol_name").and_then(|v| v.as_str());
        let chunk_type = r.payload.get("chunk_chunk_type").and_then(|v| v.as_str());
        let tenant_id = r.payload.get("tenant_id").and_then(|v| v.as_str());
        let file_path = r
            .payload
            .get("relative_path")
            .or_else(|| r.payload.get("file_path"))
            .and_then(|v| v.as_str());
        let (Some(sym), Some(ct), Some(tid), Some(fp)) =
            (symbol_name, chunk_type, tenant_id, file_path)
        else {
            continue;
        };
        if !FUSION_CODE_CHUNK_TYPES.contains(&ct) {
            continue;
        }
        candidates.push(ExpandCandidate {
            tenant_id: tid.to_string(),
            node_id: compute_node_id(tid, fp, sym, ct),
        });
    }
    candidates
}

/// Graph proximity from hop distance (TS `graphProximityScore`).
fn graph_proximity_score(depth: u32) -> f64 {
    match depth {
        0 => 1.0,
        1 => 0.8,
        _ => 0.5,
    }
}

/// Build an expanded result node (TS `nodeToSearchResult`, then `score = (1-ALPHA)*proximity`).
fn node_to_tagged(
    node: &crate::proto::TraversalNodeProto,
    collection: &str,
    tenant_id: &str,
    proximity: f64,
) -> TaggedResult {
    let mut payload = std::collections::HashMap::new();
    payload.insert(
        "tenant_id".to_string(),
        Value::String(tenant_id.to_string()),
    );
    payload.insert(
        "chunk_symbol_name".to_string(),
        Value::String(node.symbol_name.clone()),
    );
    payload.insert(
        "chunk_chunk_type".to_string(),
        Value::String(node.symbol_type.clone()),
    );
    payload.insert(
        "file_path".to_string(),
        Value::String(node.file_path.clone()),
    );
    payload.insert(
        "source".to_string(),
        Value::String("graph_expansion".to_string()),
    );
    payload.insert(
        "content".to_string(),
        Value::String(format!(
            "{} {} in {}",
            node.symbol_type, node.symbol_name, node.file_path
        )),
    );
    payload.insert("title".to_string(), Value::String(node.symbol_name.clone()));
    TaggedResult {
        id: node.node_id.clone(),
        score: (1.0 - ALPHA) * proximity,
        collection: collection.to_string(),
        payload,
        search_type: SearchType::Graph,
    }
}

// ── Public entry point ───────────────────────────────────────────────────────

/// Expand fused vector results with graph traversal and fuse scores in place.
///
/// Mirrors `expandAndFuseWithGraph`. Adds graph-expanded nodes, alpha-blends
/// scores, applies a convergence bonus to results found by both modalities, and
/// re-sorts by score. Must run BEFORE diversity re-ranking and slicing.
pub async fn expand_and_fuse_with_graph<D>(
    daemon: &mut D,
    results: &mut Vec<TaggedResult>,
    collection: &str,
) where
    D: GraphQueryDaemon,
{
    let candidates = collect_candidates(results);
    if candidates.is_empty() {
        return;
    }

    // `existing_ids` tracks both original result ids and already-added expanded
    // nodes so we never add a duplicate (TS `existingIds`).
    let mut existing_ids: HashSet<String> = results.iter().map(|r| r.id.clone()).collect();
    let mut expanded: Vec<TaggedResult> = Vec::new();
    let mut total_expanded = 0usize;

    // TS fires the top-20 candidates concurrently; the daemon handle here is
    // `&mut`, so we expand sequentially in candidate order (deterministic).
    for candidate in candidates.iter().take(MAX_CANDIDATES) {
        let req = QueryRelatedRequest {
            tenant_id: candidate.tenant_id.clone(),
            node_id: candidate.node_id.clone(),
            max_hops: FUSION_MAX_HOPS,
            edge_types: FUSION_EDGE_TYPES.iter().map(|s| s.to_string()).collect(),
            branch: None,
        };
        let response = match tokio::time::timeout(
            std::time::Duration::from_millis(GRAPH_EXPANSION_TIMEOUT_MS),
            daemon.query_related(req),
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            _ => continue, // timeout or error — swallow (TS catch → [])
        };

        for node in response.nodes.iter().take(MAX_EXPANDED_PER_RESULT) {
            if node.node_id == candidate.node_id {
                continue; // self
            }
            if existing_ids.contains(&node.node_id) {
                // Convergence: bonus an ORIGINAL result (one already in `results`;
                // previously-added expanded nodes are not in `results` yet, so
                // they receive no bonus — matches TS `results.find`).
                if let Some(existing) = results.iter_mut().find(|r| r.id == node.node_id) {
                    existing.score += CONVERGENCE_BONUS;
                }
                continue;
            }
            if total_expanded >= MAX_EXPANDED_TOTAL {
                break;
            }
            let proximity = graph_proximity_score(node.depth);
            expanded.push(node_to_tagged(
                node,
                collection,
                &candidate.tenant_id,
                proximity,
            ));
            existing_ids.insert(node.node_id.clone());
            total_expanded += 1;
        }
    }

    // Alpha-scale the original results (TS `r.score = ALPHA * r.score`; the TS
    // `+ (r.score > 0 ? 0 : 0)` term is an always-zero no-op). Convergence
    // bonuses added above are included in `r.score` here, matching TS ordering.
    for r in results.iter_mut() {
        r.score *= ALPHA;
    }

    results.extend(expanded);
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
#[path = "graph_fusion_tests.rs"]
mod tests;
