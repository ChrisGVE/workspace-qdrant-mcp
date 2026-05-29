//! Backend-agnostic cross-boundary traversal helpers.
//!
//! These functions encode the shared semantics that every `GraphStore`
//! backend must reproduce for `query_cross_boundary`:
//! - the per-edge-type base confidence used to score concept/narrative hops,
//! - the Rust-side fan-out caps applied after a backend over-fetches results.
//!
//! Both `SqliteGraphStore` and `LadybugGraphStore` call these so the two
//! backends stay equivalent (verified by the conformance suite).

use crate::config::GraphRagConfig;

use super::TraversalNode;

/// Tenant under which cross-boundary concept nodes are stored.
pub(crate) const GLOBAL_TENANT: &str = "__global__";

/// Maximum recursion depth supported by cross-boundary traversal.
pub(crate) const CROSS_BOUNDARY_MAX_HOPS: u32 = 3;

/// Per-edge-type base confidence applied to a single cross-boundary hop.
///
/// Structural edges keep full confidence; concept/narrative bridges are
/// discounted so cross-domain expansions rank below same-tenant matches.
/// The reaching edge's `weight` is multiplied by this base.
///
/// The SQLite backend inlines this mapping as a SQL `CASE`; only the LadybugDB
/// backend computes confidence in Rust, so the helper is gated to that feature.
#[cfg(feature = "ladybug")]
pub(crate) fn edge_type_base_confidence(edge_type: &str) -> f64 {
    match edge_type {
        "EXPLAINS" => 0.6,
        "COVERS_TOPIC" => 0.6,
        "IMPLEMENTS_CONCEPT" => 0.7,
        _ => 1.0,
    }
}

/// Build the tenant relaxation set for a cross-boundary query:
/// `source_tenant ∪ {"__global__"} ∪ library_tenants`.
///
/// `__global__` is always included because concept nodes live there; without
/// it, code→concept→code traversal returns nothing.
pub(crate) fn tenant_relaxation_set(
    source_tenant: &str,
    library_tenants: &[String],
) -> Vec<String> {
    let mut tenants: Vec<String> = Vec::with_capacity(library_tenants.len() + 2);
    tenants.push(source_tenant.to_string());
    tenants.push(GLOBAL_TENANT.to_string());
    for lt in library_tenants {
        tenants.push(lt.clone());
    }
    tenants
}

/// Apply Rust-side fan-out caps to over-fetched cross-boundary results.
///
/// Backends cannot rank-limit per recursion level inside the engine, so they
/// over-fetch and the caps are enforced here:
/// 1. `max_per_hit`: keep top-K hop-1 (direct) expansions by `edge_confidence`.
/// 2. `max_per_concept`: keep top-K nodes reached through any single
///    ConceptNode by `edge_confidence` (tames concept supernodes).
/// 3. `max_total`: keep the top-K nodes overall by `edge_confidence`.
///
/// Nodes are ranked by `edge_confidence` (desc), then by ascending depth and
/// node_id for a stable order, and finally re-sorted into depth-major order
/// for callers that expect breadth ordering.
pub(crate) fn apply_fan_out_caps(
    mut nodes: Vec<TraversalNode>,
    config: &GraphRagConfig,
) -> Vec<TraversalNode> {
    use std::collections::{HashMap, HashSet};

    // Identify ConceptNodes present in the result set; concept hubs are always
    // reachable (global tenant) so any concept on a surviving path appears here.
    let concept_ids: HashSet<String> = nodes
        .iter()
        .filter(|n| n.symbol_type == "concept_node")
        .map(|n| n.node_id.clone())
        .collect();

    // Stable ranking helper: higher confidence first, then shallower, then id.
    let rank = |a: &TraversalNode, b: &TraversalNode| {
        b.edge_confidence
            .partial_cmp(&a.edge_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.depth.cmp(&b.depth))
            .then(a.node_id.cmp(&b.node_id))
    };
    nodes.sort_by(rank);

    let mut direct_kept = 0usize;
    let mut per_concept: HashMap<String, usize> = HashMap::new();

    let mut kept: Vec<TraversalNode> = Vec::with_capacity(nodes.len().min(config.max_total));
    for node in nodes {
        if kept.len() >= config.max_total {
            break;
        }
        if node.depth == 1 {
            if direct_kept >= config.max_per_hit {
                continue;
            }
            direct_kept += 1;
            kept.push(node);
            continue;
        }
        // Deeper node: attribute it to the last ConceptNode on its path, if any.
        let via_concept = node
            .path
            .split(" -> ")
            .filter(|id| concept_ids.contains(*id))
            .last()
            .map(|id| id.to_string());
        if let Some(concept) = via_concept {
            let count = per_concept.entry(concept).or_insert(0);
            if *count >= config.max_per_concept {
                continue;
            }
            *count += 1;
        }
        kept.push(node);
    }

    kept.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then(a.symbol_name.cmp(&b.symbol_name))
    });
    kept
}
