//! `ladybug_store/path.rs` — BFS-driven path and cross-boundary traversal
//! methods for `LadybugGraphStore`.
//!
//! These two methods drive the BFS machinery defined in `traversal.rs` and are
//! the most complex read methods in the backend. Splitting them here keeps both
//! `query.rs` (simpler reads) and this file well under the 500-line limit while
//! keeping the BFS-driver logic in one place for reviewability.
//!
//! - `do_find_path`          — shortest-path BFS between two nodes (CR-011).
//! - `do_query_cross_boundary` — bidirectional multi-tenant BFS with fan-out
//!                              caps and acyclic guard (CR-021).
//!
//! Private BFS primitives (`fetch_path_node`, `find_path_neighbours`,
//! `cross_boundary_neighbours`, etc.) live in `traversal.rs` and are called
//! here through `self.*`.

use lbug::Value;
use tracing::warn;

use crate::graph::{
    cross_boundary::{apply_fan_out_caps, tenant_relaxation_set, CROSS_BOUNDARY_MAX_HOPS},
    is_cross_branch,
    schema::{GraphDbError, GraphDbResult},
    EdgeType, TraversalNode,
};

use super::helpers::{distinct_frontier_tails, tenant_param_list};
use super::helpers::{CrossBoundaryPath, ALL_REL_TYPES, MAX_FRONTIER_PATHS};
use super::init::LadybugGraphStore;

impl LadybugGraphStore {
    /// LadybugDB backend: branch scoping is not implemented. A branch-scoped
    /// query (`branch = Some(name)` with `name != "*"`) returns a
    /// [`GraphDbError::BranchScopingUnsupported`] (surfaced as gRPC
    /// `Status::unimplemented`) rather than silently returning cross-branch
    /// results. `None` or `Some("*")` behave as before (no filtering).
    pub(super) async fn do_find_path(
        &self,
        tenant_id: &str,
        source_id: &str,
        target_id: &str,
        max_depth: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        if !is_cross_branch(branch) {
            return Err(GraphDbError::BranchScopingUnsupported(
                branch.unwrap_or_default().to_string(),
            ));
        }
        // Self-path: source == target — return the source node at depth 0,
        // matching SQLite's BFS base-case behaviour.
        if source_id == target_id {
            return self.find_path_self(tenant_id, source_id);
        }

        // Rel-type pattern for the single-hop neighbour query.
        // Rel type names come from EdgeType::as_str() (compile-time literals),
        // so string interpolation is injection-safe.
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => ALL_REL_TYPES.join("|"),
        };

        // Rust-side BFS replicating SQLite's `WITH RECURSIVE bfs` logic:
        //   - Each candidate is a full path (Vec<PathNode>) carrying every node's
        //     identity columns, so the no-revisit check is O(path len) and
        //     reconstruction reads from memory with no extra query (CR-011).
        //   - We process complete hop levels in ascending order, so the first
        //     time target_id is reached it is guaranteed to be a minimum-hop path.
        //   - Tie-breaking among equal-depth paths is implementation-defined
        //     (matches SQLite, which also gives no tie-break guarantee via LIMIT 1
        //     on the CTE result); tests use unique-shortest-path fixtures only.

        let conn = self.connect()?;

        // The source node is the only path entry that is never returned as a
        // neighbour row, so fetch its identity columns once to seed the path.
        // A single query, not an N+1 (CR-011). A missing source means no path.
        let Some(source_node) = self.fetch_path_node(&conn, tenant_id, source_id)? else {
            return Ok(None);
        };

        // frontier: list of in-progress paths, each ending at the current node.
        let mut frontier: Vec<Vec<super::helpers::PathNode>> = vec![vec![source_node]];

        for _hop in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            // CR-011: expand the WHOLE frontier with ONE neighbour query per hop
            // instead of one per frontier path. We collect the distinct current
            // node-ids, fetch every neighbour in a single bound IN-list query,
            // then expand each path against that map in pure Rust. FFI calls per
            // hop drop from O(frontier size) to O(1); kuzu query compilation is
            // entered once per hop rather than once per path.
            let current_ids = distinct_frontier_tails(&frontier);
            let neighbours_by_source =
                self.find_path_neighbours(&conn, tenant_id, &current_ids, &rel_pattern)?;

            let mut next_frontier: Vec<Vec<super::helpers::PathNode>> = Vec::new();
            for path in &frontier {
                // A frontier path is never empty by construction (every entry
                // starts with `source_id` and only grows). Handle the empty
                // case gracefully rather than panicking: a panic here would
                // unwind the tokio runtime on a fallible async path (CR-020).
                let Some(current) = path.last() else {
                    continue;
                };
                let Some(neighbours) = neighbours_by_source.get(&current.node_id) else {
                    continue; // this node had no outgoing neighbours
                };

                for neighbour in neighbours {
                    // No-revisit within this path (mirrors SQLite's INSTR check),
                    // comparing whole node-ids.
                    if path.iter().any(|n| n.node_id == neighbour.node_id) {
                        continue;
                    }

                    let mut new_path = path.clone();
                    new_path.push(neighbour.clone());

                    if neighbour.node_id == target_id {
                        // Found the target — every node on the path already
                        // carries its columns, so reconstruction needs no query.
                        return Ok(Some(self.reconstruct_path(tenant_id, &new_path)));
                    }

                    // Frontier cap (CR-010): a dense graph can blow the frontier
                    // up exponentially. Once we hit the cap, stop expanding and
                    // give up the search — the target was not reached within the
                    // budget, so return Ok(None) just as we do when paths are
                    // exhausted.
                    if next_frontier.len() >= MAX_FRONTIER_PATHS {
                        warn!(
                            "find_path frontier reached cap ({}); returning no path",
                            MAX_FRONTIER_PATHS
                        );
                        return Ok(None);
                    }
                    next_frontier.push(new_path);
                }
            }

            frontier = next_frontier;
        }

        // Exhausted all paths within max_depth without reaching target.
        Ok(None)
    }

    pub(super) async fn do_query_cross_boundary(
        &self,
        source_tenant: &str,
        source_node_id: &str,
        edge_types: &[EdgeType],
        max_hops: u32,
        library_tenants: &[String],
    ) -> GraphDbResult<Vec<TraversalNode>> {
        if edge_types.is_empty() || max_hops == 0 {
            return Ok(Vec::new());
        }
        let hops = max_hops.clamp(1, CROSS_BOUNDARY_MAX_HOPS);
        let conn = self.connect()?;

        // Tenant relaxation set: source ∪ {"__global__"} ∪ library_tenants.
        let tenants = tenant_relaxation_set(source_tenant, library_tenants);

        // Rel-type pattern (static literals from EdgeType — injection-safe).
        let rel_pattern = edge_types
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join("|");

        // Bidirectional BFS expanded one hop length at a time so we can record
        // the TRUE minimum depth, the reaching edge type, and the node path for
        // each reached node (the shared fan-out caps depend on `path`). Each hop
        // expands the current frontier in both directions, applying the tenant
        // guard so we never traverse through a foreign tenant.
        let mut reached: std::collections::HashMap<String, TraversalNode> =
            std::collections::HashMap::new();
        // Each frontier entry maps a node_id to the [`CrossBoundaryPath`] that
        // reached it: the display path string (used for `TraversalNode.path`
        // and concept attribution in `apply_fan_out_caps`) plus the ordered set
        // of node-ids already visited on that path. The visited set drives the
        // acyclic guard via exact id comparison; a node_id that itself contains
        // the path separator " -> " no longer corrupts cycle detection (CR-021).
        let mut frontier: std::collections::HashMap<String, CrossBoundaryPath> =
            std::collections::HashMap::new();
        frontier.insert(
            source_node_id.to_string(),
            CrossBoundaryPath::seed(source_node_id),
        );

        // Seed-ownership guard (parity with SQLite): only traverse when the
        // source node belongs to the relaxation set (source_tenant ∪
        // __global__ ∪ library_tenants); otherwise a foreign seed could reach
        // __global__ / library nodes and bypass tenant scoping. Concept and
        // library seeds remain valid. The per-hop query guards reached nodes.
        {
            let (tenant_list, tenant_names) = tenant_param_list(tenants.len());
            let cypher = format!(
                "MATCH (n:GraphNode {{node_id: $id}}) \
                 WHERE n.tenant_id IN {tenant_list} RETURN n.node_id"
            );
            let mut params: Vec<(&str, Value)> = Vec::with_capacity(tenants.len() + 1);
            params.push(("id", Value::String(source_node_id.to_string())));
            for (name, tenant) in tenant_names.iter().zip(tenants.iter()) {
                params.push((name.as_str(), Value::String(tenant.clone())));
            }
            let result = self.run_prepared(&conn, &cypher, params, "cross_boundary seed guard")?;
            if result.into_iter().next().is_none() {
                return Ok(Vec::new());
            }
        }

        for depth in 1..=hops {
            let mut next: std::collections::HashMap<String, CrossBoundaryPath> =
                std::collections::HashMap::new();
            for (current_id, current_path) in &frontier {
                let neighbours =
                    self.cross_boundary_neighbours(&conn, current_id, &rel_pattern, &tenants)?;
                for nb in neighbours {
                    // Acyclic guard: skip nodes already on this path. Compares
                    // whole node-ids (not " -> "-split substrings), so an id that
                    // contains the separator cannot trigger a false revisit.
                    if current_path.visits(&nb.node_id) {
                        continue;
                    }
                    let extended = current_path.extend(&nb.node_id);
                    let new_path = extended.display.clone();
                    let confidence = nb.weight
                        * crate::graph::cross_boundary::edge_type_base_confidence(&nb.edge_type);
                    // Record the shallowest reach; on equal depth keep the
                    // higher-confidence reach (matches SQLite MIN depth / MAX conf).
                    match reached.entry(nb.node_id.clone()) {
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert(TraversalNode {
                                node_id: nb.node_id.clone(),
                                symbol_name: nb.symbol_name,
                                symbol_type: nb.symbol_type,
                                file_path: nb.file_path,
                                edge_type: nb.edge_type,
                                depth,
                                path: new_path.clone(),
                                tenant_id: nb.tenant_id,
                                edge_confidence: confidence,
                            });
                        }
                        std::collections::hash_map::Entry::Occupied(mut o) => {
                            let existing = o.get_mut();
                            if depth < existing.depth
                                || (depth == existing.depth
                                    && confidence > existing.edge_confidence)
                            {
                                existing.depth = depth;
                                existing.edge_type = nb.edge_type;
                                existing.edge_confidence = confidence;
                                existing.path = new_path.clone();
                                existing.tenant_id = nb.tenant_id;
                            }
                        }
                    }
                    // Continue BFS from the first (shortest) path to this node.
                    next.entry(nb.node_id.clone()).or_insert(extended);
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }

        let results: Vec<TraversalNode> = reached.into_values().collect();
        Ok(apply_fan_out_caps(results, &self.graph_rag))
    }
}
