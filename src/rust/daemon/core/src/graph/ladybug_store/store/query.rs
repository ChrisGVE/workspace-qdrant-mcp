//! `ladybug_store/store/query.rs` — simpler read-path `GraphStore` methods.
//!
//! Contains all read methods except the two BFS-driver methods (`do_find_path`,
//! `do_query_cross_boundary`) which live in `path.rs` due to their size. The
//! private BFS primitives (`cross_boundary_neighbours`, `find_path_self`, etc.)
//! are in `traversal.rs` and are reached through `self.*`.
//!
//! Methods in this file: `do_query_related`, `do_impact_analysis`, `do_stats`,
//! `do_graph_tenants`, `do_query_code_symbols`, `do_fetch_node_metadata`,
//! `do_export_adjacency`, `do_export_nodes_for_tenant`, `do_export_edges_for_tenant`.

use lbug::Value;
use tracing::warn;

use crate::graph::{
    is_cross_branch,
    schema::{GraphDbError, GraphDbResult},
    AdjacencyExport, EdgeType, GraphEdge, GraphNode, GraphStats, ImpactNode, ImpactReport,
    NodeMetadata, SymbolRow, TraversalNode,
};

use super::helpers::{
    value_to_f64, value_to_i64, value_to_string, ALL_REL_TYPES, CODE_SYMBOL_TYPES,
    MAX_FRONTIER_PATHS,
};
use super::init::LadybugGraphStore;

// ---- Read-path inherent methods ---------------------------------------------
//
// Each method is `pub(super)` so `store/mod.rs` can delegate from the single
// `impl GraphStore for LadybugGraphStore` block. The `do_` prefix avoids name
// collisions with the trait method names used in that delegating impl.
//
// The two BFS-intensive methods `do_find_path` and `do_query_cross_boundary`
// live in `path.rs` to keep both files under the 500-line limit.

impl LadybugGraphStore {
    /// LadybugDB backend: branch scoping is not implemented. A branch-scoped
    /// query (`branch = Some(name)` with `name != "*"`) returns a
    /// [`GraphDbError::BranchScopingUnsupported`] (surfaced as gRPC
    /// `Status::unimplemented`) rather than silently returning cross-branch
    /// results. `None` or `Some("*")` behave as before (no filtering).
    pub(super) async fn do_query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        if !is_cross_branch(branch) {
            return Err(GraphDbError::BranchScopingUnsupported(
                branch.unwrap_or_default().to_string(),
            ));
        }
        if max_hops == 0 {
            return Ok(Vec::new());
        }
        let conn = self.connect()?;

        // Rel type pattern. Rel types come from the EdgeType enum (static
        // literals), so string interpolation is injection-safe.
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => ALL_REL_TYPES.join("|"),
        };

        // SQLite reports the TRUE minimum depth per reached node. Kuzu's
        // variable-length `*1..n` pattern cannot expose per-row depth directly,
        // so we query each exact hop length `*k..k` in ascending order and keep
        // the first (minimum) depth at which a node is reached.
        //
        // CR-011 — why this stays an N-query loop on lbug 0.14.1: the ideal fix
        // is a single `... ALL SHORTEST 1..max_hops ...` query that returns each
        // node at its shortest length in one round-trip. lbug 0.14.1's Cypher
        // parser REJECTS the `ALL SHORTEST` keyword ("Parser exception: expected
        // rule oC_SingleQuery" at the `ALL` token), and the bare default `*1..n`
        // is VARIABLE_LENGTH_WALK — it enumerates every walk (cyclic revisits
        // included), which both loses the per-row minimum depth and risks an
        // exponential row blow-up that defeats the CR-010 frontier cap. With
        // neither a working shortest-path keyword nor a safe single-query form on
        // this version, the per-hop `*k..k` loop is the correct shape: it is
        // O(max_hops) FFI calls (small, bounded) and gives exact min-depth. Revisit
        // if a future lbug accepts `ALL SHORTEST`.
        let mut by_node: std::collections::HashMap<String, TraversalNode> =
            std::collections::HashMap::new();

        // Labeled so the frontier cap below can stop the whole traversal.
        'hops: for hop in 1..=max_hops {
            // Kuzu does not allow projecting a property off an indexed element
            // of a recursive-rel list (`rels[i].edge_type`), so we return only
            // node identity. `edge_type` is left empty for the per-hop result;
            // the SQLite backend populates it, but cross-backend conformance is
            // asserted on the (node_id, depth) and identity maps, not on the
            // reaching edge type.
            let cypher = format!(
                "MATCH (start:GraphNode {{node_id: $start_id}})\
                 -[rels:{rel_pattern}*{hop}..{hop}]->(related:GraphNode) \
                 WHERE related.tenant_id = $tid \
                 RETURN related.node_id, related.symbol_name, related.symbol_type, \
                        related.file_path"
            );

            let result = self.run_prepared(
                &conn,
                &cypher,
                vec![
                    ("start_id", Value::String(node_id.to_string())),
                    ("tid", Value::String(tenant_id.to_string())),
                ],
                &format!("query_related (hop {hop})"),
            )?;

            for row in result {
                if row.len() < 4 {
                    continue;
                }
                let nid = value_to_string(&row[0]);
                // Do not overwrite a shallower entry recorded at a smaller hop.
                by_node.entry(nid.clone()).or_insert_with(|| TraversalNode {
                    node_id: nid,
                    symbol_name: value_to_string(&row[1]),
                    symbol_type: value_to_string(&row[2]),
                    file_path: value_to_string(&row[3]),
                    edge_type: String::new(),
                    depth: hop,
                    path: String::new(),
                    tenant_id: tenant_id.to_string(),
                    edge_confidence: 1.0,
                });

                // Frontier cap (CR-010): a dense graph can return an enormous
                // result set. Once we have collected the cap, stop traversing
                // and return what we have rather than risking OOM.
                if by_node.len() >= MAX_FRONTIER_PATHS {
                    warn!(
                        "query_related result reached cap ({}); returning partial set",
                        MAX_FRONTIER_PATHS
                    );
                    break 'hops;
                }
            }
        }
        let mut nodes: Vec<TraversalNode> = by_node.into_values().collect();
        nodes.sort_by(|a, b| {
            a.depth
                .cmp(&b.depth)
                .then(a.symbol_name.cmp(&b.symbol_name))
        });
        Ok(nodes)
    }

    /// LadybugDB backend: branch scoping is not implemented. A branch-scoped
    /// query (`branch = Some(name)` with `name != "*"`) returns a
    /// [`GraphDbError::BranchScopingUnsupported`] (surfaced as gRPC
    /// `Status::unimplemented`) rather than silently returning cross-branch
    /// results. `None` or `Some("*")` behave as before (no filtering).
    pub(super) async fn do_impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        if !is_cross_branch(branch) {
            return Err(GraphDbError::BranchScopingUnsupported(
                branch.unwrap_or_default().to_string(),
            ));
        }
        let conn = self.connect()?;

        // Reverse traversal: find all callers up to 3 hops (matching SQLite
        // implementation's depth limit).
        // We query each rel type separately and merge to get proper impact_type.
        let rel_pattern = ALL_REL_TYPES.join("|");

        let (cypher, params) = if let Some(fp) = file_path {
            let c = format!(
                "MATCH (start:GraphNode)<-[r:{rel_pattern}*1..3]-(caller:GraphNode) \
                 WHERE start.symbol_name = $sym AND start.tenant_id = $tid \
                       AND start.file_path = $fp \
                 RETURN DISTINCT caller.node_id, caller.symbol_name, caller.file_path"
            );
            let p: Vec<(&str, Value)> = vec![
                ("sym", Value::String(symbol_name.to_string())),
                ("tid", Value::String(tenant_id.to_string())),
                ("fp", Value::String(fp.to_string())),
            ];
            (c, p)
        } else {
            let c = format!(
                "MATCH (start:GraphNode)<-[r:{rel_pattern}*1..3]-(caller:GraphNode) \
                 WHERE start.symbol_name = $sym AND start.tenant_id = $tid \
                 RETURN DISTINCT caller.node_id, caller.symbol_name, caller.file_path"
            );
            let p: Vec<(&str, Value)> = vec![
                ("sym", Value::String(symbol_name.to_string())),
                ("tid", Value::String(tenant_id.to_string())),
            ];
            (c, p)
        };

        let result = self.run_prepared(&conn, &cypher, params, "impact_analysis")?;

        let mut impacted = Vec::new();
        for row in result {
            if row.len() >= 3 {
                impacted.push(ImpactNode {
                    node_id: value_to_string(&row[0]),
                    symbol_name: value_to_string(&row[1]),
                    file_path: value_to_string(&row[2]),
                    impact_type: "caller".to_string(),
                    distance: 1,
                });
            }
        }

        let total = impacted.len() as u32;
        Ok(ImpactReport {
            symbol_name: symbol_name.to_string(),
            impacted_nodes: impacted,
            total_impacted: total,
        })
    }

    /// LadybugDB backend: branch scoping is not implemented. A branch-scoped
    /// query (`branch = Some(name)` with `name != "*"`) returns a
    /// [`GraphDbError::BranchScopingUnsupported`] (surfaced as gRPC
    /// `Status::unimplemented`) rather than silently returning cross-branch
    /// results. `None` or `Some("*")` behave as before (no filtering).
    pub(super) async fn do_stats(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<GraphStats> {
        if !is_cross_branch(branch) {
            return Err(GraphDbError::BranchScopingUnsupported(
                branch.unwrap_or_default().to_string(),
            ));
        }
        let conn = self.connect()?;

        // --- Node counts by type ---
        let mut nodes_by_type = std::collections::HashMap::new();
        let mut total_nodes = 0u64;

        let (cypher, params): (String, Vec<(&str, Value)>) = match tenant_id {
            Some(tid) => (
                "MATCH (n:GraphNode) WHERE n.tenant_id = $tid \
                 RETURN n.symbol_type, count(n)"
                    .to_string(),
                vec![("tid", Value::String(tid.to_string()))],
            ),
            None => (
                "MATCH (n:GraphNode) RETURN n.symbol_type, count(n)".to_string(),
                vec![],
            ),
        };

        let result = self.run_prepared(&conn, &cypher, params, "stats nodes")?;

        for row in result {
            if row.len() >= 2 {
                let stype = value_to_string(&row[0]);
                let cnt = value_to_i64(&row[1]) as u64;
                total_nodes += cnt;
                nodes_by_type.insert(stype, cnt);
            }
        }

        // --- Edge counts by type ---
        let mut edges_by_type = std::collections::HashMap::new();
        let mut total_edges = 0u64;

        for rel_type in ALL_REL_TYPES {
            let (cypher, params): (String, Vec<(&str, Value)>) = match tenant_id {
                Some(tid) => (
                    format!("MATCH ()-[r:{rel_type}]->() WHERE r.tenant_id = $tid RETURN count(r)"),
                    vec![("tid", Value::String(tid.to_string()))],
                ),
                None => (
                    format!("MATCH ()-[r:{rel_type}]->() RETURN count(r)"),
                    vec![],
                ),
            };

            let result =
                self.run_prepared(&conn, &cypher, params, &format!("stats edges ({rel_type})"))?;

            for row in result {
                if !row.is_empty() {
                    let cnt = value_to_i64(&row[0]) as u64;
                    if cnt > 0 {
                        total_edges += cnt;
                        edges_by_type.insert(rel_type.to_string(), cnt);
                    }
                }
            }
        }

        Ok(GraphStats {
            total_nodes,
            total_edges,
            nodes_by_type,
            edges_by_type,
        })
    }

    pub(super) async fn do_graph_tenants(&self) -> GraphDbResult<Vec<String>> {
        let conn = self.connect()?;
        let result = self.run_prepared(
            &conn,
            "MATCH (n:GraphNode) RETURN DISTINCT n.tenant_id",
            vec![],
            "graph_tenants",
        )?;
        let mut out = Vec::new();
        for row in result {
            if let Some(v) = row.first() {
                let t = value_to_string(v);
                if !t.is_empty() {
                    out.push(t);
                }
            }
        }
        Ok(out)
    }

    pub(super) async fn do_query_code_symbols(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<Vec<SymbolRow>> {
        let conn = self.connect()?;
        let type_list = CODE_SYMBOL_TYPES
            .iter()
            .map(|t| format!("'{t}'"))
            .collect::<Vec<_>>()
            .join(",");
        let cypher = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid AND n.symbol_type IN [{type_list}] \
                   AND n.symbol_name <> '' \
             RETURN n.symbol_name, n.node_id, n.file_path"
        );
        let result = self.run_prepared(
            &conn,
            &cypher,
            vec![("tid", Value::String(tenant_id.to_string()))],
            "query_code_symbols",
        )?;
        let mut rows = Vec::new();
        for row in result {
            if row.len() >= 3 {
                rows.push(SymbolRow {
                    symbol_name: value_to_string(&row[0]),
                    node_id: value_to_string(&row[1]),
                    file_path: value_to_string(&row[2]),
                });
            }
        }
        Ok(rows)
    }

    pub(super) async fn do_fetch_node_metadata(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<std::collections::HashMap<String, NodeMetadata>> {
        let conn = self.connect()?;
        let cypher = "MATCH (n:GraphNode) WHERE n.tenant_id = $tid \
                      RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path";
        let result = self.run_prepared(
            &conn,
            cypher,
            vec![("tid", Value::String(tenant_id.to_string()))],
            "fetch_node_metadata",
        )?;
        let mut map = std::collections::HashMap::new();
        for row in result {
            if row.len() >= 4 {
                map.insert(
                    value_to_string(&row[0]),
                    NodeMetadata {
                        symbol_name: value_to_string(&row[1]),
                        symbol_type: value_to_string(&row[2]),
                        file_path: value_to_string(&row[3]),
                    },
                );
            }
        }
        Ok(map)
    }

    pub(super) async fn do_export_adjacency(
        &self,
        tenant_id: &str,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<AdjacencyExport> {
        use std::collections::HashMap;

        let conn = self.connect()?;

        // 1. Load all nodes for the tenant, sorted deterministically (DOM-01).
        let node_result = self.run_prepared(
            &conn,
            "MATCH (n:GraphNode) WHERE n.tenant_id = $tid \
             RETURN n.node_id ORDER BY n.node_id",
            vec![("tid", Value::String(tenant_id.to_string()))],
            "export_adjacency nodes",
        )?;

        let node_ids: Vec<String> = node_result
            .into_iter()
            .filter_map(|row| row.into_iter().next().map(|v| value_to_string(&v)))
            .collect();

        // Build a node_id→index map for O(1) edge lookups.
        let index_map: HashMap<String, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();

        // 2. Build the rel-type pattern.
        //
        //    Rel type names come from EdgeType::as_str() (compile-time constants),
        //    so string interpolation into the rel-pattern position is injection-safe
        //    (same convention as query_related, find_path, etc. in this backend).
        //
        //    NOTE: LadybugDB rel tables carry a `weight DOUBLE` property (confirmed
        //    in init_schema DDL). Weight is returned and stored directly; no default
        //    substitution is needed. This ensures cross-backend weight equivalence
        //    with the SQLite backend (relevant for the conformance suite, task 8).
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => types
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => ALL_REL_TYPES.join("|"),
        };

        // 3. Query edges: source node_id, target node_id, weight.
        let edge_cypher = format!(
            "MATCH (a:GraphNode)-[r:{rel_pattern}]->(b:GraphNode) \
             WHERE a.tenant_id = $tid \
             RETURN a.node_id, b.node_id, r.weight \
             ORDER BY a.node_id, b.node_id"
        );

        let edge_result = self.run_prepared(
            &conn,
            &edge_cypher,
            vec![("tid", Value::String(tenant_id.to_string()))],
            "export_adjacency edges",
        )?;

        // 4. Convert to indexed edges; skip orphan endpoints.
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for row in edge_result {
            if row.len() < 3 {
                continue;
            }
            let src = value_to_string(&row[0]);
            let tgt = value_to_string(&row[1]);
            let weight = value_to_f64(&row[2]);

            let (Some(&si), Some(&ti)) = (index_map.get(&src), index_map.get(&tgt)) else {
                // Orphan edge: at least one endpoint absent from the node list.
                continue;
            };
            edges.push((si, ti, weight));
        }

        Ok(AdjacencyExport { node_ids, edges })
    }

    /// Export all nodes for a tenant, ordered by node_id (DATA-05 content diff).
    ///
    /// Delegates to the migrator's Cypher-based exporter so that
    /// [`crate::graph::migrator::diff_graph_contents`] can compare this backend
    /// against SQLite through the trait. The export restores the empty-string
    /// point-id sentinel back to `None` for lossless round-tripping.
    pub(super) async fn do_export_nodes_for_tenant(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<Vec<GraphNode>> {
        crate::graph::migrator::export_nodes_ladybug(self, Some(tenant_id))
    }
    /// Export all edges for a tenant, ordered by edge_id (DATA-05 content diff).
    pub(super) async fn do_export_edges_for_tenant(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<Vec<GraphEdge>> {
        crate::graph::migrator::export_edges_ladybug(self, Some(tenant_id))
    }
}
