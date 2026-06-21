//! `ladybug_store/traversal.rs` — private BFS helpers for path-finding and
//! cross-boundary traversal.
//!
//! These methods are inherent `impl LadybugGraphStore` helpers: they are called
//! by the public `do_*` read methods in `query.rs` and are never part of the
//! `GraphStore` trait interface. Keeping them here shortens `query.rs` (which
//! owns all trait-delegated read methods) while grouping the BFS machinery in
//! one place for reviewability.
//!
//! Methods in this file:
//! - `cross_boundary_neighbours` — 1-hop bidirectional neighbour query for BFS
//! - `find_path_self`           — base-case: source == target
//! - `reconstruct_path`         — BFS path Vec<PathNode> → Vec<TraversalNode>
//! - `fetch_path_node`          — seed fetch for BFS source node
//! - `find_path_neighbours`     — batched 1-hop neighbour query for whole frontier

use lbug::{Connection, Value};

use crate::graph::{schema::GraphDbResult, TraversalNode};

use super::helpers::{tenant_param_list, value_to_string, CrossBoundaryNeighbour, PathNode};
use super::init::LadybugGraphStore;

impl LadybugGraphStore {
    /// Fetch the direct (1-hop) neighbours of `current_id` in BOTH directions
    /// over the allowed `rel_pattern`, keeping only neighbours whose tenant is
    /// in `tenants`. Returns the reached node plus the reaching edge's type and
    /// weight so the caller can score the hop.
    pub(super) fn cross_boundary_neighbours(
        &self,
        conn: &Connection<'_>,
        current_id: &str,
        rel_pattern: &str,
        tenants: &[String],
    ) -> GraphDbResult<Vec<CrossBoundaryNeighbour>> {
        // Parameterized tenant IN-list: `[$t0,$t1,...]` with one bound param per
        // tenant, so tenant ids are never interpolated into the query text
        // (no Cypher string-literal escaping pitfalls).
        let (tenant_list, tenant_names) = tenant_param_list(tenants.len());

        let mut out = Vec::new();

        // Outgoing: (current)-[r]->(n); Incoming: (current)<-[r]-(n).
        let patterns = [
            format!("(c:GraphNode {{node_id: $cid}})-[r:{rel_pattern}]->(n:GraphNode)"),
            format!("(c:GraphNode {{node_id: $cid}})<-[r:{rel_pattern}]-(n:GraphNode)"),
        ];
        for pattern in patterns {
            let cypher = format!(
                "MATCH {pattern} \
                 WHERE n.tenant_id IN {tenant_list} \
                 RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path, \
                        n.tenant_id, label(r), r.weight"
            );
            let mut params: Vec<(&str, Value)> = Vec::with_capacity(tenants.len() + 1);
            params.push(("cid", Value::String(current_id.to_string())));
            for (name, tenant) in tenant_names.iter().zip(tenants.iter()) {
                params.push((name.as_str(), Value::String(tenant.clone())));
            }
            let result = self.run_prepared(conn, &cypher, params, "cross_boundary_neighbours")?;
            for row in result {
                if row.len() < 7 {
                    continue;
                }
                let weight = match &row[6] {
                    Value::Double(d) => *d,
                    Value::Int64(i) => *i as f64,
                    _ => 1.0,
                };
                out.push(CrossBoundaryNeighbour {
                    node_id: value_to_string(&row[0]),
                    symbol_name: value_to_string(&row[1]),
                    symbol_type: value_to_string(&row[2]),
                    file_path: value_to_string(&row[3]),
                    tenant_id: value_to_string(&row[4]),
                    edge_type: value_to_string(&row[5]),
                    weight,
                });
            }
        }

        Ok(out)
    }

    // ---- find_path helpers ---------------------------------------------------

    /// Return `Ok(Some([source_node]))` for the self-path case, or `Ok(None)`
    /// when the source node does not exist in the tenant (matching SQLite: the
    /// recursive CTE base case joins `graph_nodes`, so a missing node yields
    /// no rows → None).
    pub(super) fn find_path_self(
        &self,
        tenant_id: &str,
        node_id: &str,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        let conn = self.connect()?;
        let cypher = "MATCH (n:GraphNode {node_id: $id}) \
                      WHERE n.tenant_id = $tid \
                      RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path";
        let result = self.run_prepared(
            &conn,
            cypher,
            vec![
                ("id", Value::String(node_id.to_string())),
                ("tid", Value::String(tenant_id.to_string())),
            ],
            "find_path_self",
        )?;

        for row in result {
            if row.len() < 4 {
                continue;
            }
            return Ok(Some(vec![TraversalNode {
                node_id: value_to_string(&row[0]),
                symbol_name: value_to_string(&row[1]),
                symbol_type: value_to_string(&row[2]),
                file_path: value_to_string(&row[3]),
                edge_type: String::new(),
                depth: 0,
                path: String::new(),
                tenant_id: tenant_id.to_string(),
                edge_confidence: 1.0,
            }]));
        }
        Ok(None)
    }

    /// Turn an already-traversed BFS path into `TraversalNode`s, with `depth` =
    /// index in the path.
    ///
    /// The forward BFS in [`find_path`](Self::do_find_path) already fetched every
    /// node's `(symbol_name, symbol_type, file_path)` columns when it expanded
    /// the frontier and carries them on each [`PathNode`]. Reconstruction
    /// therefore reads straight from memory and issues **no** further queries.
    /// Previously this re-queried every node on the path — a second wave of
    /// per-node FFI lookups for nodes the BFS had already seen (the N+1 in
    /// CR-011); capturing the columns during the forward pass removes it.
    ///
    /// There is no "node vanished mid-path" branch anymore: the columns were
    /// captured at traversal time, so they cannot disappear between the BFS and
    /// reconstruction (the old per-node re-query could race a concurrent delete;
    /// reading from the captured path cannot).
    pub(super) fn reconstruct_path(
        &self,
        tenant_id: &str,
        path: &[PathNode],
    ) -> Vec<TraversalNode> {
        path.iter()
            .enumerate()
            .map(|(depth, node)| TraversalNode {
                node_id: node.node_id.clone(),
                symbol_name: node.symbol_name.clone(),
                symbol_type: node.symbol_type.clone(),
                file_path: node.file_path.clone(),
                edge_type: String::new(),
                depth: depth as u32,
                path: String::new(),
                tenant_id: tenant_id.to_string(),
                edge_confidence: 1.0,
            })
            .collect()
    }

    /// Fetch one node's identity columns by id within `tenant_id`, returning
    /// `None` when it does not exist. Used to seed the `find_path` BFS with the
    /// source node (the only path entry never seen as a neighbour row). One FFI
    /// round-trip, through the [`run_prepared`](Self::run_prepared) choke-point.
    pub(super) fn fetch_path_node(
        &self,
        conn: &Connection<'_>,
        tenant_id: &str,
        node_id: &str,
    ) -> GraphDbResult<Option<PathNode>> {
        let cypher = "MATCH (n:GraphNode {node_id: $id}) \
                      WHERE n.tenant_id = $tid \
                      RETURN n.node_id, n.symbol_name, n.symbol_type, n.file_path";
        let result = self.run_prepared(
            conn,
            cypher,
            vec![
                ("id", Value::String(node_id.to_string())),
                ("tid", Value::String(tenant_id.to_string())),
            ],
            "find_path source seed",
        )?;
        for row in result {
            if row.len() < 4 {
                continue;
            }
            return Ok(Some(PathNode {
                node_id: value_to_string(&row[0]),
                symbol_name: value_to_string(&row[1]),
                symbol_type: value_to_string(&row[2]),
                file_path: value_to_string(&row[3]),
            }));
        }
        Ok(None)
    }

    /// Fetch the 1-hop outgoing neighbours of EVERY node in `current_ids` over
    /// `rel_pattern`, in a single query, grouped by their source node-id.
    ///
    /// This collapses the `find_path` per-hop N+1 (CR-011): instead of one
    /// neighbour query per frontier path, one bound IN-list query returns all
    /// neighbours for the whole frontier at once. `a.node_id` is returned so the
    /// caller can route each neighbour back to the path(s) it extends. The ids
    /// are bound as parameters (`[$n0,$n1,...]`) — never interpolated — reusing
    /// the same mechanism as [`cross_boundary_neighbours`](Self::cross_boundary_neighbours).
    pub(super) fn find_path_neighbours(
        &self,
        conn: &Connection<'_>,
        tenant_id: &str,
        current_ids: &[String],
        rel_pattern: &str,
    ) -> GraphDbResult<std::collections::HashMap<String, Vec<PathNode>>> {
        let mut by_source: std::collections::HashMap<String, Vec<PathNode>> =
            std::collections::HashMap::new();
        if current_ids.is_empty() {
            return Ok(by_source);
        }

        // Bind one parameter per source id: `a.node_id IN [$n0,$n1,...]`.
        let names: Vec<String> = (0..current_ids.len()).map(|i| format!("n{i}")).collect();
        let id_list = format!(
            "[{}]",
            names
                .iter()
                .map(|n| format!("${n}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let cypher = format!(
            "MATCH (a:GraphNode)-[:{rel_pattern}*1..1]->(b:GraphNode) \
             WHERE a.node_id IN {id_list} AND b.tenant_id = $tid \
             RETURN a.node_id, b.node_id, b.symbol_name, b.symbol_type, b.file_path"
        );
        let mut params: Vec<(&str, Value)> = Vec::with_capacity(current_ids.len() + 1);
        params.push(("tid", Value::String(tenant_id.to_string())));
        for (name, id) in names.iter().zip(current_ids.iter()) {
            params.push((name.as_str(), Value::String(id.clone())));
        }

        let result = self.run_prepared(conn, &cypher, params, "find_path neighbours")?;
        for row in result {
            if row.len() < 5 {
                continue;
            }
            let source_id = value_to_string(&row[0]);
            by_source.entry(source_id).or_default().push(PathNode {
                node_id: value_to_string(&row[1]),
                symbol_name: value_to_string(&row[2]),
                symbol_type: value_to_string(&row[3]),
                file_path: value_to_string(&row[4]),
            });
        }
        Ok(by_source)
    }
}
