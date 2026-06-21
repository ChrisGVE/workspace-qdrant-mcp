//! `ladybug_store/store/mutate.rs` — write-path methods for `LadybugGraphStore`.
//!
//! Every method in this file mutates graph state: upsert/insert of nodes and
//! edges, deletion of edges or entire tenants, narrative node cleanup,
//! orphan pruning, and stub-edge resolution. All writes serialize through
//! `self.write_lock` (a `Mutex<()>`) and flow through the FFI choke-point
//! helpers in `init.rs` (`connect`, `run_prepared`). The two low-level DML
//! primitives (`upsert_node_with_conn`, `insert_edge_with_conn`) live in
//! `init.rs` alongside the other FFI boundary helpers so that the full FFI
//! surface stays collocated. No reads outside of the stub-resolution pre-flight
//! are performed here; those live in `query.rs`.
//!
//! These methods are inherent `impl LadybugGraphStore` rather than a trait
//! impl. The single `impl GraphStore for LadybugGraphStore` block lives in
//! `store/mod.rs` and delegates to the `pub(super)` methods here.

use lbug::Value;
use tracing::debug;

use crate::graph::{compute_edge_id, schema::GraphDbResult, EdgeType, GraphEdge, GraphNode};

use super::helpers::ALL_REL_TYPES;
use super::init::LadybugGraphStore;

// ---- Write-path inherent methods (delegated from GraphStore trait impl) ------

impl LadybugGraphStore {
    pub(super) async fn do_upsert_node(&self, node: &GraphNode) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        self.upsert_node_with_conn(&conn, node)
    }

    pub(super) async fn do_upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for node in nodes {
            self.upsert_node_with_conn(&conn, node)?;
        }
        debug!("Upserted {} graph nodes (LadybugDB)", nodes.len());
        Ok(())
    }

    pub(super) async fn do_insert_edge(&self, edge: &GraphEdge) -> GraphDbResult<()> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        self.insert_edge_with_conn(&conn, edge)
    }

    pub(super) async fn do_insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        for edge in edges {
            self.insert_edge_with_conn(&conn, edge)?;
        }
        debug!("Inserted {} graph edges (LadybugDB)", edges.len());
        Ok(())
    }

    pub(super) async fn do_delete_edges_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Must delete from each rel table separately (LadybugDB requires
        // typed MATCH patterns per T34 findings).
        for rel_type in ALL_REL_TYPES {
            let cypher = format!(
                "MATCH (a:GraphNode)-[r:{rel_type}]->(b:GraphNode) \
                 WHERE r.tenant_id = $tid AND r.source_file = $fp \
                 DELETE r"
            );
            self.run_prepared(
                &conn,
                &cypher,
                vec![
                    ("tid", Value::String(tenant_id.to_string())),
                    ("fp", Value::String(file_path.to_string())),
                ],
                "delete_edges",
            )?;
        }

        debug!(
            "Deleted edges for file {} in tenant {} (LadybugDB)",
            file_path, tenant_id
        );
        // LadybugDB does not return affected row counts from DELETE
        Ok(0)
    }

    pub(super) async fn do_delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Delete all edges first (per rel table)
        for rel_type in ALL_REL_TYPES {
            let cypher = format!(
                "MATCH (a:GraphNode)-[r:{rel_type}]->(b:GraphNode) \
                 WHERE r.tenant_id = $tid DELETE r"
            );
            self.run_prepared(
                &conn,
                &cypher,
                vec![("tid", Value::String(tenant_id.to_string()))],
                "delete_tenant edges",
            )?;
        }

        // Then delete all nodes for this tenant
        self.run_prepared(
            &conn,
            "MATCH (n:GraphNode) WHERE n.tenant_id = $tid DELETE n",
            vec![("tid", Value::String(tenant_id.to_string()))],
            "delete_tenant nodes",
        )?;

        debug!("Deleted tenant {} data (LadybugDB)", tenant_id);
        Ok(0)
    }

    pub(super) async fn do_prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Find nodes with no incident edges across any rel table, then delete.
        // LadybugDB's EXISTS subquery checks all rel tables via the union pattern.
        let all_rels = ALL_REL_TYPES.join("|");
        let cypher = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid \
             AND NOT EXISTS {{ MATCH (n)-[:{all_rels}]-() }} \
             DELETE n"
        );

        // This may not be supported by all LadybugDB versions; if it fails,
        // fall back gracefully.
        let result = self
            .run_prepared(
                &conn,
                &cypher,
                vec![("tid", Value::String(tenant_id.to_string()))],
                "prune_orphans",
            )
            .map(|_| ());

        if let Err(e) = result {
            debug!("prune_orphans subquery not supported, skipping: {}", e);
        }

        Ok(0)
    }

    pub(super) async fn do_resolve_stub_edges(&self, tenant_id: &str) -> GraphDbResult<u64> {
        use std::collections::HashMap;

        // Only structural code rel types carry tree-sitter name-only stubs.
        const CODE_REL_TYPES: &[&str] = &[
            "CALLS",
            "CONTAINS",
            "IMPORTS",
            "USES_TYPE",
            "EXTENDS",
            "IMPLEMENTS",
        ];

        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;

        // Real candidate nodes (resolved file_path, not file-typed), indexed by
        // symbol_name -> [(node_id, file_path)].
        let mut by_name: HashMap<String, Vec<(String, String)>> = HashMap::new();
        {
            let result = self.run_prepared(
                &conn,
                "MATCH (n:GraphNode) \
                 WHERE n.tenant_id = $tid AND n.file_path <> '' AND n.symbol_type <> 'file' \
                 RETURN n.node_id, n.symbol_name, n.file_path",
                vec![("tid", Value::String(tenant_id.to_string()))],
                "stub candidates",
            )?;
            for row in result {
                if row.len() < 3 {
                    continue;
                }
                let nid = super::helpers::value_to_string(&row[0]);
                let name = super::helpers::value_to_string(&row[1]);
                let fp = super::helpers::value_to_string(&row[2]);
                by_name.entry(name).or_default().push((nid, fp));
            }
        }

        let mut repointed: u64 = 0;

        for rel_type in CODE_REL_TYPES {
            let Some(edge_type) = EdgeType::from_str(rel_type) else {
                continue;
            };

            // Dangling edges of this type: target is a stub (empty file_path).
            // Collect rows first so the connection is free for the follow-up
            // delete/create mutations.
            let find = format!(
                "MATCH (s:GraphNode)-[e:{rel_type}]->(t:GraphNode) \
                 WHERE e.tenant_id = $tid AND t.tenant_id = $tid \
                       AND (t.file_path = '' OR t.file_path IS NULL) \
                 RETURN s.node_id, e.edge_id, e.source_file, e.weight, e.metadata_json, \
                        t.symbol_name"
            );
            let rows: Vec<(String, String, String, f64, String, String)> = {
                let result = self.run_prepared(
                    &conn,
                    &find,
                    vec![("tid", Value::String(tenant_id.to_string()))],
                    &format!("stub dangling ({rel_type})"),
                )?;
                result
                    .into_iter()
                    .filter_map(|row| {
                        if row.len() < 6 {
                            return None;
                        }
                        Some((
                            super::helpers::value_to_string(&row[0]),
                            super::helpers::value_to_string(&row[1]),
                            super::helpers::value_to_string(&row[2]),
                            super::helpers::value_to_f64(&row[3]),
                            super::helpers::value_to_string(&row[4]),
                            super::helpers::value_to_string(&row[5]),
                        ))
                    })
                    .collect()
            };

            for (source_node_id, old_edge_id, source_file, weight, metadata_json, target_name) in
                rows
            {
                let Some(candidates) = by_name.get(&target_name) else {
                    continue; // external/stdlib — no project node with this name.
                };
                // Prefer a definition in the caller's own file; else require a
                // unique tenant-wide match. Ambiguous names are skipped.
                let chosen: Option<&String> = candidates
                    .iter()
                    .find(|(_, fp)| *fp == source_file)
                    .map(|(nid, _)| nid)
                    .or_else(|| {
                        if candidates.len() == 1 {
                            Some(&candidates[0].0)
                        } else {
                            None
                        }
                    });
                let Some(new_target) = chosen else {
                    continue;
                };
                if &source_node_id == new_target {
                    continue; // skip self-loops
                }
                let new_edge_id = compute_edge_id(&source_node_id, new_target, edge_type);

                // Repoint. Kuzu has no multi-statement transaction here, so CREATE
                // the repointed rel FIRST, then DELETE the old dangling one: if the
                // CREATE fails we propagate the error with the old edge still
                // intact (the next sweep retries) rather than losing connectivity.
                // The `branch` scope is not carried — the LadybugDB rel tables have
                // no branch column (this backend does not track edge branches).
                let ins = format!(
                    "MATCH (a:GraphNode {{node_id: $src}}), (b:GraphNode {{node_id: $dst}}) \
                     CREATE (a)-[:{rel_type} {{weight: $weight, source_file: $sf, \
                     edge_id: $neid, tenant_id: $tid, metadata_json: $md}}]->(b)"
                );
                self.run_prepared(
                    &conn,
                    &ins,
                    vec![
                        ("src", Value::String(source_node_id.clone())),
                        ("dst", Value::String(new_target.clone())),
                        ("weight", Value::Double(weight)),
                        ("sf", Value::String(source_file.clone())),
                        ("neid", Value::String(new_edge_id)),
                        ("tid", Value::String(tenant_id.to_string())),
                        ("md", Value::String(metadata_json.clone())),
                    ],
                    &format!("stub create ({rel_type})"),
                )?;

                let del = format!(
                    "MATCH (a:GraphNode)-[r:{rel_type}]->(b:GraphNode) \
                     WHERE r.edge_id = $eid AND r.tenant_id = $tid DELETE r"
                );
                self.run_prepared(
                    &conn,
                    &del,
                    vec![
                        ("eid", Value::String(old_edge_id.clone())),
                        ("tid", Value::String(tenant_id.to_string())),
                    ],
                    &format!("stub delete ({rel_type})"),
                )?;

                repointed += 1;
            }
        }

        // Prune stub nodes (empty file_path) that are now edgeless. Mirrors
        // `prune_orphans` but scoped to stubs; tolerated if the EXISTS subquery
        // is unsupported by the LadybugDB version.
        let all_rels = ALL_REL_TYPES.join("|");
        let prune = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid AND (n.file_path = '' OR n.file_path IS NULL) \
             AND NOT EXISTS {{ MATCH (n)-[r:{all_rels}]-() WHERE r.tenant_id = $tid }} \
             DELETE n"
        );
        let prune_res = self
            .run_prepared(
                &conn,
                &prune,
                vec![("tid", Value::String(tenant_id.to_string()))],
                "stub prune",
            )
            .map(|_| ());
        if let Err(e) = prune_res {
            debug!("stub-node prune subquery not supported, skipping: {}", e);
        }

        debug!(
            "Resolved {} stub edges for tenant {} (LadybugDB)",
            repointed, tenant_id
        );
        Ok(repointed)
    }

    pub(super) async fn do_resolve_all_stub_edges(&self) -> GraphDbResult<u64> {
        let mut total: u64 = 0;
        // Each tenant takes the write lock independently via do_resolve_stub_edges,
        // so the lock is released between tenants.
        for tenant in self.do_graph_tenants().await? {
            total += self.do_resolve_stub_edges(&tenant).await?;
        }
        Ok(total)
    }

    pub(super) async fn do_delete_narrative_nodes_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        let _lock = self.write_lock.lock().await;
        let conn = self.connect()?;
        let type_list = super::helpers::NARRATIVE_FILE_NODE_TYPES
            .iter()
            .map(|t| format!("'{t}'"))
            .collect::<Vec<_>>()
            .join(",");

        // Edge-first deletion (mirrors `delete_tenant`). LadybugDB enforces
        // node<->edge referential integrity, so a narrative node that is the
        // endpoint of any edge (e.g. an EXPLAINS edge to the code it documents)
        // cannot be deleted while that edge exists. Delete every incident edge
        // across all rel types -- in both directions, since a narrative node
        // may be the source (EXPLAINS) or the target (REFERENCES_DOC) -- before
        // removing the nodes themselves.
        let node_predicate = format!(
            "m.tenant_id = $tid AND m.file_path = $fp \
             AND m.symbol_type IN [{type_list}]"
        );
        for rel_type in ALL_REL_TYPES {
            // Outgoing (narrative -> other) and incoming (other -> narrative).
            let out_cypher = format!(
                "MATCH (m:GraphNode)-[r:{rel_type}]->(:GraphNode) \
                 WHERE {node_predicate} DELETE r"
            );
            let in_cypher = format!(
                "MATCH (:GraphNode)-[r:{rel_type}]->(m:GraphNode) \
                 WHERE {node_predicate} DELETE r"
            );
            for cypher in [out_cypher, in_cypher] {
                self.run_prepared(
                    &conn,
                    &cypher,
                    vec![
                        ("tid", Value::String(tenant_id.to_string())),
                        ("fp", Value::String(file_path.to_string())),
                    ],
                    "delete_narrative_nodes_by_file edges",
                )?;
            }
        }

        // Now the (edge-free) narrative nodes can be removed.
        let cypher = format!(
            "MATCH (n:GraphNode) \
             WHERE n.tenant_id = $tid AND n.file_path = $fp \
                   AND n.symbol_type IN [{type_list}] \
             DELETE n"
        );
        self.run_prepared(
            &conn,
            &cypher,
            vec![
                ("tid", Value::String(tenant_id.to_string())),
                ("fp", Value::String(file_path.to_string())),
            ],
            "delete_narrative_nodes_by_file",
        )?;
        // LadybugDB does not return affected row counts from DELETE.
        Ok(0)
    }
}
