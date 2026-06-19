//! Export functions for extracting graph data from SQLite and LadybugDB backends.

use sqlx::{Row, SqlitePool};
use tracing::{info, warn};

use super::GraphSnapshot;
use crate::graph::schema::GraphDbResult;
use crate::graph::{EdgeType, GraphEdge, GraphNode, NodeType};

// ─── Export from SQLite ─────────────────────────────────────────────────

/// Export all nodes from SQLite, optionally filtered by tenant.
pub async fn export_nodes_sqlite(
    pool: &SqlitePool,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphNode>> {
    let rows = match tenant_id {
        Some(tid) => {
            sqlx::query(
                "SELECT node_id, tenant_id, symbol_name, symbol_type,
                        file_path, start_line, end_line, signature, language,
                        branches, qdrant_point_id, point_id_state
                 FROM graph_nodes WHERE tenant_id = ?1
                 ORDER BY node_id",
            )
            .bind(tid)
            .fetch_all(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT node_id, tenant_id, symbol_name, symbol_type,
                        file_path, start_line, end_line, signature, language,
                        branches, qdrant_point_id, point_id_state
                 FROM graph_nodes ORDER BY node_id",
            )
            .fetch_all(pool)
            .await?
        }
    };

    let nodes = rows
        .iter()
        .filter_map(|row| {
            let stype_str: String = row.get("symbol_type");
            let symbol_type = NodeType::from_str(&stype_str)?;
            Some(GraphNode {
                node_id: row.get("node_id"),
                tenant_id: row.get("tenant_id"),
                symbol_name: row.get("symbol_name"),
                symbol_type,
                file_path: row.get("file_path"),
                start_line: row.get::<Option<i64>, _>("start_line").map(|v| v as u32),
                end_line: row.get::<Option<i64>, _>("end_line").map(|v| v as u32),
                signature: row.get("signature"),
                language: row.get("language"),
                branches: row.get("branches"),
                qdrant_point_id: row.get("qdrant_point_id"),
                point_id_state: row
                    .get::<Option<String>, _>("point_id_state")
                    .unwrap_or_else(|| "none".to_string()),
            })
        })
        .collect::<Vec<_>>();

    info!("Exported {} nodes from SQLite", nodes.len());
    Ok(nodes)
}

/// Export all edges from SQLite, optionally filtered by tenant.
pub async fn export_edges_sqlite(
    pool: &SqlitePool,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphEdge>> {
    let rows = match tenant_id {
        Some(tid) => {
            sqlx::query(
                "SELECT edge_id, tenant_id, source_node_id, target_node_id,
                        edge_type, source_file, weight, metadata_json, branch
                 FROM graph_edges WHERE tenant_id = ?1
                 ORDER BY edge_id",
            )
            .bind(tid)
            .fetch_all(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT edge_id, tenant_id, source_node_id, target_node_id,
                        edge_type, source_file, weight, metadata_json, branch
                 FROM graph_edges ORDER BY edge_id",
            )
            .fetch_all(pool)
            .await?
        }
    };

    let mut edges = Vec::with_capacity(rows.len());
    let mut skipped = 0u64;

    for row in &rows {
        let etype_str: String = row.get("edge_type");
        match EdgeType::from_str(&etype_str) {
            Some(edge_type) => {
                edges.push(GraphEdge {
                    edge_id: row.get("edge_id"),
                    tenant_id: row.get("tenant_id"),
                    source_node_id: row.get("source_node_id"),
                    target_node_id: row.get("target_node_id"),
                    edge_type,
                    source_file: row.get("source_file"),
                    weight: row.get("weight"),
                    metadata_json: row.get("metadata_json"),
                    branch: row.get("branch"),
                });
            }
            None => {
                skipped += 1;
                warn!("Skipping edge with unknown type: {}", etype_str);
            }
        }
    }

    if skipped > 0 {
        warn!("Skipped {} edges with unrecognized types", skipped);
    }
    info!("Exported {} edges from SQLite", edges.len());
    Ok(edges)
}

/// Export a full snapshot from SQLite.
pub async fn export_sqlite(
    pool: &SqlitePool,
    tenant_id: Option<&str>,
) -> GraphDbResult<GraphSnapshot> {
    let nodes = export_nodes_sqlite(pool, tenant_id).await?;
    let edges = export_edges_sqlite(pool, tenant_id).await?;
    Ok(GraphSnapshot { nodes, edges })
}

// ─── Export from LadybugDB ──────────────────────────────────────────────

/// Export all nodes from LadybugDB, optionally filtered by tenant.
#[cfg(feature = "ladybug")]
pub fn export_nodes_ladybug(
    store: &crate::graph::LadybugGraphStore,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphNode>> {
    let filter = match tenant_id {
        Some(tid) => format!(" WHERE n.tenant_id = '{}'", tid.replace('\'', "\\'")),
        None => String::new(),
    };

    let cypher = format!(
        "MATCH (n:GraphNode){} \
         RETURN n.node_id, n.tenant_id, n.symbol_name, n.symbol_type, \
                n.file_path, n.start_line, n.end_line, n.signature, n.language, \
                n.qdrant_point_id, n.point_id_state",
        filter
    );

    let rows = store.execute_cypher(&cypher)?;
    let mut nodes = Vec::with_capacity(rows.len());

    for row in &rows {
        if row.len() < 5 {
            continue;
        }
        let symbol_type = match NodeType::from_str(&row[3]) {
            Some(t) => t,
            None => continue,
        };
        // Kuzu stores empty string as the "no link" sentinel; restore None on export.
        let qdrant_point_id = row.get(9).cloned().filter(|s| !s.is_empty());
        let point_id_state = row
            .get(10)
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "none".to_string());
        nodes.push(GraphNode {
            node_id: row[0].clone(),
            tenant_id: row[1].clone(),
            symbol_name: row[2].clone(),
            symbol_type,
            file_path: row[4].clone(),
            // Kuzu INT64 cannot be NULL, so the upsert stores 0 for an absent
            // line (`unwrap_or(0)`). Line numbers are 1-based (chunking.rs adds
            // +1), so 0 unambiguously means "unknown" — restore None for a
            // lossless round-trip against the SQLite backend (DATA-05).
            start_line: row
                .get(5)
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&v| v != 0),
            end_line: row
                .get(6)
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&v| v != 0),
            signature: row.get(7).cloned().filter(|s| !s.is_empty()),
            language: row.get(8).cloned().filter(|s| !s.is_empty()),
            branches: r#"["main"]"#.to_string(),
            qdrant_point_id,
            point_id_state,
        });
    }

    info!("Exported {} nodes from LadybugDB", nodes.len());
    Ok(nodes)
}

/// Export all edges from LadybugDB, optionally filtered by tenant.
#[cfg(feature = "ladybug")]
pub fn export_edges_ladybug(
    store: &crate::graph::LadybugGraphStore,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphEdge>> {
    let mut edges = Vec::new();

    for edge_type_str in &[
        // Structural
        "CALLS",
        "CONTAINS",
        "IMPORTS",
        "USES_TYPE",
        "EXTENDS",
        "IMPLEMENTS",
        // Narrative
        "EXPLAINS",
        "DESCRIBES",
        "REFERENCES_DOC",
        "ELABORATES",
        // Concept
        "COVERS_TOPIC",
        "IMPLEMENTS_CONCEPT",
    ] {
        let filter = match tenant_id {
            Some(tid) => format!(" WHERE r.tenant_id = '{}'", tid.replace('\'', "\\'")),
            None => String::new(),
        };

        let cypher = format!(
            "MATCH (a:GraphNode)-[r:{}]->(b:GraphNode){} \
             RETURN r.edge_id, r.tenant_id, a.node_id, b.node_id, \
                    r.source_file, r.weight, r.metadata_json",
            edge_type_str, filter
        );

        let edge_type = match EdgeType::from_str(edge_type_str) {
            Some(t) => t,
            None => continue,
        };

        let rows = store.execute_cypher(&cypher)?;
        for row in &rows {
            if row.len() < 6 {
                continue;
            }
            // Kuzu stores the empty string as the "no metadata" sentinel (rel
            // tables cannot hold NULL STRING); restore None on export so the
            // round-trip is lossless against the SQLite backend (DATA-05).
            let metadata_json = row.get(6).cloned().filter(|s| !s.is_empty());
            edges.push(GraphEdge {
                edge_id: row[0].clone(),
                tenant_id: row[1].clone(),
                source_node_id: row[2].clone(),
                target_node_id: row[3].clone(),
                edge_type,
                source_file: row[4].clone(),
                weight: row[5].parse().unwrap_or(1.0),
                metadata_json,
                // Rel tables carry no `branch` column; branch is not round-tripped
                // on this backend and is reported as a known gap by the conformance
                // suite if a fixture ever sets it.
                branch: None,
            });
        }
    }

    info!("Exported {} edges from LadybugDB", edges.len());
    Ok(edges)
}

/// Export a full snapshot from LadybugDB.
#[cfg(feature = "ladybug")]
pub fn export_ladybug(
    store: &crate::graph::LadybugGraphStore,
    tenant_id: Option<&str>,
) -> GraphDbResult<GraphSnapshot> {
    let nodes = export_nodes_ladybug(store, tenant_id)?;
    let edges = export_edges_ladybug(store, tenant_id)?;
    Ok(GraphSnapshot { nodes, edges })
}
