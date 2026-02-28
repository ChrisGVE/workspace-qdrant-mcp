//! Export functions for extracting graph data from SQLite and LadybugDB backends.

use sqlx::{Row, SqlitePool};
use tracing::{info, warn};

use crate::graph::schema::GraphDbResult;
use crate::graph::{EdgeType, GraphEdge, GraphNode, NodeType};
use super::GraphSnapshot;

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
                        file_path, start_line, end_line, signature, language
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
                        file_path, start_line, end_line, signature, language
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
                start_line: row
                    .get::<Option<i64>, _>("start_line")
                    .map(|v| v as u32),
                end_line: row.get::<Option<i64>, _>("end_line").map(|v| v as u32),
                signature: row.get("signature"),
                language: row.get("language"),
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
                        edge_type, source_file, weight, metadata_json
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
                        edge_type, source_file, weight, metadata_json
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
        Some(tid) => format!(
            " WHERE n.tenant_id = '{}'",
            tid.replace('\'', "\\'")
        ),
        None => String::new(),
    };

    let cypher = format!(
        "MATCH (n:GraphNode){} \
         RETURN n.node_id, n.tenant_id, n.symbol_name, n.symbol_type, \
                n.file_path, n.start_line, n.end_line, n.signature, n.language",
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
        nodes.push(GraphNode {
            node_id: row[0].clone(),
            tenant_id: row[1].clone(),
            symbol_name: row[2].clone(),
            symbol_type,
            file_path: row[4].clone(),
            start_line: row.get(5).and_then(|s| s.parse().ok()),
            end_line: row.get(6).and_then(|s| s.parse().ok()),
            signature: row.get(7).map(|s| s.clone()).filter(|s| !s.is_empty()),
            language: row.get(8).map(|s| s.clone()).filter(|s| !s.is_empty()),
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
        "CALLS",
        "CONTAINS",
        "IMPORTS",
        "USES_TYPE",
        "EXTENDS",
        "IMPLEMENTS",
    ] {
        let filter = match tenant_id {
            Some(tid) => format!(
                " WHERE r.tenant_id = '{}'",
                tid.replace('\'', "\\'")
            ),
            None => String::new(),
        };

        let cypher = format!(
            "MATCH (a:GraphNode)-[r:{}]->(b:GraphNode){} \
             RETURN r.edge_id, r.tenant_id, a.node_id, b.node_id, \
                    r.source_file, r.weight",
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
            edges.push(GraphEdge {
                edge_id: row[0].clone(),
                tenant_id: row[1].clone(),
                source_node_id: row[2].clone(),
                target_node_id: row[3].clone(),
                edge_type,
                source_file: row[4].clone(),
                weight: row[5].parse().unwrap_or(1.0),
                metadata_json: None,
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
