/// Graph algorithms: PageRank, community detection, betweenness centrality.
///
/// Implemented as pure functions over adjacency data loaded from any
/// `GraphStore` backend (SQLite or LadybugDB). The algorithms operate on
/// in-memory adjacency lists, so they work identically regardless of backend.
mod betweenness;
mod community;
mod pagerank;

pub use betweenness::{compute_betweenness_centrality, BetweennessEntry};
pub use community::{detect_communities, Community, CommunityConfig, CommunityMember};
pub use pagerank::{compute_pagerank, PageRankConfig, PageRankEntry};

use std::collections::HashMap;

use sqlx::{Row, SqlitePool};
use tracing::debug;

// ─── Internal adjacency representation ─────────────────────────────────

/// Node metadata loaded from the graph.
#[derive(Debug, Clone)]
pub(super) struct NodeInfo {
    pub(super) symbol_name: String,
    pub(super) symbol_type: String,
    pub(super) file_path: String,
}

/// Adjacency list representation for algorithm execution.
#[derive(Debug)]
pub(super) struct AdjacencyGraph {
    /// node_id → metadata
    pub(super) nodes: HashMap<String, NodeInfo>,
    /// node_id → set of outgoing neighbor node_ids
    pub(super) outgoing: HashMap<String, Vec<String>>,
    /// node_id → set of incoming neighbor node_ids (reverse edges)
    pub(super) incoming: HashMap<String, Vec<String>>,
}

/// Load the full adjacency graph for a tenant from SQLite.
pub(super) async fn load_adjacency_graph(
    pool: &SqlitePool,
    tenant_id: &str,
    edge_types: Option<&[&str]>,
) -> Result<AdjacencyGraph, sqlx::Error> {
    // Load nodes
    let node_rows = sqlx::query(
        "SELECT node_id, symbol_name, symbol_type, file_path
         FROM graph_nodes WHERE tenant_id = ?1",
    )
    .bind(tenant_id)
    .fetch_all(pool)
    .await?;

    let mut nodes = HashMap::with_capacity(node_rows.len());
    for row in &node_rows {
        let node_id: String = row.get("node_id");
        nodes.insert(
            node_id,
            NodeInfo {
                symbol_name: row.get("symbol_name"),
                symbol_type: row.get("symbol_type"),
                file_path: row.get("file_path"),
            },
        );
    }

    // Load edges with optional type filter
    let edge_rows = if let Some(types) = edge_types {
        let placeholders: Vec<String> = types.iter().map(|t| format!("'{}'", t)).collect();
        let query = format!(
            "SELECT source_node_id, target_node_id FROM graph_edges
             WHERE tenant_id = ?1 AND edge_type IN ({})",
            placeholders.join(", ")
        );
        sqlx::query(&query).bind(tenant_id).fetch_all(pool).await?
    } else {
        sqlx::query(
            "SELECT source_node_id, target_node_id FROM graph_edges
             WHERE tenant_id = ?1",
        )
        .bind(tenant_id)
        .fetch_all(pool)
        .await?
    };

    let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
    let mut incoming: HashMap<String, Vec<String>> = HashMap::new();

    for row in &edge_rows {
        let src: String = row.get("source_node_id");
        let tgt: String = row.get("target_node_id");
        outgoing.entry(src.clone()).or_default().push(tgt.clone());
        incoming.entry(tgt).or_default().push(src);
    }

    debug!(
        tenant_id,
        nodes = nodes.len(),
        edges = edge_rows.len(),
        "Loaded adjacency graph"
    );

    Ok(AdjacencyGraph {
        nodes,
        outgoing,
        incoming,
    })
}

#[cfg(test)]
mod tests;
