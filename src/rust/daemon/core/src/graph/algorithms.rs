/// Graph algorithms: PageRank, community detection, betweenness centrality.
///
/// Implemented as pure functions over adjacency data loaded from any
/// `GraphStore` backend (SQLite or LadybugDB). The algorithms operate on
/// in-memory adjacency lists, so they work identically regardless of backend.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

// ─── Result types ──────────────────────────────────────────────────────

/// PageRank score for a graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankEntry {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    pub score: f64,
}

/// A detected community (cluster) of nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub community_id: u32,
    pub members: Vec<CommunityMember>,
}

/// A node within a community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityMember {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
}

/// Betweenness centrality score for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetweennessEntry {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    pub score: f64,
}

// ─── Internal adjacency representation ─────────────────────────────────

/// Node metadata loaded from the graph.
#[derive(Debug, Clone)]
struct NodeInfo {
    symbol_name: String,
    symbol_type: String,
    file_path: String,
}

/// Adjacency list representation for algorithm execution.
#[derive(Debug)]
struct AdjacencyGraph {
    /// node_id → metadata
    nodes: HashMap<String, NodeInfo>,
    /// node_id → set of outgoing neighbor node_ids
    outgoing: HashMap<String, Vec<String>>,
    /// node_id → set of incoming neighbor node_ids (reverse edges)
    incoming: HashMap<String, Vec<String>>,
}

/// Load the full adjacency graph for a tenant from SQLite.
async fn load_adjacency_graph(
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
        sqlx::query(&query)
            .bind(tenant_id)
            .fetch_all(pool)
            .await?
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

// ─── PageRank ──────────────────────────────────────────────────────────

/// Configuration for PageRank computation.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor (probability of following a link vs random jump).
    /// Standard value: 0.85.
    pub damping: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold (stop when max score change < this).
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Compute PageRank scores for all nodes in a tenant's graph.
///
/// Uses the iterative power method on the adjacency graph.
pub async fn compute_pagerank(
    pool: &SqlitePool,
    tenant_id: &str,
    config: &PageRankConfig,
    edge_types: Option<&[&str]>,
) -> Result<Vec<PageRankEntry>, sqlx::Error> {
    let graph = load_adjacency_graph(pool, tenant_id, edge_types).await?;

    if graph.nodes.is_empty() {
        return Ok(Vec::new());
    }

    let n = graph.nodes.len();
    let node_ids: Vec<&String> = graph.nodes.keys().collect();

    // Initialize scores uniformly
    let initial = 1.0 / n as f64;
    let mut scores: HashMap<&str, f64> = node_ids
        .iter()
        .map(|id| (id.as_str(), initial))
        .collect();

    let teleport = (1.0 - config.damping) / n as f64;

    for iteration in 0..config.max_iterations {
        let mut new_scores: HashMap<&str, f64> = HashMap::with_capacity(n);

        // Compute dangling node contribution (nodes with no outgoing edges)
        let mut dangling_sum = 0.0;
        for id in &node_ids {
            if !graph.outgoing.contains_key(id.as_str())
                || graph.outgoing[id.as_str()].is_empty()
            {
                dangling_sum += scores[id.as_str()];
            }
        }
        let dangling_contrib = config.damping * dangling_sum / n as f64;

        for id in &node_ids {
            let mut incoming_sum = 0.0;

            if let Some(predecessors) = graph.incoming.get(id.as_str()) {
                for pred in predecessors {
                    let pred_out_degree = graph
                        .outgoing
                        .get(pred.as_str())
                        .map(|v| v.len())
                        .unwrap_or(1);
                    incoming_sum += scores.get(pred.as_str()).unwrap_or(&0.0) / pred_out_degree as f64;
                }
            }

            new_scores.insert(
                id.as_str(),
                teleport + config.damping * incoming_sum + dangling_contrib,
            );
        }

        // Check convergence
        let max_diff = node_ids
            .iter()
            .map(|id| (new_scores[id.as_str()] - scores[id.as_str()]).abs())
            .fold(0.0f64, f64::max);

        scores = new_scores;

        if max_diff < config.tolerance {
            debug!(
                tenant_id,
                iterations = iteration + 1,
                "PageRank converged"
            );
            break;
        }
    }

    // Build results sorted by score descending
    let mut results: Vec<PageRankEntry> = scores
        .into_iter()
        .filter_map(|(id, score)| {
            graph.nodes.get(id).map(|info| PageRankEntry {
                node_id: id.to_string(),
                symbol_name: info.symbol_name.clone(),
                symbol_type: info.symbol_type.clone(),
                file_path: info.file_path.clone(),
                score,
            })
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    info!(
        tenant_id,
        nodes = results.len(),
        "PageRank computation complete"
    );

    Ok(results)
}

// ─── Community detection (Label Propagation) ───────────────────────────

/// Configuration for community detection.
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Maximum iterations for label propagation.
    pub max_iterations: usize,
    /// Minimum community size to include in results.
    pub min_community_size: usize,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            min_community_size: 2,
        }
    }
}

/// Detect communities using label propagation algorithm.
///
/// Each node starts with a unique label. In each iteration, each node
/// adopts the most frequent label among its neighbors. Converges when
/// no labels change.
///
/// Treats edges as undirected for community detection.
pub async fn detect_communities(
    pool: &SqlitePool,
    tenant_id: &str,
    config: &CommunityConfig,
    edge_types: Option<&[&str]>,
) -> Result<Vec<Community>, sqlx::Error> {
    let graph = load_adjacency_graph(pool, tenant_id, edge_types).await?;

    if graph.nodes.is_empty() {
        return Ok(Vec::new());
    }

    let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

    // Build undirected adjacency (union of outgoing and incoming)
    let mut neighbors: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (src, targets) in &graph.outgoing {
        for tgt in targets {
            neighbors.entry(src.as_str()).or_default().insert(tgt.as_str());
            neighbors.entry(tgt.as_str()).or_default().insert(src.as_str());
        }
    }

    // Initialize: each node gets its index as label
    let id_to_idx: HashMap<&str, u32> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i as u32))
        .collect();
    let mut labels: HashMap<&str, u32> = id_to_idx.clone();

    // Iterate
    for iteration in 0..config.max_iterations {
        let mut changed = false;

        // Process nodes in order (deterministic for reproducibility)
        for id in &node_ids {
            let id_str = id.as_str();
            let nbrs = match neighbors.get(id_str) {
                Some(n) if !n.is_empty() => n,
                _ => continue, // isolated node keeps its label
            };

            // Count neighbor labels
            let mut label_counts: HashMap<u32, usize> = HashMap::new();
            for &nbr in nbrs {
                let label = labels[nbr];
                *label_counts.entry(label).or_default() += 1;
            }

            // Pick most frequent label (tie-break: smallest label for determinism)
            let best_label = label_counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                .map(|(label, _)| label)
                .unwrap();

            if labels[id_str] != best_label {
                labels.insert(id_str, best_label);
                changed = true;
            }
        }

        if !changed {
            debug!(
                tenant_id,
                iterations = iteration + 1,
                "Label propagation converged"
            );
            break;
        }
    }

    // Group nodes by label → communities
    let mut label_groups: HashMap<u32, Vec<CommunityMember>> = HashMap::new();
    for (id, &label) in &labels {
        if let Some(info) = graph.nodes.get(*id) {
            label_groups
                .entry(label)
                .or_default()
                .push(CommunityMember {
                    node_id: id.to_string(),
                    symbol_name: info.symbol_name.clone(),
                    symbol_type: info.symbol_type.clone(),
                    file_path: info.file_path.clone(),
                });
        }
    }

    // Filter by min size, assign sequential community IDs
    let mut communities: Vec<Community> = label_groups
        .into_values()
        .filter(|members| members.len() >= config.min_community_size)
        .enumerate()
        .map(|(i, mut members)| {
            members.sort_by(|a, b| a.symbol_name.cmp(&b.symbol_name));
            Community {
                community_id: i as u32,
                members,
            }
        })
        .collect();

    // Sort by size descending
    communities.sort_by(|a, b| b.members.len().cmp(&a.members.len()));

    // Re-number after sort
    for (i, community) in communities.iter_mut().enumerate() {
        community.community_id = i as u32;
    }

    info!(
        tenant_id,
        communities = communities.len(),
        "Community detection complete"
    );

    Ok(communities)
}

// ─── Betweenness centrality ────────────────────────────────────────────

/// Compute approximate betweenness centrality using Brandes' algorithm.
///
/// For each node s, runs BFS from s, then accumulates dependency values
/// along shortest paths. Normalized to [0, 1].
pub async fn compute_betweenness_centrality(
    pool: &SqlitePool,
    tenant_id: &str,
    edge_types: Option<&[&str]>,
    max_samples: Option<usize>,
) -> Result<Vec<BetweennessEntry>, sqlx::Error> {
    let graph = load_adjacency_graph(pool, tenant_id, edge_types).await?;

    if graph.nodes.len() < 3 {
        return Ok(graph
            .nodes
            .iter()
            .map(|(id, info)| BetweennessEntry {
                node_id: id.clone(),
                symbol_name: info.symbol_name.clone(),
                symbol_type: info.symbol_type.clone(),
                file_path: info.file_path.clone(),
                score: 0.0,
            })
            .collect());
    }

    let node_ids: Vec<&String> = graph.nodes.keys().collect();

    // Build undirected adjacency for betweenness
    let mut neighbors: HashMap<&str, Vec<&str>> = HashMap::new();
    for (src, targets) in &graph.outgoing {
        for tgt in targets {
            neighbors.entry(src.as_str()).or_default().push(tgt.as_str());
            neighbors.entry(tgt.as_str()).or_default().push(src.as_str());
        }
    }

    let mut betweenness: HashMap<&str, f64> = node_ids
        .iter()
        .map(|id| (id.as_str(), 0.0))
        .collect();

    // Select source nodes (all, or a sample for large graphs)
    let sources: Vec<&str> = match max_samples {
        Some(limit) if limit < node_ids.len() => {
            node_ids.iter().take(limit).map(|id| id.as_str()).collect()
        }
        _ => node_ids.iter().map(|id| id.as_str()).collect(),
    };

    // Brandes' algorithm: BFS from each source
    for &source in &sources {
        let mut stack: Vec<&str> = Vec::new();
        let mut predecessors: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut sigma: HashMap<&str, f64> = HashMap::new(); // num shortest paths
        let mut dist: HashMap<&str, i64> = HashMap::new();

        sigma.insert(source, 1.0);
        dist.insert(source, 0);

        // BFS
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let d_v = dist[v];

            for &w in neighbors.get(v).unwrap_or(&Vec::new()) {
                // First visit
                if !dist.contains_key(w) {
                    dist.insert(w, d_v + 1);
                    queue.push_back(w);
                }
                // Shortest path through v
                if dist[w] == d_v + 1 {
                    let sigma_v = *sigma.get(v).unwrap_or(&0.0);
                    *sigma.entry(w).or_default() += sigma_v;
                    predecessors.entry(w).or_default().push(v);
                }
            }
        }

        // Back-propagation of dependencies
        let mut delta: HashMap<&str, f64> = HashMap::new();

        while let Some(w) = stack.pop() {
            if let Some(preds) = predecessors.get(w) {
                let sigma_w = *sigma.get(w).unwrap_or(&1.0);
                let delta_w = *delta.get(w).unwrap_or(&0.0);

                for &v in preds {
                    let sigma_v = *sigma.get(v).unwrap_or(&1.0);
                    let contribution = (sigma_v / sigma_w) * (1.0 + delta_w);
                    *delta.entry(v).or_default() += contribution;
                }
            }

            if w != source {
                *betweenness.entry(w).or_default() += delta.get(w).unwrap_or(&0.0);
            }
        }
    }

    // Normalize: for undirected graph, divide by 2; then by (n-1)(n-2)
    let n = graph.nodes.len() as f64;
    let normalizer = if n > 2.0 {
        (n - 1.0) * (n - 2.0) / 2.0
    } else {
        1.0
    };

    // If using sampling, scale up
    let sample_scale = if sources.len() < node_ids.len() {
        node_ids.len() as f64 / sources.len() as f64
    } else {
        1.0
    };

    let mut results: Vec<BetweennessEntry> = betweenness
        .into_iter()
        .filter_map(|(id, raw_score)| {
            graph.nodes.get(id).map(|info| {
                let normalized = (raw_score * sample_scale) / normalizer;
                BetweennessEntry {
                    node_id: id.to_string(),
                    symbol_name: info.symbol_name.clone(),
                    symbol_type: info.symbol_type.clone(),
                    file_path: info.file_path.clone(),
                    score: normalized.min(1.0),
                }
            })
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    info!(
        tenant_id,
        nodes = results.len(),
        sources = sources.len(),
        "Betweenness centrality computation complete"
    );

    Ok(results)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    /// Create an in-memory SQLite pool with graph schema.
    async fn setup_graph_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        sqlx::query(
            "CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                signature TEXT,
                language TEXT,
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query("CREATE INDEX idx_nodes_tenant ON graph_nodes(tenant_id)")
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(
            "CREATE TABLE graph_edges (
                edge_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT ''
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query("CREATE INDEX idx_edges_tenant ON graph_edges(tenant_id)")
            .execute(&pool)
            .await
            .unwrap();

        pool
    }

    async fn insert_node(pool: &SqlitePool, tenant: &str, id: &str, name: &str, stype: &str) {
        sqlx::query(
            "INSERT OR IGNORE INTO graph_nodes (node_id, tenant_id, symbol_name, symbol_type, file_path)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(id)
        .bind(tenant)
        .bind(name)
        .bind(stype)
        .bind(format!("{}.rs", name))
        .execute(pool)
        .await
        .unwrap();
    }

    async fn insert_edge(pool: &SqlitePool, tenant: &str, src: &str, tgt: &str, etype: &str) {
        let edge_id = format!("{}_{}_{}_{}", tenant, src, tgt, etype);
        sqlx::query(
            "INSERT OR IGNORE INTO graph_edges (edge_id, tenant_id, source_node_id, target_node_id, edge_type, source_file)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&edge_id)
        .bind(tenant)
        .bind(src)
        .bind(tgt)
        .bind(etype)
        .bind("src.rs")
        .execute(pool)
        .await
        .unwrap();
    }

    /// Build a diamond graph: A -> B, A -> C, B -> D, C -> D
    async fn build_diamond(pool: &SqlitePool) {
        for (id, name) in &[("a", "alpha"), ("b", "beta"), ("c", "gamma"), ("d", "delta")] {
            insert_node(pool, "t1", id, name, "function").await;
        }
        insert_edge(pool, "t1", "a", "b", "CALLS").await;
        insert_edge(pool, "t1", "a", "c", "CALLS").await;
        insert_edge(pool, "t1", "b", "d", "CALLS").await;
        insert_edge(pool, "t1", "c", "d", "CALLS").await;
    }

    /// Build a chain: A -> B -> C -> D -> E
    async fn build_chain(pool: &SqlitePool) {
        for (id, name) in &[("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("e", "e")] {
            insert_node(pool, "t1", id, name, "function").await;
        }
        insert_edge(pool, "t1", "a", "b", "CALLS").await;
        insert_edge(pool, "t1", "b", "c", "CALLS").await;
        insert_edge(pool, "t1", "c", "d", "CALLS").await;
        insert_edge(pool, "t1", "d", "e", "CALLS").await;
    }

    /// Build two clusters: {A,B,C} densely connected, {D,E,F} densely connected,
    /// with one bridge B->D.
    async fn build_two_clusters(pool: &SqlitePool) {
        for (id, name) in &[("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("e", "e"), ("f", "f")] {
            insert_node(pool, "t1", id, name, "function").await;
        }
        // Cluster 1: a-b-c fully connected
        insert_edge(pool, "t1", "a", "b", "CALLS").await;
        insert_edge(pool, "t1", "b", "a", "CALLS").await;
        insert_edge(pool, "t1", "a", "c", "CALLS").await;
        insert_edge(pool, "t1", "c", "a", "CALLS").await;
        insert_edge(pool, "t1", "b", "c", "CALLS").await;
        insert_edge(pool, "t1", "c", "b", "CALLS").await;
        // Cluster 2: d-e-f fully connected
        insert_edge(pool, "t1", "d", "e", "CALLS").await;
        insert_edge(pool, "t1", "e", "d", "CALLS").await;
        insert_edge(pool, "t1", "d", "f", "CALLS").await;
        insert_edge(pool, "t1", "f", "d", "CALLS").await;
        insert_edge(pool, "t1", "e", "f", "CALLS").await;
        insert_edge(pool, "t1", "f", "e", "CALLS").await;
        // Bridge: b -> d
        insert_edge(pool, "t1", "b", "d", "CALLS").await;
    }

    // ─── PageRank tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_pagerank_empty_graph() {
        let pool = setup_graph_pool().await;
        let config = PageRankConfig::default();
        let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_pagerank_single_node() {
        let pool = setup_graph_pool().await;
        insert_node(&pool, "t1", "a", "alpha", "function").await;

        let config = PageRankConfig::default();
        let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].score - 1.0).abs() < 0.01); // single node gets all rank
    }

    #[tokio::test]
    async fn test_pagerank_diamond() {
        let pool = setup_graph_pool().await;
        build_diamond(&pool).await;

        let config = PageRankConfig::default();
        let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
        assert_eq!(results.len(), 4);

        // Node D should have highest PageRank (two incoming edges)
        let d_score = results.iter().find(|r| r.node_id == "d").unwrap().score;
        let a_score = results.iter().find(|r| r.node_id == "a").unwrap().score;
        assert!(
            d_score > a_score,
            "D (sink with 2 inputs) should rank higher than A (source): d={}, a={}",
            d_score,
            a_score
        );
    }

    #[tokio::test]
    async fn test_pagerank_chain() {
        let pool = setup_graph_pool().await;
        build_chain(&pool).await;

        let config = PageRankConfig::default();
        let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();
        assert_eq!(results.len(), 5);

        // All scores should sum to approximately 1.0
        let total: f64 = results.iter().map(|r| r.score).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "PageRank scores should sum to ~1.0, got {}",
            total
        );
    }

    #[tokio::test]
    async fn test_pagerank_convergence() {
        let pool = setup_graph_pool().await;
        build_diamond(&pool).await;

        let config = PageRankConfig {
            damping: 0.85,
            max_iterations: 1000,
            tolerance: 1e-10,
            ..Default::default()
        };
        let results = compute_pagerank(&pool, "t1", &config, None).await.unwrap();

        // Should converge to stable values
        let total: f64 = results.iter().map(|r| r.score).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_pagerank_edge_type_filter() {
        let pool = setup_graph_pool().await;
        // A -CALLS-> B, A -IMPORTS-> C
        insert_node(&pool, "t1", "a", "a", "function").await;
        insert_node(&pool, "t1", "b", "b", "function").await;
        insert_node(&pool, "t1", "c", "c", "function").await;
        insert_edge(&pool, "t1", "a", "b", "CALLS").await;
        insert_edge(&pool, "t1", "a", "c", "IMPORTS").await;

        let config = PageRankConfig::default();
        let results = compute_pagerank(&pool, "t1", &config, Some(&["CALLS"]))
            .await
            .unwrap();

        // C should have low PageRank since IMPORTS edges are excluded
        let b_score = results.iter().find(|r| r.node_id == "b").unwrap().score;
        let c_score = results.iter().find(|r| r.node_id == "c").unwrap().score;
        assert!(
            b_score > c_score,
            "B should rank higher when only CALLS are considered"
        );
    }

    // ─── Community detection tests ───────────────────────────────────

    #[tokio::test]
    async fn test_communities_empty() {
        let pool = setup_graph_pool().await;
        let config = CommunityConfig::default();
        let communities = detect_communities(&pool, "t1", &config, None).await.unwrap();
        assert!(communities.is_empty());
    }

    #[tokio::test]
    async fn test_communities_two_disconnected_clusters() {
        let pool = setup_graph_pool().await;

        // Two disconnected clusters: {a,b,c} and {d,e,f}
        for (id, name) in &[("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"), ("e", "e"), ("f", "f")] {
            insert_node(&pool, "t1", id, name, "function").await;
        }
        // Cluster 1
        insert_edge(&pool, "t1", "a", "b", "CALLS").await;
        insert_edge(&pool, "t1", "b", "c", "CALLS").await;
        insert_edge(&pool, "t1", "c", "a", "CALLS").await;
        // Cluster 2
        insert_edge(&pool, "t1", "d", "e", "CALLS").await;
        insert_edge(&pool, "t1", "e", "f", "CALLS").await;
        insert_edge(&pool, "t1", "f", "d", "CALLS").await;

        let config = CommunityConfig {
            max_iterations: 100,
            min_community_size: 2,
        };
        let communities = detect_communities(&pool, "t1", &config, None).await.unwrap();

        // Should detect exactly 2 communities
        assert_eq!(
            communities.len(), 2,
            "Expected 2 disconnected communities, got {}",
            communities.len()
        );

        // Each community should have 3 members
        assert_eq!(communities[0].members.len(), 3);
        assert_eq!(communities[1].members.len(), 3);
    }

    #[tokio::test]
    async fn test_communities_fully_connected() {
        let pool = setup_graph_pool().await;

        // All nodes connected → one community
        for (id, name) in &[("a", "a"), ("b", "b"), ("c", "c")] {
            insert_node(&pool, "t1", id, name, "function").await;
        }
        insert_edge(&pool, "t1", "a", "b", "CALLS").await;
        insert_edge(&pool, "t1", "b", "c", "CALLS").await;
        insert_edge(&pool, "t1", "c", "a", "CALLS").await;

        let config = CommunityConfig::default();
        let communities = detect_communities(&pool, "t1", &config, None).await.unwrap();

        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].members.len(), 3);
    }

    #[tokio::test]
    async fn test_communities_min_size_filter() {
        let pool = setup_graph_pool().await;

        // Two nodes connected, one isolated
        insert_node(&pool, "t1", "a", "a", "function").await;
        insert_node(&pool, "t1", "b", "b", "function").await;
        insert_node(&pool, "t1", "c", "c", "function").await;
        insert_edge(&pool, "t1", "a", "b", "CALLS").await;

        let config = CommunityConfig {
            min_community_size: 2,
            ..Default::default()
        };
        let communities = detect_communities(&pool, "t1", &config, None).await.unwrap();

        // Only the {a, b} community should pass the filter
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].members.len(), 2);
    }

    #[tokio::test]
    async fn test_communities_sorted_by_size() {
        let pool = setup_graph_pool().await;
        build_two_clusters(&pool).await;

        // Add extra node to cluster 1 to make it bigger
        insert_node(&pool, "t1", "g", "g", "function").await;
        insert_edge(&pool, "t1", "a", "g", "CALLS").await;
        insert_edge(&pool, "t1", "g", "a", "CALLS").await;

        let config = CommunityConfig::default();
        let communities = detect_communities(&pool, "t1", &config, None).await.unwrap();

        if communities.len() >= 2 {
            assert!(
                communities[0].members.len() >= communities[1].members.len(),
                "Communities should be sorted by size descending"
            );
        }
    }

    // ─── Betweenness centrality tests ────────────────────────────────

    #[tokio::test]
    async fn test_betweenness_empty() {
        let pool = setup_graph_pool().await;
        let results = compute_betweenness_centrality(&pool, "t1", None, None)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_betweenness_chain() {
        let pool = setup_graph_pool().await;
        build_chain(&pool).await;

        let results = compute_betweenness_centrality(&pool, "t1", None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 5);

        // Middle nodes (b, c, d) should have higher betweenness than endpoints
        let b_score = results.iter().find(|r| r.node_id == "b").unwrap().score;
        let c_score = results.iter().find(|r| r.node_id == "c").unwrap().score;
        let a_score = results.iter().find(|r| r.node_id == "a").unwrap().score;
        let e_score = results.iter().find(|r| r.node_id == "e").unwrap().score;

        assert!(
            c_score >= a_score,
            "Center node c should have >= betweenness than endpoint a: c={}, a={}",
            c_score,
            a_score
        );
        assert!(
            c_score >= e_score,
            "Center node c should have >= betweenness than endpoint e: c={}, e={}",
            c_score,
            e_score
        );
    }

    #[tokio::test]
    async fn test_betweenness_bridge_node() {
        let pool = setup_graph_pool().await;
        build_two_clusters(&pool).await;

        let results = compute_betweenness_centrality(&pool, "t1", None, None)
            .await
            .unwrap();

        // Bridge nodes (b and d) should have highest betweenness
        let b_score = results.iter().find(|r| r.node_id == "b").unwrap().score;
        let d_score = results.iter().find(|r| r.node_id == "d").unwrap().score;
        let a_score = results.iter().find(|r| r.node_id == "a").unwrap().score;

        // b connects the two clusters, so it should have high betweenness
        assert!(
            b_score > a_score || d_score > a_score,
            "Bridge nodes should have higher betweenness: b={}, d={}, a={}",
            b_score,
            d_score,
            a_score
        );
    }

    #[tokio::test]
    async fn test_betweenness_small_graph() {
        let pool = setup_graph_pool().await;

        // Two nodes, one edge
        insert_node(&pool, "t1", "a", "a", "function").await;
        insert_node(&pool, "t1", "b", "b", "function").await;
        insert_edge(&pool, "t1", "a", "b", "CALLS").await;

        let results = compute_betweenness_centrality(&pool, "t1", None, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        // With only 2 nodes, betweenness is 0 for both
        assert!(results.iter().all(|r| r.score == 0.0));
    }

    #[tokio::test]
    async fn test_betweenness_with_sampling() {
        let pool = setup_graph_pool().await;
        build_chain(&pool).await;

        // Sample only 2 source nodes
        let results = compute_betweenness_centrality(&pool, "t1", None, Some(2))
            .await
            .unwrap();
        assert_eq!(results.len(), 5);
    }

    // ─── Load adjacency ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_load_adjacency() {
        let pool = setup_graph_pool().await;
        build_diamond(&pool).await;

        let graph = load_adjacency_graph(&pool, "t1", None).await.unwrap();
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.outgoing.get("a").unwrap().len(), 2); // a -> b, a -> c
        assert_eq!(graph.incoming.get("d").unwrap().len(), 2); // b -> d, c -> d
    }

    #[tokio::test]
    async fn test_load_adjacency_filtered() {
        let pool = setup_graph_pool().await;

        insert_node(&pool, "t1", "a", "a", "function").await;
        insert_node(&pool, "t1", "b", "b", "function").await;
        insert_edge(&pool, "t1", "a", "b", "CALLS").await;
        insert_edge(&pool, "t1", "a", "b", "IMPORTS").await;

        // Filter to CALLS only
        let graph = load_adjacency_graph(&pool, "t1", Some(&["CALLS"])).await.unwrap();
        let out = graph.outgoing.get("a").unwrap();
        assert_eq!(out.len(), 1); // only the CALLS edge
    }
}
