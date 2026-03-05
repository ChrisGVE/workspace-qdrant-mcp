/// Betweenness centrality using Brandes' algorithm.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tracing::info;

use super::load_adjacency_graph;

/// Betweenness centrality score for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetweennessEntry {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    pub score: f64,
}

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

    let mut neighbors: HashMap<&str, Vec<&str>> = HashMap::new();
    for (src, targets) in &graph.outgoing {
        for tgt in targets {
            neighbors
                .entry(src.as_str())
                .or_default()
                .push(tgt.as_str());
            neighbors
                .entry(tgt.as_str())
                .or_default()
                .push(src.as_str());
        }
    }

    let mut betweenness: HashMap<&str, f64> =
        node_ids.iter().map(|id| (id.as_str(), 0.0)).collect();

    let sources: Vec<&str> = match max_samples {
        Some(limit) if limit < node_ids.len() => {
            node_ids.iter().take(limit).map(|id| id.as_str()).collect()
        }
        _ => node_ids.iter().map(|id| id.as_str()).collect(),
    };

    for &source in &sources {
        brandes_bfs(source, &neighbors, &mut betweenness);
    }

    let results = normalize_betweenness(betweenness, &graph.nodes, &node_ids, &sources);

    info!(
        tenant_id,
        nodes = results.len(),
        sources = sources.len(),
        "Betweenness centrality computation complete"
    );

    Ok(results)
}

/// Normalize raw betweenness scores and convert to sorted `BetweennessEntry` list.
fn normalize_betweenness<'a>(
    betweenness: HashMap<&'a str, f64>,
    nodes: &'a HashMap<String, super::NodeInfo>,
    node_ids: &[&String],
    sources: &[&str],
) -> Vec<BetweennessEntry> {
    let n = node_ids.len() as f64;
    let normalizer = if n > 2.0 {
        (n - 1.0) * (n - 2.0) / 2.0
    } else {
        1.0
    };
    let sample_scale = if sources.len() < node_ids.len() {
        n / sources.len() as f64
    } else {
        1.0
    };

    let mut results: Vec<BetweennessEntry> = betweenness
        .into_iter()
        .filter_map(|(id, raw_score)| {
            nodes.get(id).map(|info| BetweennessEntry {
                node_id: id.to_string(),
                symbol_name: info.symbol_name.clone(),
                symbol_type: info.symbol_type.clone(),
                file_path: info.file_path.clone(),
                score: ((raw_score * sample_scale) / normalizer).min(1.0),
            })
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Run BFS from a single source and accumulate betweenness contributions.
fn brandes_bfs<'a>(
    source: &'a str,
    neighbors: &HashMap<&'a str, Vec<&'a str>>,
    betweenness: &mut HashMap<&'a str, f64>,
) {
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
