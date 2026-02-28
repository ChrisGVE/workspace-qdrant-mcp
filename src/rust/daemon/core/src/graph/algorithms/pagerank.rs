/// PageRank algorithm for code relationship graphs.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::load_adjacency_graph;

/// PageRank score for a graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankEntry {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    pub score: f64,
}

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
