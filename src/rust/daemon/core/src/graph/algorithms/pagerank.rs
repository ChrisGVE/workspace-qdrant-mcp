//! PageRank algorithm for code relationship graphs.
//!
//! Location: `src/rust/daemon/core/src/graph/algorithms/`.
//!
//! Role: pure-function analytics that consumes an [`AdjacencyExport`] produced
//! by `GraphStore::export_adjacency` and returns ranked nodes. Performs no
//! database I/O. Invoked by the gRPC graph service
//! (`grpc/.../graph_service/analytics_handlers.rs`).
//!
//! # References
//! - Page, Brin, Motwani & Winograd, "The PageRank Citation Ranking: Bringing
//!   Order to the Web", Stanford InfoLab Technical Report, 1999.
//!
//! Dangling nodes (no outgoing edges) have their rank mass redistributed
//! uniformly across all nodes each iteration, preserving the total rank sum.
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::graph::AdjacencyExport;

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

/// Compute PageRank scores for all nodes represented in an [`AdjacencyExport`].
///
/// Operates entirely on the in-memory export — no database I/O.  The caller is
/// responsible for acquiring the export (via `GraphStore::export_adjacency`)
/// and releasing any read lock before invoking this function (LOCK-SCOPE contract).
///
/// `symbol_name`, `symbol_type`, and `file_path` fields in the returned entries
/// are left empty; callers that need display metadata should enrich the results
/// separately after this function returns.
pub fn compute_pagerank(adj: &AdjacencyExport, config: &PageRankConfig) -> Vec<PageRankEntry> {
    let n = adj.node_ids.len();
    if n == 0 {
        return Vec::new();
    }

    // Build outgoing and incoming adjacency lists indexed by node position.
    let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(src, tgt, _w) in &adj.edges {
        outgoing[src].push(tgt);
        incoming[tgt].push(src);
    }

    let scores = run_pagerank_iterations(n, &outgoing, &incoming, config);

    let mut results: Vec<PageRankEntry> = scores
        .into_iter()
        .enumerate()
        .map(|(i, score)| PageRankEntry {
            node_id: adj.node_ids[i].clone(),
            symbol_name: String::new(),
            symbol_type: String::new(),
            file_path: String::new(),
            score,
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    info!(nodes = results.len(), "PageRank computation complete");
    results
}

fn run_pagerank_iterations(
    n: usize,
    outgoing: &[Vec<usize>],
    incoming: &[Vec<usize>],
    config: &PageRankConfig,
) -> Vec<f64> {
    let initial = 1.0 / n as f64;
    let teleport = (1.0 - config.damping) / n as f64;

    let mut scores: Vec<f64> = vec![initial; n];

    for iteration in 0..config.max_iterations {
        // Dangling contribution: nodes with no outgoing edges leak rank to all.
        let dangling_sum: f64 = (0..n)
            .filter(|&i| outgoing[i].is_empty())
            .map(|i| scores[i])
            .sum();
        let dangling_contrib = config.damping * dangling_sum / n as f64;

        let mut new_scores: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            let incoming_sum: f64 = incoming[i]
                .iter()
                .map(|&pred| {
                    let out_degree = outgoing[pred].len().max(1);
                    scores[pred] / out_degree as f64
                })
                .sum();
            new_scores[i] = teleport + config.damping * incoming_sum + dangling_contrib;
        }

        let max_diff = (0..n)
            .map(|i| (new_scores[i] - scores[i]).abs())
            .fold(0.0f64, f64::max);

        scores = new_scores;
        if max_diff < config.tolerance {
            debug!(iterations = iteration + 1, "PageRank converged");
            break;
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AdjacencyExport;

    fn make_export(n: usize, edges: Vec<(usize, usize)>) -> AdjacencyExport {
        let node_ids: Vec<String> = (0..n).map(|i| format!("n{}", i)).collect();
        let edges: Vec<(usize, usize, f64)> = edges.into_iter().map(|(s, t)| (s, t, 1.0)).collect();
        AdjacencyExport { node_ids, edges }
    }

    #[test]
    fn test_pagerank_empty() {
        let adj = make_export(0, vec![]);
        let results = compute_pagerank(&adj, &PageRankConfig::default());
        assert!(results.is_empty());
    }

    #[test]
    fn test_pagerank_single_node() {
        let adj = make_export(1, vec![]);
        let results = compute_pagerank(&adj, &PageRankConfig::default());
        assert_eq!(results.len(), 1);
        // Single node with no edges: teleport only, score sums to 1.
        assert!(
            (results[0].score - 1.0).abs() < 0.01,
            "got {}",
            results[0].score
        );
    }

    #[test]
    fn test_pagerank_diamond() {
        // Diamond: 0->1, 0->2, 1->3, 2->3  (a->b, a->c, b->d, c->d)
        let adj = make_export(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
        let results = compute_pagerank(&adj, &PageRankConfig::default());
        assert_eq!(results.len(), 4);

        let d = results.iter().find(|r| r.node_id == "n3").unwrap().score;
        let a = results.iter().find(|r| r.node_id == "n0").unwrap().score;
        assert!(
            d > a,
            "sink n3 should rank higher than source n0: d={d}, a={a}"
        );
    }

    #[test]
    fn test_pagerank_scores_sum_to_one() {
        // Chain: 0->1->2->3->4
        let adj = make_export(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let results = compute_pagerank(&adj, &PageRankConfig::default());
        let total: f64 = results.iter().map(|r| r.score).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "scores should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn test_pagerank_identical_output_on_identical_input() {
        // Assert determinism: two calls on identical export yield byte-identical scores.
        let adj = make_export(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
        let config = PageRankConfig::default();
        let r1 = compute_pagerank(&adj, &config);
        let r2 = compute_pagerank(&adj, &config);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.node_id, b.node_id);
            assert_eq!(
                a.score.to_bits(),
                b.score.to_bits(),
                "scores must be bit-identical"
            );
        }
    }
}
