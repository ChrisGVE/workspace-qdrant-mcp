/// Betweenness centrality using Brandes' algorithm.
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::graph::AdjacencyExport;

/// Betweenness centrality score for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetweennessEntry {
    pub node_id: String,
    pub symbol_name: String,
    pub symbol_type: String,
    pub file_path: String,
    pub score: f64,
}

/// Compute approximate betweenness centrality using Brandes' algorithm over an
/// [`AdjacencyExport`].
///
/// Operates entirely in memory — no database I/O.  The caller is responsible
/// for acquiring the export (via `GraphStore::export_adjacency`) and releasing
/// any read lock before invoking this function (LOCK-SCOPE contract).
///
/// For each source node, runs BFS then accumulates dependency values along
/// shortest paths.  Scores are normalised to \[0, 1\].
///
/// `max_samples`: when `Some(k)` with `k < n`, only the first `k` nodes (in
/// index order) are used as BFS sources, with a scale correction applied.
///
/// `symbol_name`, `symbol_type`, and `file_path` fields in the returned entries
/// are left empty; callers that need display metadata should enrich the results
/// separately after this function returns.
pub fn compute_betweenness_centrality(
    adj: &AdjacencyExport,
    max_samples: Option<usize>,
) -> Vec<BetweennessEntry> {
    let n = adj.node_ids.len();

    if n < 3 {
        // With fewer than 3 nodes no path can pass through an intermediate node.
        return adj
            .node_ids
            .iter()
            .map(|id| BetweennessEntry {
                node_id: id.clone(),
                symbol_name: String::new(),
                symbol_type: String::new(),
                file_path: String::new(),
                score: 0.0,
            })
            .collect();
    }

    // Build an undirected adjacency list (index-based).
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(src, tgt, _w) in &adj.edges {
        if src != tgt {
            neighbors[src].push(tgt);
            neighbors[tgt].push(src);
        }
    }

    let mut betweenness: Vec<f64> = vec![0.0; n];

    let num_sources = match max_samples {
        Some(k) if k < n => k,
        _ => n,
    };
    let sources: Vec<usize> = (0..num_sources).collect();

    for &source in &sources {
        brandes_bfs(source, n, &neighbors, &mut betweenness);
    }

    let results = normalize_betweenness(betweenness, &adj.node_ids, n, sources.len());

    info!(
        nodes = results.len(),
        sources = sources.len(),
        "Betweenness centrality computation complete"
    );

    results
}

/// Normalise raw betweenness scores and build a sorted `BetweennessEntry` list.
fn normalize_betweenness(
    betweenness: Vec<f64>,
    node_ids: &[String],
    n: usize,
    num_sources: usize,
) -> Vec<BetweennessEntry> {
    let n_f = n as f64;
    let normalizer = if n_f > 2.0 {
        (n_f - 1.0) * (n_f - 2.0) / 2.0
    } else {
        1.0
    };
    let sample_scale = if num_sources < n {
        n_f / num_sources as f64
    } else {
        1.0
    };

    let mut results: Vec<BetweennessEntry> = betweenness
        .into_iter()
        .enumerate()
        .map(|(i, raw)| BetweennessEntry {
            node_id: node_ids[i].clone(),
            symbol_name: String::new(),
            symbol_type: String::new(),
            file_path: String::new(),
            score: ((raw * sample_scale) / normalizer).min(1.0),
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Run BFS from `source` (Brandes) and accumulate betweenness contributions.
fn brandes_bfs(source: usize, n: usize, neighbors: &[Vec<usize>], betweenness: &mut Vec<f64>) {
    let mut stack: Vec<usize> = Vec::new();
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut sigma: Vec<f64> = vec![0.0; n]; // num shortest paths
    let mut dist: Vec<i64> = vec![-1; n];

    sigma[source] = 1.0;
    dist[source] = 0;

    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        let d_v = dist[v];

        for &w in &neighbors[v] {
            // First visit.
            if dist[w] < 0 {
                dist[w] = d_v + 1;
                queue.push_back(w);
            }
            // Shortest path through v.
            if dist[w] == d_v + 1 {
                sigma[w] += sigma[v];
                predecessors[w].push(v);
            }
        }
    }

    // Back-propagation of dependencies.
    let mut delta: Vec<f64> = vec![0.0; n];

    while let Some(w) = stack.pop() {
        for &v in &predecessors[w] {
            let contribution = (sigma[v] / sigma[w].max(1.0)) * (1.0 + delta[w]);
            delta[v] += contribution;
        }
        if w != source {
            betweenness[w] += delta[w];
        }
    }
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
    fn test_betweenness_empty() {
        let adj = make_export(0, vec![]);
        assert!(compute_betweenness_centrality(&adj, None).is_empty());
    }

    #[test]
    fn test_betweenness_two_nodes_score_zero() {
        let adj = make_export(2, vec![(0, 1)]);
        let results = compute_betweenness_centrality(&adj, None);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.score == 0.0));
    }

    #[test]
    fn test_betweenness_chain_middle_higher() {
        // Chain: 0-1-2-3-4  (undirected via directed edges)
        let adj = make_export(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let results = compute_betweenness_centrality(&adj, None);
        assert_eq!(results.len(), 5);

        let c = results.iter().find(|r| r.node_id == "n2").unwrap().score;
        let a = results.iter().find(|r| r.node_id == "n0").unwrap().score;
        let e = results.iter().find(|r| r.node_id == "n4").unwrap().score;
        assert!(
            c >= a,
            "center n2 should have >= betweenness than endpoint n0: c={c}, a={a}"
        );
        assert!(
            c >= e,
            "center n2 should have >= betweenness than endpoint n4: c={c}, e={e}"
        );
    }

    #[test]
    fn test_betweenness_bridge_node_higher() {
        // Two triangles joined by a bridge: {0,1,2} and {3,4,5}, bridge 1->3.
        let adj = make_export(
            6,
            vec![
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (3, 4),
                (4, 3),
                (3, 5),
                (5, 3),
                (4, 5),
                (5, 4),
                (1, 3), // bridge
            ],
        );
        let results = compute_betweenness_centrality(&adj, None);
        let b = results.iter().find(|r| r.node_id == "n1").unwrap().score;
        let d = results.iter().find(|r| r.node_id == "n3").unwrap().score;
        let a = results.iter().find(|r| r.node_id == "n0").unwrap().score;
        assert!(
            b > a || d > a,
            "bridge nodes should have higher betweenness: b={b}, d={d}, a={a}"
        );
    }

    #[test]
    fn test_betweenness_with_sampling() {
        let adj = make_export(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let results = compute_betweenness_centrality(&adj, Some(2));
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_betweenness_identical_output_on_identical_input() {
        let adj = make_export(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let r1 = compute_betweenness_centrality(&adj, None);
        let r2 = compute_betweenness_centrality(&adj, None);
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
