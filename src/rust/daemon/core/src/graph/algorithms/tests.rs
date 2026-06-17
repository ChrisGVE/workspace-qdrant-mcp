/// Integration-level tests for the algorithms module.
///
/// All tests operate over [`crate::graph::AdjacencyExport`] values built
/// in-memory — no SQLite pool required.  This verifies that every algorithm
/// produces correct output on the same graph topologies that were previously
/// tested via the pool-based helpers, and additionally asserts
/// backend-independence: identical input → identical output regardless of
/// which `GraphStore` implementation produced the export.
use crate::graph::AdjacencyExport;

use super::{
    compute_betweenness_centrality, compute_pagerank, detect_communities, CommunityConfig,
    PageRankConfig,
};

// ─── Shared helpers ──────────────────────────────────────────────────────────

/// Build an [`AdjacencyExport`] from a node count and a directed edge list.
/// All edges carry weight 1.0.
fn make_export(n: usize, edges: Vec<(usize, usize)>) -> AdjacencyExport {
    let node_ids: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
    let edges: Vec<(usize, usize, f64)> = edges.into_iter().map(|(s, t)| (s, t, 1.0)).collect();
    AdjacencyExport { node_ids, edges }
}

/// Diamond topology: n0→n1, n0→n2, n1→n3, n2→n3.
fn diamond() -> AdjacencyExport {
    make_export(4, vec![(0, 1), (0, 2), (1, 3), (2, 3)])
}

/// Chain topology: n0→n1→n2→n3→n4.
fn chain() -> AdjacencyExport {
    make_export(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)])
}

/// Two densely connected clusters with a single bridge n1→n3.
///
/// Cluster A: {n0,n1,n2} fully bidirectional.
/// Cluster B: {n3,n4,n5} fully bidirectional.
/// Bridge:    n1 → n3.
fn two_clusters_with_bridge() -> AdjacencyExport {
    make_export(
        6,
        vec![
            // Cluster A
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (1, 2),
            (2, 1),
            // Cluster B
            (3, 4),
            (4, 3),
            (3, 5),
            (5, 3),
            (4, 5),
            (5, 4),
            // Bridge
            (1, 3),
        ],
    )
}

// ─── PageRank tests ──────────────────────────────────────────────────────────

#[test]
fn test_pagerank_empty_graph() {
    let adj = make_export(0, vec![]);
    let results = compute_pagerank(&adj, &PageRankConfig::default());
    assert!(results.is_empty());
}

#[test]
fn test_pagerank_single_node() {
    let adj = make_export(1, vec![]);
    let results = compute_pagerank(&adj, &PageRankConfig::default());
    assert_eq!(results.len(), 1);
    assert!(
        (results[0].score - 1.0).abs() < 0.01,
        "single node should get all rank, got {}",
        results[0].score
    );
}

#[test]
fn test_pagerank_diamond() {
    let results = compute_pagerank(&diamond(), &PageRankConfig::default());
    assert_eq!(results.len(), 4);

    let d_score = results.iter().find(|r| r.node_id == "n3").unwrap().score;
    let a_score = results.iter().find(|r| r.node_id == "n0").unwrap().score;
    assert!(
        d_score > a_score,
        "sink n3 (2 inputs) should rank above source n0: d={d_score}, a={a_score}"
    );
}

#[test]
fn test_pagerank_chain_scores_sum_to_one() {
    let results = compute_pagerank(&chain(), &PageRankConfig::default());
    assert_eq!(results.len(), 5);
    let total: f64 = results.iter().map(|r| r.score).sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "PageRank scores should sum to ~1.0, got {total}"
    );
}

#[test]
fn test_pagerank_convergence() {
    let config = PageRankConfig {
        damping: 0.85,
        max_iterations: 1000,
        tolerance: 1e-10,
        ..Default::default()
    };
    let results = compute_pagerank(&diamond(), &config);
    let total: f64 = results.iter().map(|r| r.score).sum();
    assert!((total - 1.0).abs() < 1e-6, "converged total={total}");
}

#[test]
fn test_pagerank_identical_output_on_identical_input() {
    let adj = diamond();
    let config = PageRankConfig::default();
    let r1 = compute_pagerank(&adj, &config);
    let r2 = compute_pagerank(&adj, &config);
    assert_eq!(r1.len(), r2.len());
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.node_id, b.node_id);
        assert_eq!(
            a.score.to_bits(),
            b.score.to_bits(),
            "scores must be bit-identical for node {}",
            a.node_id
        );
    }
}

// ─── Community detection tests ───────────────────────────────────────────────

#[test]
fn test_communities_empty() {
    let adj = make_export(0, vec![]);
    let communities = detect_communities(&adj, &CommunityConfig::default());
    assert!(communities.is_empty());
}

#[test]
fn test_communities_two_disconnected_clusters() {
    // Two disconnected triangles: {n0,n1,n2} and {n3,n4,n5}.
    let adj = make_export(
        6,
        vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 0),
            (0, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (5, 3),
            (3, 5),
        ],
    );
    let config = CommunityConfig {
        max_iterations: 100,
        min_community_size: 2,
    };
    let communities = detect_communities(&adj, &config);
    assert_eq!(
        communities.len(),
        2,
        "Expected 2 disconnected communities, got {}",
        communities.len()
    );
    assert_eq!(communities[0].members.len(), 3);
    assert_eq!(communities[1].members.len(), 3);
}

#[test]
fn test_communities_fully_connected() {
    let adj = make_export(3, vec![(0, 1), (1, 2), (2, 0)]);
    let communities = detect_communities(&adj, &CommunityConfig::default());
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].members.len(), 3);
}

#[test]
fn test_communities_min_size_filter() {
    // n0-n1 connected; n2 isolated.
    let adj = make_export(3, vec![(0, 1)]);
    let config = CommunityConfig {
        min_community_size: 2,
        ..Default::default()
    };
    let communities = detect_communities(&adj, &config);
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].members.len(), 2);
}

#[test]
fn test_communities_sorted_by_size() {
    // Cluster of 4 vs cluster of 2.
    let adj = make_export(
        6,
        vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 0),
            (0, 3),
            (4, 5),
            (5, 4),
        ],
    );
    let communities = detect_communities(&adj, &CommunityConfig::default());
    if communities.len() >= 2 {
        assert!(
            communities[0].members.len() >= communities[1].members.len(),
            "Communities should be sorted by size descending"
        );
    }
}

#[test]
fn test_communities_identical_output_on_identical_input() {
    let adj = make_export(
        6,
        vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 0),
            (0, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
        ],
    );
    let config = CommunityConfig::default();
    let r1 = detect_communities(&adj, &config);
    let r2 = detect_communities(&adj, &config);
    assert_eq!(r1.len(), r2.len(), "community count must be deterministic");
    for (c1, c2) in r1.iter().zip(r2.iter()) {
        assert_eq!(c1.community_id, c2.community_id);
        assert_eq!(c1.members.len(), c2.members.len());
    }
}

// ─── Betweenness centrality tests ────────────────────────────────────────────

#[test]
fn test_betweenness_empty() {
    let adj = make_export(0, vec![]);
    assert!(compute_betweenness_centrality(&adj, None).is_empty());
}

#[test]
fn test_betweenness_chain() {
    let results = compute_betweenness_centrality(&chain(), None);
    assert_eq!(results.len(), 5);

    let c_score = results.iter().find(|r| r.node_id == "n2").unwrap().score;
    let a_score = results.iter().find(|r| r.node_id == "n0").unwrap().score;
    let e_score = results.iter().find(|r| r.node_id == "n4").unwrap().score;
    assert!(
        c_score >= a_score,
        "center n2 should have >= betweenness than endpoint n0: c={c_score}, a={a_score}"
    );
    assert!(
        c_score >= e_score,
        "center n2 should have >= betweenness than endpoint n4: c={c_score}, e={e_score}"
    );
}

#[test]
fn test_betweenness_bridge_node() {
    let results = compute_betweenness_centrality(&two_clusters_with_bridge(), None);
    let b_score = results.iter().find(|r| r.node_id == "n1").unwrap().score;
    let d_score = results.iter().find(|r| r.node_id == "n3").unwrap().score;
    let a_score = results.iter().find(|r| r.node_id == "n0").unwrap().score;
    assert!(
        b_score > a_score || d_score > a_score,
        "bridge nodes should have higher betweenness: b={b_score}, d={d_score}, a={a_score}"
    );
}

#[test]
fn test_betweenness_small_graph_score_zero() {
    let adj = make_export(2, vec![(0, 1)]);
    let results = compute_betweenness_centrality(&adj, None);
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.score == 0.0));
}

#[test]
fn test_betweenness_with_sampling() {
    let results = compute_betweenness_centrality(&chain(), Some(2));
    assert_eq!(results.len(), 5);
}

#[test]
fn test_betweenness_identical_output_on_identical_input() {
    let adj = two_clusters_with_bridge();
    let r1 = compute_betweenness_centrality(&adj, None);
    let r2 = compute_betweenness_centrality(&adj, None);
    assert_eq!(r1.len(), r2.len());
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.node_id, b.node_id);
        assert_eq!(
            a.score.to_bits(),
            b.score.to_bits(),
            "scores must be bit-identical for node {}",
            a.node_id
        );
    }
}
