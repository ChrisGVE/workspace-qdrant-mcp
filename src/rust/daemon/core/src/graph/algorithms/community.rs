/// Community detection using label propagation algorithm.
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::graph::AdjacencyExport;

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

/// Detect communities using label propagation algorithm over an [`AdjacencyExport`].
///
/// Each node starts with a unique label.  In each iteration, each node adopts
/// the most frequent label among its undirected neighbours.  Converges when no
/// labels change.
///
/// Operates entirely in memory — no database I/O.  The caller is responsible
/// for acquiring the export (via `GraphStore::export_adjacency`) and releasing
/// any read lock before invoking this function (LOCK-SCOPE contract).
///
/// `symbol_name`, `symbol_type`, and `file_path` fields in the returned members
/// are left empty; callers that need display metadata should enrich the results
/// separately after this function returns.
///
/// # References
/// - Raghavan, Albert & Kumara, "Near linear time algorithm to detect community
///   structures in large-scale networks", Physical Review E 76, 036106, 2007.
///   <https://doi.org/10.1103/PhysRevE.76.036106>
///
/// Note: this implementation uses a deterministic min-label tiebreak (see the
/// `max_by` comparator below) instead of the paper's random tiebreak, to make
/// community assignment reproducible across runs.
pub fn detect_communities(adj: &AdjacencyExport, config: &CommunityConfig) -> Vec<Community> {
    let n = adj.node_ids.len();
    if n == 0 {
        return Vec::new();
    }

    let neighbors = build_undirected_neighbors(n, &adj.edges);
    let labels = run_label_propagation(n, &neighbors, config);
    let communities = assemble_communities(&labels, &adj.node_ids, config.min_community_size);

    info!(
        communities = communities.len(),
        "Community detection complete"
    );

    communities
}

/// Build an undirected neighbour map (index → set of neighbour indices).
fn build_undirected_neighbors(n: usize, edges: &[(usize, usize, f64)]) -> Vec<HashSet<usize>> {
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &(src, tgt, _w) in edges {
        if src != tgt {
            neighbors[src].insert(tgt);
            neighbors[tgt].insert(src);
        }
    }
    neighbors
}

/// Run label-propagation until convergence or `max_iterations`.
fn run_label_propagation(
    n: usize,
    neighbors: &[HashSet<usize>],
    config: &CommunityConfig,
) -> Vec<u32> {
    let mut labels: Vec<u32> = (0..n as u32).collect();

    for iteration in 0..config.max_iterations {
        let mut changed = false;
        for i in 0..n {
            if neighbors[i].is_empty() {
                continue;
            }
            let mut label_counts: HashMap<u32, usize> = HashMap::new();
            for &nbr in &neighbors[i] {
                *label_counts.entry(labels[nbr]).or_default() += 1;
            }
            let best = label_counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                .map(|(label, _)| label)
                .unwrap();
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            debug!(iterations = iteration + 1, "Label propagation converged");
            break;
        }
    }
    labels
}

/// Group labeled nodes into `Community` values and sort by size descending.
fn assemble_communities(labels: &[u32], node_ids: &[String], min_size: usize) -> Vec<Community> {
    let mut groups: HashMap<u32, Vec<CommunityMember>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        groups.entry(label).or_default().push(CommunityMember {
            node_id: node_ids[i].clone(),
            symbol_name: String::new(),
            symbol_type: String::new(),
            file_path: String::new(),
        });
    }

    let mut communities: Vec<Community> = groups
        .into_values()
        .filter(|m| m.len() >= min_size)
        .enumerate()
        .map(|(i, mut m)| {
            m.sort_by(|a, b| a.node_id.cmp(&b.node_id));
            Community {
                community_id: i as u32,
                members: m,
            }
        })
        .collect();

    communities.sort_by_key(|c| std::cmp::Reverse(c.members.len()));
    for (i, c) in communities.iter_mut().enumerate() {
        c.community_id = i as u32;
    }
    communities
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
    fn test_communities_empty() {
        let adj = make_export(0, vec![]);
        let result = detect_communities(&adj, &CommunityConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn test_communities_two_disconnected_clusters() {
        // {0,1,2} triangle, {3,4,5} triangle, no inter-cluster edges.
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
    fn test_communities_min_size_filter() {
        // n0-n1 connected; n2 isolated.
        let adj = make_export(3, vec![(0, 1)]);
        let config = CommunityConfig {
            min_community_size: 2,
            ..Default::default()
        };
        let communities = detect_communities(&adj, &config);
        // Only the {n0,n1} community passes the filter.
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].members.len(), 2);
    }

    #[test]
    fn test_communities_sorted_by_size_descending() {
        // Cluster of 4 (0-3) and cluster of 2 (4-5).
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
                "Communities must be sorted by size descending"
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
}
