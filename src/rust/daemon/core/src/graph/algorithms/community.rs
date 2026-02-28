/// Community detection using label propagation algorithm.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::load_adjacency_graph;

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
