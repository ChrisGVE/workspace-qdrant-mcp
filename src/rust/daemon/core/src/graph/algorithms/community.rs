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
    let neighbors = build_undirected_neighbors(&graph.outgoing, &node_ids);
    let labels = run_label_propagation(&node_ids, &neighbors, config, tenant_id);
    let communities = assemble_communities(&labels, &graph.nodes, config.min_community_size);

    info!(
        tenant_id,
        communities = communities.len(),
        "Community detection complete"
    );

    Ok(communities)
}

/// Build an undirected neighbor map from a directed adjacency list.
fn build_undirected_neighbors<'a>(
    outgoing: &'a HashMap<String, Vec<String>>,
    node_ids: &'a [String],
) -> HashMap<&'a str, HashSet<&'a str>> {
    let _ = node_ids; // node_ids used for ordering elsewhere
    let mut neighbors: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (src, targets) in outgoing {
        for tgt in targets {
            neighbors
                .entry(src.as_str())
                .or_default()
                .insert(tgt.as_str());
            neighbors
                .entry(tgt.as_str())
                .or_default()
                .insert(src.as_str());
        }
    }
    neighbors
}

/// Run label-propagation until convergence or max_iterations.
fn run_label_propagation<'a>(
    node_ids: &'a [String],
    neighbors: &HashMap<&'a str, HashSet<&'a str>>,
    config: &CommunityConfig,
    tenant_id: &str,
) -> HashMap<&'a str, u32> {
    let mut labels: HashMap<&str, u32> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i as u32))
        .collect();

    for iteration in 0..config.max_iterations {
        let mut changed = false;
        for id in node_ids {
            let id_str = id.as_str();
            let nbrs = match neighbors.get(id_str) {
                Some(n) if !n.is_empty() => n,
                _ => continue,
            };
            let mut label_counts: HashMap<u32, usize> = HashMap::new();
            for &nbr in nbrs {
                *label_counts.entry(labels[nbr]).or_default() += 1;
            }
            let best = label_counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                .map(|(label, _)| label)
                .unwrap();
            if labels[id_str] != best {
                labels.insert(id_str, best);
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
    labels
}

/// Group labeled nodes into Community values and sort by size descending.
fn assemble_communities(
    labels: &HashMap<&str, u32>,
    nodes: &HashMap<String, super::NodeInfo>,
    min_size: usize,
) -> Vec<Community> {
    let mut groups: HashMap<u32, Vec<CommunityMember>> = HashMap::new();
    for (id, &label) in labels {
        if let Some(info) = nodes.get(*id) {
            groups.entry(label).or_default().push(CommunityMember {
                node_id: id.to_string(),
                symbol_name: info.symbol_name.clone(),
                symbol_type: info.symbol_type.clone(),
                file_path: info.file_path.clone(),
            });
        }
    }

    let mut communities: Vec<Community> = groups
        .into_values()
        .filter(|m| m.len() >= min_size)
        .enumerate()
        .map(|(i, mut m)| {
            m.sort_by(|a, b| a.symbol_name.cmp(&b.symbol_name));
            Community {
                community_id: i as u32,
                members: m,
            }
        })
        .collect();

    communities.sort_by(|a, b| b.members.len().cmp(&a.members.len()));
    for (i, c) in communities.iter_mut().enumerate() {
        c.community_id = i as u32;
    }
    communities
}
