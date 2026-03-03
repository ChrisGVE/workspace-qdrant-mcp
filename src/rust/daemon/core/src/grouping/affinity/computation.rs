/// Pairwise affinity computation and group formation algorithms.

use crate::keyword_extraction::semantic_rerank::cosine_similarity;
use crate::tagging::aggregate_document_embedding;

// ---- Types -----------------------------------------------------------------

/// A computed affinity between two projects.
#[derive(Debug, Clone)]
pub struct ProjectAffinity {
    pub tenant_a: String,
    pub tenant_b: String,
    pub similarity: f64,
}

// ---- Pairwise computation --------------------------------------------------

/// Compute pairwise cosine similarities and return pairs above threshold.
pub fn compute_pairwise_affinities(
    embeddings: &[(String, Vec<f32>)],
    threshold: f64,
) -> Vec<ProjectAffinity> {
    let mut affinities = Vec::new();

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            if sim >= threshold {
                affinities.push(ProjectAffinity {
                    tenant_a: embeddings[i].0.clone(),
                    tenant_b: embeddings[j].0.clone(),
                    similarity: sim,
                });
            }
        }
    }

    affinities
}

// ---- Group formation -------------------------------------------------------

/// Build connected components from pairwise affinities.
///
/// Groups projects transitively: if A~B and B~C, then {A,B,C} form one group.
pub fn build_affinity_groups(affinities: &[ProjectAffinity]) -> Vec<Vec<String>> {
    use std::collections::{HashMap, HashSet};

    if affinities.is_empty() {
        return Vec::new();
    }

    // Build adjacency list
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    for a in affinities {
        adj.entry(&a.tenant_a).or_default().insert(&a.tenant_b);
        adj.entry(&a.tenant_b).or_default().insert(&a.tenant_a);
    }

    // BFS to find connected components
    let mut visited: HashSet<&str> = HashSet::new();
    let mut groups: Vec<Vec<String>> = Vec::new();

    for start in adj.keys() {
        if visited.contains(*start) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = vec![*start];

        while let Some(node) = queue.pop() {
            if !visited.insert(node) {
                continue;
            }
            component.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        queue.push(neighbor);
                    }
                }
            }
        }

        component.sort();
        groups.push(component);
    }

    groups.sort_by(|a, b| a[0].cmp(&b[0]));
    groups
}

// ---- Helpers ---------------------------------------------------------------

/// Generate a deterministic group_id for an affinity group.
///
/// Uses sorted tenant_ids to produce a stable hash.
pub(crate) fn affinity_group_id(members: &[String]) -> String {
    use sha2::{Digest, Sha256};

    let mut sorted = members.to_vec();
    sorted.sort();
    let input = sorted.join("|");
    let hash = Sha256::digest(input.as_bytes());
    format!("affinity:{:x}", hash)[..24].to_string()
}

/// Compute the mean embedding of a group of projects.
pub(crate) fn group_mean_embedding(
    members: &[String],
    embeddings: &[(String, Vec<f32>)],
) -> Option<Vec<f32>> {
    let member_embeddings: Vec<Vec<f32>> = members
        .iter()
        .filter_map(|t| {
            embeddings
                .iter()
                .find(|(id, _)| id == t)
                .map(|(_, e)| e.clone())
        })
        .collect();

    aggregate_document_embedding(&member_embeddings)
}

/// Compute mean similarity among all pairs within a group.
pub(crate) fn compute_group_mean_similarity(
    members: &[String],
    affinities: &[ProjectAffinity],
) -> f64 {
    let mut total = 0.0;
    let mut count = 0;

    for a in affinities {
        let a_in = members.contains(&a.tenant_a);
        let b_in = members.contains(&a.tenant_b);
        if a_in && b_in {
            total += a.similarity;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}
