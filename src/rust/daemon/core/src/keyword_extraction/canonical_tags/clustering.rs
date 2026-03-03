//! Clustering algorithms for canonical tag deduplication and hierarchy building.

use super::super::semantic_rerank::cosine_similarity;
use super::types::{CanonicalTag, TagWithVector};

/// Merge near-duplicate tags using single-linkage clustering.
///
/// Tags with cosine similarity > threshold are merged. The label closest
/// to the centroid is chosen as the canonical name.
pub(super) fn merge_duplicates(tags: &[TagWithVector], threshold: f64) -> Vec<CanonicalTag> {
    let n = tags.len();
    let mut cluster_id: Vec<Option<usize>> = vec![None; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    // Greedy merging: for each unassigned tag, find all similar tags
    for i in 0..n {
        if cluster_id[i].is_some() {
            continue;
        }

        let cid = clusters.len();
        let mut members = vec![i];
        cluster_id[i] = Some(cid);

        for j in (i + 1)..n {
            if cluster_id[j].is_some() {
                continue;
            }
            let sim = cosine_similarity(&tags[i].vector, &tags[j].vector);
            if sim > threshold {
                members.push(j);
                cluster_id[j] = Some(cid);
            }
        }

        clusters.push(members);
    }

    // Build canonical tags from clusters
    clusters
        .iter()
        .map(|members| {
            let centroid = compute_centroid(members.iter().map(|&i| &tags[i].vector));
            let total_docs: u32 = members.iter().map(|&i| tags[i].doc_count).sum();

            // Choose label: phrase closest to centroid
            let label_idx = members
                .iter()
                .max_by(|&&a, &&b| {
                    let sim_a = cosine_similarity(&tags[a].vector, &centroid);
                    let sim_b = cosine_similarity(&tags[b].vector, &centroid);
                    sim_a
                        .partial_cmp(&sim_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(members[0]);

            let label = tags[label_idx].phrase.clone();
            let aliases: Vec<String> = members
                .iter()
                .filter(|&&i| i != label_idx)
                .map(|&i| tags[i].phrase.clone())
                .collect();

            CanonicalTag {
                label,
                aliases,
                centroid,
                doc_count: total_docs,
                level: 3,
                parent_index: None,
                parent_similarity: None,
            }
        })
        .collect()
}

/// Cluster canonical tags into higher-level groups using average-linkage.
///
/// Sets `parent_index` on each input tag to point to the index of its
/// parent cluster in the returned output vector.
pub(super) fn cluster_tags(
    tags: &mut [CanonicalTag],
    threshold: f64,
    level: u8,
) -> Vec<CanonicalTag> {
    let n = tags.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute pairwise similarity matrix
    let mut sim_matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&tags[i].centroid, &tags[j].centroid);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;
        }
    }

    // Agglomerative clustering with average linkage
    let mut cluster_id: Vec<usize> = (0..n).collect();
    let mut active: Vec<bool> = vec![true; n];
    let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    loop {
        // Find the most similar pair of active clusters
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_pair = (0, 0);

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                // Average linkage: mean similarity between all pairs
                let avg_sim = average_linkage_sim(
                    &cluster_members[i],
                    &cluster_members[j],
                    &sim_matrix,
                );
                if avg_sim > best_sim {
                    best_sim = avg_sim;
                    best_pair = (i, j);
                }
            }
        }

        if best_sim < threshold || best_pair == (0, 0) && n > 1 {
            // Check if we found any valid pair
            if best_sim < threshold {
                break;
            }
        }

        // Merge best_pair.1 into best_pair.0
        let (a, b) = best_pair;
        let members_b = cluster_members[b].clone();
        cluster_members[a].extend(members_b);
        active[b] = false;

        // Update cluster assignments
        for &m in &cluster_members[a] {
            cluster_id[m] = a;
        }

        // Check if only one active cluster remains
        if active.iter().filter(|&&a| a).count() <= 1 {
            break;
        }
    }

    // Suppress unused warning: cluster_id is maintained for correctness but
    // the final result is built from active cluster indices directly.
    let _ = cluster_id;

    // Build output from active clusters
    let mut result = Vec::new();
    for i in 0..n {
        if !active[i] {
            continue;
        }

        let members = &cluster_members[i];
        if members.is_empty() {
            continue;
        }

        let centroid = compute_centroid(members.iter().map(|&m| &tags[m].centroid));
        let total_docs: u32 = members.iter().map(|&m| tags[m].doc_count).sum();

        // Choose label: member closest to centroid
        let label_idx = members
            .iter()
            .max_by(|&&a, &&b| {
                let sim_a = cosine_similarity(&tags[a].centroid, &centroid);
                let sim_b = cosine_similarity(&tags[b].centroid, &centroid);
                sim_a
                    .partial_cmp(&sim_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(members[0]);

        let label = tags[label_idx].label.clone();
        let aliases: Vec<String> = members
            .iter()
            .filter(|&&m| m != label_idx)
            .map(|&m| tags[m].label.clone())
            .collect();

        let parent_idx = result.len();

        // Set parent_index and parent_similarity on each input tag
        for &m in members {
            tags[m].parent_index = Some(parent_idx);
            tags[m].parent_similarity =
                Some(cosine_similarity(&tags[m].centroid, &centroid));
        }

        result.push(CanonicalTag {
            label,
            aliases,
            centroid,
            doc_count: total_docs,
            level,
            parent_index: None,
            parent_similarity: None,
        });
    }

    result
}

/// Compute average linkage similarity between two clusters.
pub(super) fn average_linkage_sim(
    cluster_a: &[usize],
    cluster_b: &[usize],
    sim_matrix: &[Vec<f64>],
) -> f64 {
    if cluster_a.is_empty() || cluster_b.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for &a in cluster_a {
        for &b in cluster_b {
            total += sim_matrix[a][b];
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Compute centroid (mean) of a set of vectors.
pub(super) fn compute_centroid<'a>(vectors: impl Iterator<Item = &'a Vec<f32>>) -> Vec<f32> {
    let vecs: Vec<&Vec<f32>> = vectors.collect();
    if vecs.is_empty() {
        return Vec::new();
    }

    let dim = vecs[0].len();
    let mut centroid = vec![0.0f32; dim];
    let n = vecs.len() as f32;

    for v in &vecs {
        if v.len() != dim {
            continue;
        }
        for (i, val) in v.iter().enumerate() {
            centroid[i] += val;
        }
    }

    for val in &mut centroid {
        *val /= n;
    }

    centroid
}
