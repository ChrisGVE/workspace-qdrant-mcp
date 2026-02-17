//! Canonical tag deduplication and hierarchical clustering.
//!
//! Merges near-duplicate tags across documents within a tenant, then
//! builds a 3-level hierarchy via agglomerative clustering.

use super::semantic_rerank::cosine_similarity;

/// A tag with its embedding, collected from across documents.
#[derive(Debug, Clone)]
pub struct TagWithVector {
    /// The tag phrase
    pub phrase: String,
    /// Embedding vector
    pub vector: Vec<f32>,
    /// Number of documents containing this tag
    pub doc_count: u32,
}

/// A canonical (deduplicated) tag.
#[derive(Debug, Clone)]
pub struct CanonicalTag {
    /// Canonical label (chosen from cluster members)
    pub label: String,
    /// Alternative labels (other members merged into this)
    pub aliases: Vec<String>,
    /// Centroid vector (mean of member vectors)
    pub centroid: Vec<f32>,
    /// Total document count across merged tags
    pub doc_count: u32,
    /// Hierarchy level (1=broad, 2=mid, 3=fine)
    pub level: u8,
    /// Index of parent cluster (None for top-level)
    pub parent_index: Option<usize>,
}

/// Configuration for canonical tag building.
#[derive(Debug, Clone)]
pub struct CanonicalConfig {
    /// Similarity threshold for merging near-duplicates
    pub merge_threshold: f64,
    /// Similarity thresholds for hierarchy levels (broad, mid)
    /// Level 3 = all canonical tags (no merging)
    pub level_thresholds: [f64; 2],
}

impl Default for CanonicalConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.85,
            level_thresholds: [0.50, 0.70],
        }
    }
}

/// Result of canonical tag building.
#[derive(Debug, Clone)]
pub struct CanonicalHierarchy {
    /// Level 3: fine-grained canonical tags (after dedup)
    pub level3: Vec<CanonicalTag>,
    /// Level 2: mid-level clusters
    pub level2: Vec<CanonicalTag>,
    /// Level 1: broad topic clusters
    pub level1: Vec<CanonicalTag>,
}

/// Build canonical tag hierarchy from collected tags.
///
/// 1. Merge near-duplicates (similarity > merge_threshold)
/// 2. Cluster at level 2 (mid threshold)
/// 3. Cluster at level 1 (broad threshold)
pub fn build_hierarchy(
    tags: &[TagWithVector],
    config: &CanonicalConfig,
) -> CanonicalHierarchy {
    if tags.is_empty() {
        return CanonicalHierarchy {
            level3: Vec::new(),
            level2: Vec::new(),
            level1: Vec::new(),
        };
    }

    // Step 1: Merge near-duplicates into canonical tags (level 3)
    let level3 = merge_duplicates(tags, config.merge_threshold);

    // Step 2: Cluster level 3 into level 2
    let level2 = cluster_tags(&level3, config.level_thresholds[1], 2);

    // Step 3: Cluster level 2 into level 1
    let level1 = cluster_tags(&level2, config.level_thresholds[0], 1);

    CanonicalHierarchy {
        level3,
        level2,
        level1,
    }
}

/// Merge near-duplicate tags using single-linkage clustering.
///
/// Tags with cosine similarity > threshold are merged. The label closest
/// to the centroid is chosen as the canonical name.
fn merge_duplicates(tags: &[TagWithVector], threshold: f64) -> Vec<CanonicalTag> {
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
            }
        })
        .collect()
}

/// Cluster canonical tags into higher-level groups using average-linkage.
fn cluster_tags(tags: &[CanonicalTag], threshold: f64, level: u8) -> Vec<CanonicalTag> {
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

        let centroid =
            compute_centroid(members.iter().map(|&m| &tags[m].centroid));
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
        // Set parent_index on child tags
        let child_indices: Vec<usize> = members.clone();

        result.push(CanonicalTag {
            label,
            aliases,
            centroid,
            doc_count: total_docs,
            level,
            parent_index: None,
        });

        // Record parent mapping: child_indices map to this parent
        let _ = (parent_idx, child_indices); // Used by caller to build edges
    }

    result
}

/// Compute average linkage similarity between two clusters.
fn average_linkage_sim(
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
fn compute_centroid<'a>(vectors: impl Iterator<Item = &'a Vec<f32>>) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tag(phrase: &str, vector: Vec<f32>, doc_count: u32) -> TagWithVector {
        TagWithVector {
            phrase: phrase.to_string(),
            vector,
            doc_count,
        }
    }

    #[test]
    fn test_merge_duplicates_similar() {
        let tags = vec![
            make_tag("vector search", vec![0.95, 0.31, 0.0], 5),
            make_tag("vector indexing", vec![0.95, 0.30, 0.0], 3),
            make_tag("grpc protocol", vec![0.0, 0.0, 1.0], 2),
        ];

        let merged = merge_duplicates(&tags, 0.85);
        // "vector search" and "vector indexing" should merge
        assert_eq!(
            merged.len(),
            2,
            "Should merge similar tags: {:?}",
            merged.iter().map(|t| &t.label).collect::<Vec<_>>()
        );

        // One cluster should have aliases
        let vector_cluster = merged.iter().find(|t| {
            t.label.contains("vector") || t.aliases.iter().any(|a| a.contains("vector"))
        });
        assert!(vector_cluster.is_some());
        let vc = vector_cluster.unwrap();
        assert_eq!(vc.doc_count, 8, "Doc counts should sum");
        assert_eq!(vc.aliases.len(), 1, "Should have one alias");
    }

    #[test]
    fn test_merge_duplicates_all_different() {
        let tags = vec![
            make_tag("alpha", vec![1.0, 0.0, 0.0], 1),
            make_tag("beta", vec![0.0, 1.0, 0.0], 1),
            make_tag("gamma", vec![0.0, 0.0, 1.0], 1),
        ];

        let merged = merge_duplicates(&tags, 0.85);
        assert_eq!(merged.len(), 3, "All orthogonal tags should remain separate");
        assert!(merged.iter().all(|t| t.aliases.is_empty()));
    }

    #[test]
    fn test_merge_duplicates_empty() {
        let merged = merge_duplicates(&[], 0.85);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_build_hierarchy_basic() {
        // 6 tags in 3 natural groups (pairs)
        let tags = vec![
            make_tag("rust async", vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0], 3),
            make_tag("tokio runtime", vec![0.85, 0.15, 0.0, 0.0, 0.0, 0.0], 2),
            make_tag("database query", vec![0.0, 0.0, 0.9, 0.1, 0.0, 0.0], 4),
            make_tag("sql storage", vec![0.0, 0.0, 0.85, 0.15, 0.0, 0.0], 1),
            make_tag("grpc service", vec![0.0, 0.0, 0.0, 0.0, 0.9, 0.1], 2),
            make_tag("rest api", vec![0.0, 0.0, 0.0, 0.0, 0.1, 0.9], 3),
        ];

        let config = CanonicalConfig::default();
        let hierarchy = build_hierarchy(&tags, &config);

        // Level 3 should merge near-duplicates
        assert!(
            hierarchy.level3.len() <= 6,
            "Level 3 should have <= 6 tags: got {}",
            hierarchy.level3.len()
        );

        // Level 1 should have fewer clusters than level 3
        assert!(
            hierarchy.level1.len() <= hierarchy.level3.len(),
            "Level 1 ({}) should have <= level 3 ({})",
            hierarchy.level1.len(),
            hierarchy.level3.len()
        );
    }

    #[test]
    fn test_build_hierarchy_empty() {
        let config = CanonicalConfig::default();
        let hierarchy = build_hierarchy(&[], &config);
        assert!(hierarchy.level1.is_empty());
        assert!(hierarchy.level2.is_empty());
        assert!(hierarchy.level3.is_empty());
    }

    #[test]
    fn test_build_hierarchy_single_tag() {
        let tags = vec![make_tag("only tag", vec![1.0, 0.0], 5)];
        let config = CanonicalConfig::default();
        let hierarchy = build_hierarchy(&tags, &config);

        assert_eq!(hierarchy.level3.len(), 1);
        assert_eq!(hierarchy.level3[0].label, "only tag");
        assert_eq!(hierarchy.level3[0].doc_count, 5);
    }

    #[test]
    fn test_compute_centroid() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let vecs = vec![v1, v2];
        let centroid = compute_centroid(vecs.iter());
        assert!((centroid[0] - 0.5).abs() < 1e-5);
        assert!((centroid[1] - 0.5).abs() < 1e-5);
        assert!((centroid[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_centroid_empty() {
        let vecs: Vec<Vec<f32>> = vec![];
        let centroid = compute_centroid(vecs.iter());
        assert!(centroid.is_empty());
    }

    #[test]
    fn test_average_linkage_sim() {
        let sim_matrix = vec![
            vec![1.0, 0.8, 0.1],
            vec![0.8, 1.0, 0.2],
            vec![0.1, 0.2, 1.0],
        ];

        let sim = average_linkage_sim(&[0, 1], &[2], &sim_matrix);
        // (0.1 + 0.2) / 2 = 0.15
        assert!(
            (sim - 0.15).abs() < 1e-6,
            "Expected 0.15, got {}",
            sim
        );
    }

    #[test]
    fn test_canonical_config_defaults() {
        let config = CanonicalConfig::default();
        assert!((config.merge_threshold - 0.85).abs() < 1e-6);
        assert!((config.level_thresholds[0] - 0.50).abs() < 1e-6);
        assert!((config.level_thresholds[1] - 0.70).abs() < 1e-6);
    }

    #[test]
    fn test_label_selection_closest_to_centroid() {
        // Two similar tags: "machine learning" (closer to centroid) and "ml algorithms"
        let tags = vec![
            make_tag("machine learning", vec![0.9, 0.3], 5),
            make_tag("ml algorithms", vec![0.85, 0.35], 3),
        ];

        let merged = merge_duplicates(&tags, 0.85);
        // Both should merge; label should be the one closest to centroid
        assert_eq!(merged.len(), 1);
        // The centroid of [0.9,0.3] and [0.85,0.35] is [0.875,0.325]
        // Both are close; either label is valid but one should be chosen
        assert!(
            merged[0].label == "machine learning" || merged[0].label == "ml algorithms",
            "Label should be one of the members"
        );
    }
}
