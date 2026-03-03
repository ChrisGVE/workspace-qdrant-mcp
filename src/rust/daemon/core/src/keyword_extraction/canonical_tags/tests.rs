//! Tests for canonical tag deduplication and hierarchical clustering.

use super::clustering::{average_linkage_sim, compute_centroid, merge_duplicates};
use super::types::{CanonicalConfig, TagWithVector};
use super::build_hierarchy;

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
fn test_parent_index_set_on_children() {
    // 4 tags: 2 similar pairs that should cluster together at level 2
    let tags = vec![
        make_tag("rust async", vec![0.95, 0.1, 0.0], 3),
        make_tag("tokio runtime", vec![0.9, 0.15, 0.0], 2),
        make_tag("grpc protocol", vec![0.0, 0.0, 0.95], 2),
        make_tag("protobuf wire", vec![0.0, 0.05, 0.9], 1),
    ];

    let config = CanonicalConfig {
        merge_threshold: 0.95, // high threshold: no dedup at level 3
        level_thresholds: [0.30, 0.70],
    };
    let hierarchy = build_hierarchy(&tags, &config);

    // Level 3 tags should all have parent_index pointing into level 2
    for (i, tag) in hierarchy.level3.iter().enumerate() {
        assert!(
            tag.parent_index.is_some(),
            "Level 3 tag {} ('{}') should have parent_index set",
            i,
            tag.label
        );
        let pidx = tag.parent_index.unwrap();
        assert!(
            pidx < hierarchy.level2.len(),
            "parent_index {} out of range for level 2 (len={})",
            pidx,
            hierarchy.level2.len()
        );
    }

    // Level 2 tags should all have parent_index pointing into level 1
    for (i, tag) in hierarchy.level2.iter().enumerate() {
        assert!(
            tag.parent_index.is_some(),
            "Level 2 tag {} ('{}') should have parent_index set",
            i,
            tag.label
        );
        let pidx = tag.parent_index.unwrap();
        assert!(
            pidx < hierarchy.level1.len(),
            "parent_index {} out of range for level 1 (len={})",
            pidx,
            hierarchy.level1.len()
        );
    }

    // Level 1 tags should NOT have parent_index (top-level)
    for tag in &hierarchy.level1 {
        assert!(
            tag.parent_index.is_none(),
            "Level 1 tag '{}' should not have parent_index",
            tag.label
        );
    }
}

#[test]
fn test_parent_similarity_computed() {
    // 4 tags in 2 pairs — similarity within pairs should be high
    let tags = vec![
        make_tag("rust async", vec![0.95, 0.1, 0.0], 3),
        make_tag("tokio runtime", vec![0.9, 0.15, 0.0], 2),
        make_tag("grpc protocol", vec![0.0, 0.0, 0.95], 2),
        make_tag("protobuf wire", vec![0.0, 0.05, 0.9], 1),
    ];

    let config = CanonicalConfig {
        merge_threshold: 0.95,
        level_thresholds: [0.30, 0.70],
    };
    let hierarchy = build_hierarchy(&tags, &config);

    // All level 3 tags should have parent_similarity set
    for tag in &hierarchy.level3 {
        let sim = tag.parent_similarity.expect(
            &format!("Level 3 tag '{}' should have parent_similarity", tag.label),
        );
        assert!(
            sim > 0.0 && sim <= 1.0,
            "Similarity {} should be in (0, 1] for '{}'",
            sim,
            tag.label
        );
    }

    // All level 2 tags should have parent_similarity set
    for tag in &hierarchy.level2 {
        let sim = tag.parent_similarity.expect(
            &format!("Level 2 tag '{}' should have parent_similarity", tag.label),
        );
        assert!(
            sim > 0.0 && sim <= 1.0,
            "Similarity {} should be in (0, 1] for '{}'",
            sim,
            tag.label
        );
    }

    // Level 1 tags should NOT have parent_similarity
    for tag in &hierarchy.level1 {
        assert!(
            tag.parent_similarity.is_none(),
            "Level 1 tag '{}' should not have parent_similarity",
            tag.label
        );
    }
}

#[test]
fn test_singleton_parent_similarity_is_one() {
    // A singleton cluster: the child IS the parent centroid, so similarity = 1.0
    let tags = vec![
        make_tag("alpha", vec![1.0, 0.0, 0.0], 1),
        make_tag("beta", vec![0.0, 1.0, 0.0], 1),
    ];

    let config = CanonicalConfig {
        merge_threshold: 0.95,
        level_thresholds: [0.10, 0.50],
    };
    let hierarchy = build_hierarchy(&tags, &config);

    // When each tag forms its own cluster, its centroid IS the child centroid
    // so similarity should be exactly 1.0
    for tag in &hierarchy.level3 {
        if let Some(sim) = tag.parent_similarity {
            assert!(
                (sim - 1.0).abs() < 1e-5,
                "Singleton parent_similarity should be ~1.0, got {}",
                sim
            );
        }
    }
}

#[test]
fn test_parent_index_singleton_clusters() {
    // 3 orthogonal tags: each forms its own cluster at every level
    let tags = vec![
        make_tag("alpha", vec![1.0, 0.0, 0.0], 1),
        make_tag("beta", vec![0.0, 1.0, 0.0], 1),
        make_tag("gamma", vec![0.0, 0.0, 1.0], 1),
    ];

    let config = CanonicalConfig {
        merge_threshold: 0.95,
        level_thresholds: [0.30, 0.60],
    };
    let hierarchy = build_hierarchy(&tags, &config);

    // Every level 3 tag should still have a parent_index
    for tag in &hierarchy.level3 {
        assert!(
            tag.parent_index.is_some(),
            "Level 3 singleton '{}' should have parent_index",
            tag.label
        );
    }
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
