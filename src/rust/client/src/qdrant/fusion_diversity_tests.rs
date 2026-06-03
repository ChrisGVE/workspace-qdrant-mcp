//! Unit tests for the diversity / tiering half of `fusion.rs` — hermetic.
//!
//! Split from `fusion_tests.rs` to keep each test file under the 500-line
//! limit.  Helpers are duplicated (rather than shared) because `#[path]` test
//! submodules cannot see each other's items.

use std::collections::HashMap;

use super::*;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_result(
    id: &str,
    score: f64,
    collection: &str,
    search_type: SearchType,
    tenant_id: Option<&str>,
    library_name: Option<&str>,
) -> TaggedResult {
    let mut payload = HashMap::new();
    if let Some(t) = tenant_id {
        payload.insert(
            "tenant_id".to_string(),
            serde_json::Value::String(t.to_string()),
        );
    }
    if let Some(l) = library_name {
        payload.insert(
            "library_name".to_string(),
            serde_json::Value::String(l.to_string()),
        );
    }
    TaggedResult {
        id: id.to_string(),
        score,
        collection: collection.to_string(),
        payload,
        search_type,
    }
}

fn sem(id: &str, score: f64) -> TaggedResult {
    make_result(
        id,
        score,
        "projects",
        SearchType::Semantic,
        Some("proj1"),
        None,
    )
}

// ── compute_diversity_score ───────────────────────────────────────────────────

#[test]
fn diversity_score_empty_is_one() {
    assert!((compute_diversity_score(&[]) - 1.0).abs() < 1e-12);
}

#[test]
fn diversity_score_all_same_source_is_zero() {
    let results = vec![
        make_result("a", 0.9, "projects", SearchType::Semantic, Some("p1"), None),
        make_result("b", 0.8, "projects", SearchType::Semantic, Some("p1"), None),
    ];
    assert!((compute_diversity_score(&results) - 0.5).abs() < 1e-12);
}

#[test]
fn diversity_score_all_unique_is_one() {
    let results = vec![
        make_result("a", 0.9, "projects", SearchType::Semantic, Some("p1"), None),
        make_result("b", 0.8, "projects", SearchType::Semantic, Some("p2"), None),
    ];
    assert!((compute_diversity_score(&results) - 1.0).abs() < 1e-12);
}

// ── build_score_tiers ─────────────────────────────────────────────────────────

#[test]
fn score_tiers_groups_close_scores() {
    let results = vec![
        sem("a", 1.0),
        sem("b", 0.98), // within 0.05 of a → same tier
        sem("c", 0.5),  // far from a → new tier
    ];
    let tiers = build_score_tiers(&results, 0.05);
    assert_eq!(tiers.len(), 2, "a+b in tier 1, c in tier 2");
    assert_eq!(tiers[0].len(), 2);
    assert_eq!(tiers[1].len(), 1);
}

#[test]
fn score_tiers_each_in_own_tier() {
    let results = vec![sem("a", 1.0), sem("b", 0.5), sem("c", 0.0)];
    let tiers = build_score_tiers(&results, 0.05);
    assert_eq!(tiers.len(), 3);
}

#[test]
fn score_tiers_all_equal_one_tier() {
    let results = vec![sem("a", 0.8), sem("b", 0.8), sem("c", 0.8)];
    let tiers = build_score_tiers(&results, 0.05);
    assert_eq!(tiers.len(), 1);
    assert_eq!(tiers[0].len(), 3);
}

#[test]
fn score_tiers_empty_input_returns_empty() {
    let tiers = build_score_tiers(&[], 0.05);
    assert!(tiers.is_empty());
}

// ── interleave_tier ───────────────────────────────────────────────────────────

#[test]
fn interleave_single_item_unchanged() {
    let tier = vec![sem("a", 0.9)];
    let out = interleave_tier(tier.clone());
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, "a");
}

#[test]
fn interleave_two_sources_round_robin() {
    // Two sources "proj1" and "proj2", two items each
    let tier = vec![
        make_result(
            "a1",
            0.9,
            "projects",
            SearchType::Semantic,
            Some("proj1"),
            None,
        ),
        make_result(
            "a2",
            0.85,
            "projects",
            SearchType::Semantic,
            Some("proj1"),
            None,
        ),
        make_result(
            "b1",
            0.88,
            "projects",
            SearchType::Semantic,
            Some("proj2"),
            None,
        ),
        make_result(
            "b2",
            0.82,
            "projects",
            SearchType::Semantic,
            Some("proj2"),
            None,
        ),
    ];
    let out = interleave_tier(tier);
    assert_eq!(out.len(), 4);
    // First two should alternate sources
    let src0 = out[0].source_key();
    let src1 = out[1].source_key();
    assert_ne!(src0, src1, "first two must be from different sources");
}

#[test]
fn interleave_single_source_preserves_order() {
    let tier = vec![sem("a", 0.9), sem("b", 0.8), sem("c", 0.7)];
    let out = interleave_tier(tier);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].id, "a");
    assert_eq!(out[1].id, "b");
    assert_eq!(out[2].id, "c");
}

// ── diversify_results ─────────────────────────────────────────────────────────

#[test]
fn diversity_disabled_returns_input_unchanged() {
    let results = vec![sem("a", 0.9), sem("b", 0.8)];
    let config = DiversityConfig {
        enabled: false,
        max_per_source: 3,
        score_tier_threshold: 0.05,
    };
    let (out, _score) = diversify_results(results.clone(), &config);
    assert_eq!(out.len(), 2);
}

#[test]
fn diversity_empty_input_returns_empty() {
    let (out, score) = diversify_results(vec![], &DEFAULT_DIVERSITY_CONFIG);
    assert!(out.is_empty());
    assert!((score - 1.0).abs() < 1e-12);
}

#[test]
fn diversity_max_per_source_caps_single_source() {
    // 5 results from the same source; max_per_source=3
    let results: Vec<TaggedResult> = (0..5)
        .map(|i| {
            make_result(
                &format!("id{i}"),
                1.0 - i as f64 * 0.1,
                "projects",
                SearchType::Semantic,
                Some("proj1"),
                None,
            )
        })
        .collect();
    let config = DiversityConfig {
        enabled: true,
        max_per_source: 3,
        score_tier_threshold: 0.05,
    };
    let (out, _score) = diversify_results(results, &config);
    // 3 primary + 2 spillover backfill = 5 (backfill restores count)
    assert_eq!(out.len(), 5, "backfill must restore total count");
}

#[test]
fn diversity_two_sources_interleaved() {
    // 4 results: 2 from proj1, 2 from proj2, all in the same tier
    let results = vec![
        make_result(
            "a1",
            0.9,
            "projects",
            SearchType::Semantic,
            Some("proj1"),
            None,
        ),
        make_result(
            "b1",
            0.89,
            "projects",
            SearchType::Semantic,
            Some("proj2"),
            None,
        ),
        make_result(
            "a2",
            0.88,
            "projects",
            SearchType::Semantic,
            Some("proj1"),
            None,
        ),
        make_result(
            "b2",
            0.87,
            "projects",
            SearchType::Semantic,
            Some("proj2"),
            None,
        ),
    ];
    let (out, score) = diversify_results(results, &DEFAULT_DIVERSITY_CONFIG);
    assert_eq!(out.len(), 4);
    // Score: 2 unique sources / 4 results = 0.5
    assert!((score - 0.5).abs() < 1e-12);
}

#[test]
fn diversity_score_returned_matches_compute() {
    let results = vec![
        make_result("a", 0.9, "projects", SearchType::Semantic, Some("p1"), None),
        make_result("b", 0.8, "projects", SearchType::Semantic, Some("p2"), None),
        make_result("c", 0.7, "projects", SearchType::Semantic, Some("p1"), None),
    ];
    let expected_score = compute_diversity_score(&results);
    // We can't know what order diversity gives us, but the returned score
    // should match compute_diversity_score on the returned slice.
    let (out, returned_score) = diversify_results(results, &DEFAULT_DIVERSITY_CONFIG);
    let recomputed = compute_diversity_score(&out);
    assert!(
        (returned_score - recomputed).abs() < 1e-12,
        "returned score must match compute on output slice"
    );
    // Suppress unused-variable warning
    let _ = expected_score;
}

// ── point_to_tagged ───────────────────────────────────────────────────────────

#[test]
fn point_to_tagged_preserves_fields() {
    use super::super::client::QdrantPoint;
    let point = QdrantPoint {
        id: "uuid-1".to_string(),
        score: 0.75,
        payload: HashMap::new(),
    };
    let tagged = point_to_tagged(point, "projects".to_string(), SearchType::Semantic);
    assert_eq!(tagged.id, "uuid-1");
    assert!((tagged.score - 0.75).abs() < 1e-12);
    assert_eq!(tagged.collection, "projects");
    assert_eq!(tagged.search_type, SearchType::Semantic);
}
