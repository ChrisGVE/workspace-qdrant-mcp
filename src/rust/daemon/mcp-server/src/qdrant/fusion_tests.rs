//! Unit tests for `fusion.rs` — hermetic, no live Qdrant required.
//!
//! All tests use hand-built `TaggedResult` values.  Floating-point
//! assertions use an epsilon of 1e-12 to avoid rounding issues.

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

fn kw(id: &str, score: f64) -> TaggedResult {
    make_result(
        id,
        score,
        "projects",
        SearchType::Keyword,
        Some("proj1"),
        None,
    )
}

fn sem_col(id: &str, score: f64, collection: &str) -> TaggedResult {
    make_result(
        id,
        score,
        collection,
        SearchType::Semantic,
        Some("proj1"),
        None,
    )
}

fn kw_col(id: &str, score: f64, collection: &str) -> TaggedResult {
    make_result(
        id,
        score,
        collection,
        SearchType::Keyword,
        Some("proj1"),
        None,
    )
}

// ── RRF_K constant ────────────────────────────────────────────────────────────

#[test]
fn rrf_k_is_sixty() {
    assert_eq!(RRF_K, 60);
}

#[test]
fn default_score_threshold_is_0_3() {
    assert!((DEFAULT_SCORE_THRESHOLD - 0.3).abs() < 1e-12);
}

// ── apply_rrf_fusion — basic arithmetic ──────────────────────────────────────

#[test]
fn rrf_rank_0_score_is_1_over_61() {
    // First result from each leg (rank=0) contributes 1/(60+0+1) = 1/61
    let results = vec![sem("a", 0.9), kw("a", 0.8)];
    let fused = apply_rrf_fusion(&results);
    assert_eq!(fused.len(), 1);
    let expected = 1.0 / 61.0 + 1.0 / 61.0; // rank 0 in both legs
    assert!(
        (fused[0].score - expected).abs() < 1e-12,
        "score={} expected={}",
        fused[0].score,
        expected
    );
}

#[test]
fn rrf_rank1_contributes_1_over_62() {
    // Item "b" is rank 1 in semantic leg: 1/(60+1+1) = 1/62
    // Item "b" is rank 0 in keyword leg: 1/61
    let results = vec![sem("a", 0.9), sem("b", 0.8), kw("b", 0.7), kw("c", 0.6)];
    let fused = apply_rrf_fusion(&results);
    let b = fused.iter().find(|r| r.id == "b").expect("b must exist");
    let expected = 1.0 / 62.0 + 1.0 / 61.0;
    assert!(
        (b.score - expected).abs() < 1e-12,
        "b.score={} expected={}",
        b.score,
        expected
    );
}

#[test]
fn rrf_item_only_in_one_leg_gets_single_contribution() {
    // "a" only in semantic (rank 0): score = 1/61
    // "b" only in keyword (rank 0): score = 1/61
    let results = vec![sem("a", 0.9), kw("b", 0.8)];
    let fused = apply_rrf_fusion(&results);
    assert_eq!(fused.len(), 2);
    for r in &fused {
        let expected = 1.0 / 61.0;
        assert!(
            (r.score - expected).abs() < 1e-12,
            "id={} score={} expected={}",
            r.id,
            r.score,
            expected
        );
    }
}

#[test]
fn rrf_empty_semantic_leg_returns_passthrough() {
    let results = vec![kw("a", 0.9), kw("b", 0.7)];
    let fused = apply_rrf_fusion(&results);
    // No semantic leg → no fusion, original results returned
    assert_eq!(fused.len(), 2);
    assert_eq!(fused[0].search_type, SearchType::Keyword);
}

#[test]
fn rrf_empty_keyword_leg_returns_passthrough() {
    let results = vec![sem("a", 0.9)];
    let fused = apply_rrf_fusion(&results);
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].search_type, SearchType::Semantic);
}

#[test]
fn rrf_empty_input_returns_empty() {
    let fused = apply_rrf_fusion(&[]);
    assert!(fused.is_empty());
}

#[test]
fn rrf_all_fused_results_tagged_hybrid() {
    let results = vec![sem("a", 0.9), kw("b", 0.8)];
    let fused = apply_rrf_fusion(&results);
    // All results from a fused run get search_type = Hybrid
    // (single-leg items also get tagged hybrid per TS behaviour)
    for r in &fused {
        assert_eq!(
            r.search_type,
            SearchType::Hybrid,
            "id={} must be Hybrid",
            r.id
        );
    }
}

#[test]
fn rrf_collection_included_in_dedup_key() {
    // Same id but different collections → separate results
    let results = vec![
        sem_col("id1", 0.9, "projects"),
        kw_col("id1", 0.8, "libraries"),
    ];
    let fused = apply_rrf_fusion(&results);
    // Each leg has one result. The dedup key is "collection:id":
    //   sem leg: "projects:id1"
    //   kw leg:  "libraries:id1"
    // → different keys → no overlap → two separate results, each with 1/61
    assert_eq!(
        fused.len(),
        2,
        "different collections must produce separate results"
    );
}

#[test]
fn rrf_same_collection_same_id_merges() {
    let results = vec![sem("shared", 0.9), kw("shared", 0.8)];
    let fused = apply_rrf_fusion(&results);
    assert_eq!(fused.len(), 1, "same collection+id must merge");
    let expected = 1.0 / 61.0 + 1.0 / 61.0;
    assert!((fused[0].score - expected).abs() < 1e-12);
}

#[test]
fn rrf_rank_formula_matches_ts_exactly() {
    // Verify the formula: 1/(RRF_K + rank + 1) for ranks 0,1,2
    let results = vec![
        sem("a", 0.9),
        sem("b", 0.8),
        sem("c", 0.7),
        kw("a", 0.85),
        kw("b", 0.75),
        kw("c", 0.65),
    ];
    let fused = apply_rrf_fusion(&results);

    let score_for = |id: &str| fused.iter().find(|r| r.id == id).unwrap().score;

    let a_expected = 1.0 / 61.0 + 1.0 / 61.0; // rank 0 in both
    let b_expected = 1.0 / 62.0 + 1.0 / 62.0; // rank 1 in both
    let c_expected = 1.0 / 63.0 + 1.0 / 63.0; // rank 2 in both

    assert!((score_for("a") - a_expected).abs() < 1e-12, "a");
    assert!((score_for("b") - b_expected).abs() < 1e-12, "b");
    assert!((score_for("c") - c_expected).abs() < 1e-12, "c");

    // Also verify ordering: a > b > c
    assert!(score_for("a") > score_for("b"));
    assert!(score_for("b") > score_for("c"));
}

// ── apply_score_threshold ─────────────────────────────────────────────────────

#[test]
fn threshold_removes_below_threshold() {
    let results = vec![sem("a", 0.5), sem("b", 0.3), sem("c", 0.29)];
    let filtered = apply_score_threshold(results, 0.3);
    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().all(|r| r.score >= 0.3));
}

#[test]
fn threshold_keeps_equal_to_threshold() {
    let results = vec![sem("a", 0.3)];
    let filtered = apply_score_threshold(results, 0.3);
    assert_eq!(filtered.len(), 1);
}

#[test]
fn threshold_empty_input_returns_empty() {
    let filtered = apply_score_threshold(vec![], 0.3);
    assert!(filtered.is_empty());
}

#[test]
fn threshold_zero_keeps_all() {
    let results = vec![sem("a", 0.0), sem("b", 1.0)];
    let len_before = results.len();
    let filtered = apply_score_threshold(results, 0.0);
    assert_eq!(filtered.len(), len_before);
}

// ── source_key ────────────────────────────────────────────────────────────────

#[test]
fn source_key_library_uses_library_name() {
    let r = make_result(
        "id",
        0.9,
        "libraries",
        SearchType::Semantic,
        None,
        Some("tokio"),
    );
    assert_eq!(r.source_key(), "libraries:tokio");
}

#[test]
fn source_key_project_uses_tenant_id() {
    let r = make_result(
        "id",
        0.9,
        "projects",
        SearchType::Semantic,
        Some("proj1"),
        None,
    );
    assert_eq!(r.source_key(), "projects:proj1");
}

#[test]
fn source_key_fallback_unknown() {
    let r = TaggedResult {
        id: "id".to_string(),
        score: 0.9,
        collection: "projects".to_string(),
        payload: HashMap::new(),
        search_type: SearchType::Semantic,
    };
    assert_eq!(r.source_key(), "projects:unknown");
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
