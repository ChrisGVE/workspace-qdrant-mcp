use std::collections::HashMap;

use super::*;
use crate::storage::SearchResult;

fn make_result(id: &str, score: f32, source_field: &str, source_val: &str) -> SearchResult {
    let mut payload = HashMap::new();
    payload.insert(source_field.to_string(), serde_json::json!(source_val));

    SearchResult {
        id: id.to_string(),
        score,
        payload,
        dense_vector: None,
        sparse_vector: None,
    }
}

fn make_project_result(id: &str, score: f32, tenant: &str) -> SearchResult {
    make_result(id, score, "tenant_id", tenant)
}

fn make_library_result(id: &str, score: f32, library: &str) -> SearchResult {
    make_result(id, score, "library_name", library)
}

/// Helper: build a result with both tenant_id and file_path set.
fn make_file_result(id: &str, score: f32, tenant: &str, file_path: &str) -> SearchResult {
    let mut payload = HashMap::new();
    payload.insert("tenant_id".to_string(), serde_json::json!(tenant));
    payload.insert("file_path".to_string(), serde_json::json!(file_path));

    SearchResult {
        id: id.to_string(),
        score,
        payload,
        dense_vector: None,
        sparse_vector: None,
    }
}

// ─── Extract source tests ───────────────────────────────────────────

#[test]
fn test_extract_source_library() {
    let r = make_library_result("r1", 0.9, "my-lib");
    assert_eq!(extract_source(&r), "my-lib");
}

#[test]
fn test_extract_source_project() {
    let r = make_project_result("r1", 0.9, "proj-a");
    assert_eq!(extract_source(&r), "proj-a");
}

#[test]
fn test_extract_source_unknown() {
    let r = SearchResult {
        id: "r1".to_string(),
        score: 0.9,
        payload: HashMap::new(),
        dense_vector: None,
        sparse_vector: None,
    };
    assert_eq!(extract_source(&r), "unknown");
}

// ─── Extract file tests ────────────────────────────────────────────

#[test]
fn test_extract_file_from_file_path() {
    let r = make_file_result("r1", 0.9, "proj-a", "/src/main.rs");
    assert_eq!(extract_file(&r), "/src/main.rs");
}

#[test]
fn test_extract_file_from_relative_path() {
    let mut payload = HashMap::new();
    payload.insert("relative_path".to_string(), serde_json::json!("src/lib.rs"));
    let r = SearchResult {
        id: "r1".to_string(),
        score: 0.9,
        payload,
        dense_vector: None,
        sparse_vector: None,
    };
    assert_eq!(extract_file(&r), "src/lib.rs");
}

#[test]
fn test_extract_file_fallback_to_id() {
    let r = SearchResult {
        id: "fallback-id".to_string(),
        score: 0.9,
        payload: HashMap::new(),
        dense_vector: None,
        sparse_vector: None,
    };
    assert_eq!(extract_file(&r), "fallback-id");
}

// ─── Extract project tests ─────────────────────────────────────────

#[test]
fn test_extract_project_from_tenant() {
    let r = make_file_result("r1", 0.9, "proj-x", "/src/main.rs");
    assert_eq!(extract_project(&r), "proj-x");
}

#[test]
fn test_extract_project_from_library() {
    let r = make_library_result("r1", 0.9, "my-lib");
    assert_eq!(extract_project(&r), "my-lib");
}

// ─── Score tier tests ───────────────────────────────────────────────

#[test]
fn test_build_tiers_single_tier() {
    let results = vec![
        make_project_result("r1", 0.90, "a"),
        make_project_result("r2", 0.88, "b"),
        make_project_result("r3", 0.87, "c"),
    ];
    let tiers = build_score_tiers(&results, 0.05);
    assert_eq!(tiers.len(), 1, "All within 0.05 of 0.90");
}

#[test]
fn test_build_tiers_multiple() {
    let results = vec![
        make_project_result("r1", 0.95, "a"),
        make_project_result("r2", 0.93, "b"),
        // Gap > 0.05
        make_project_result("r3", 0.80, "c"),
        make_project_result("r4", 0.78, "d"),
    ];
    let tiers = build_score_tiers(&results, 0.05);
    assert_eq!(tiers.len(), 2);
    assert_eq!(tiers[0].len(), 2);
    assert_eq!(tiers[1].len(), 2);
}

#[test]
fn test_build_tiers_empty() {
    let tiers = build_score_tiers(&[], 0.05);
    assert!(tiers.is_empty());
}

// ─── Tier-based diversity tests ────────────────────────────────────

#[test]
fn test_diversify_disabled() {
    let config = DiversityConfig {
        enabled: false,
        ..Default::default()
    };
    let results = vec![
        make_project_result("r1", 0.9, "a"),
        make_project_result("r2", 0.8, "a"),
    ];
    let diversified = diversify_results(results.clone(), &config);
    assert_eq!(diversified.len(), results.len());
}

#[test]
fn test_diversify_caps_per_source() {
    let config = DiversityConfig {
        enabled: true,
        max_per_source: 2,
        score_tier_threshold: 0.05,
    };
    let results = vec![
        make_project_result("a1", 0.95, "a"),
        make_project_result("a2", 0.93, "a"),
        make_project_result("a3", 0.91, "a"),
        make_project_result("a4", 0.89, "a"),
        make_project_result("b1", 0.87, "b"),
    ];
    let diversified = diversify_results(results, &config);

    let a_count = diversified
        .iter()
        .filter(|r| extract_source(r) == "a")
        .count();
    assert_eq!(a_count, 2, "Source 'a' should be capped at 2");
    assert!(diversified.iter().any(|r| r.id == "b1"));
}

#[test]
fn test_diversify_preserves_cross_tier_order() {
    let config = DiversityConfig {
        enabled: true,
        max_per_source: 5,
        score_tier_threshold: 0.05,
    };
    let results = vec![
        make_project_result("high1", 0.95, "a"),
        make_project_result("high2", 0.93, "b"),
        make_project_result("low1", 0.70, "a"),
        make_project_result("low2", 0.68, "b"),
    ];
    let diversified = diversify_results(results, &config);

    let high1_pos = diversified.iter().position(|r| r.id == "high1").unwrap();
    let low1_pos = diversified.iter().position(|r| r.id == "low1").unwrap();
    assert!(high1_pos < low1_pos, "High-tier before low-tier");
}

#[test]
fn test_diversify_same_tier_interleaves() {
    let config = DiversityConfig {
        enabled: true,
        max_per_source: 5,
        score_tier_threshold: 0.10,
    };
    let results = vec![
        make_project_result("a1", 0.95, "a"),
        make_project_result("a2", 0.94, "a"),
        make_project_result("b1", 0.93, "b"),
        make_project_result("b2", 0.92, "b"),
    ];
    let diversified = diversify_results(results, &config);

    assert_eq!(diversified[0].id, "a1");
    assert_eq!(diversified[1].id, "b1");
}

#[test]
fn test_diversify_empty() {
    let config = DiversityConfig::default();
    let diversified = diversify_results(Vec::new(), &config);
    assert!(diversified.is_empty());
}

#[test]
fn test_diversify_single_result() {
    let config = DiversityConfig::default();
    let results = vec![make_project_result("r1", 0.9, "a")];
    let diversified = diversify_results(results, &config);
    assert_eq!(diversified.len(), 1);
}

// ─── Diversity score tests ──────────────────────────────────────────

#[test]
fn test_diversity_score_all_unique() {
    let results = vec![
        make_project_result("r1", 0.9, "a"),
        make_project_result("r2", 0.8, "b"),
        make_project_result("r3", 0.7, "c"),
    ];
    let score = diversity_score(&results);
    assert!((score - 1.0).abs() < 1e-6, "All unique = 1.0");
}

#[test]
fn test_diversity_score_all_same() {
    let results = vec![
        make_project_result("r1", 0.9, "a"),
        make_project_result("r2", 0.8, "a"),
        make_project_result("r3", 0.7, "a"),
    ];
    let score = diversity_score(&results);
    assert!(
        (score - 1.0 / 3.0).abs() < 1e-6,
        "1 unique / 3 total = 0.333"
    );
}

#[test]
fn test_diversity_score_empty() {
    assert_eq!(diversity_score(&[]), 0.0);
}

#[test]
fn test_diversity_score_mixed() {
    let results = vec![
        make_project_result("r1", 0.9, "a"),
        make_project_result("r2", 0.8, "b"),
        make_library_result("r3", 0.7, "lib-1"),
        make_project_result("r4", 0.6, "a"),
    ];
    let score = diversity_score(&results);
    assert!((score - 0.75).abs() < 1e-6);
}

#[test]
fn test_diversity_library_name_takes_precedence() {
    let mut payload = HashMap::new();
    payload.insert("library_name".to_string(), serde_json::json!("my-lib"));
    payload.insert("tenant_id".to_string(), serde_json::json!("proj-a"));

    let r = SearchResult {
        id: "r1".to_string(),
        score: 0.9,
        payload,
        dense_vector: None,
        sparse_vector: None,
    };
    assert_eq!(extract_source(&r), "my-lib");
}

// ─── Penalty-based re-ranking tests ─────────────────────────────────

#[test]
fn test_penalty_disabled_returns_unchanged() {
    let config = DiversityPenaltyConfig {
        enabled: false,
        ..Default::default()
    };
    let results = vec![
        make_file_result("r1", 0.9, "proj-a", "/src/main.rs"),
        make_file_result("r2", 0.8, "proj-a", "/src/main.rs"),
    ];
    let penalized = apply_diversity_penalty(results.clone(), &config);
    assert_eq!(penalized.len(), 2);
    assert!((penalized[0].score - 0.9).abs() < 1e-6);
    assert!((penalized[1].score - 0.8).abs() < 1e-6);
}

#[test]
fn test_penalty_empty_results() {
    let config = DiversityPenaltyConfig::default();
    let penalized = apply_diversity_penalty(Vec::new(), &config);
    assert!(penalized.is_empty());
}

#[test]
fn test_penalty_single_result() {
    let config = DiversityPenaltyConfig::default();
    let results = vec![make_file_result("r1", 0.9, "proj-a", "/src/main.rs")];
    let penalized = apply_diversity_penalty(results, &config);
    assert_eq!(penalized.len(), 1);
    assert!((penalized[0].score - 0.9).abs() < 1e-6);
}

#[test]
fn test_penalty_same_file_penalized() {
    let config = DiversityPenaltyConfig::default();
    let results = vec![
        make_file_result("r1", 0.90, "proj-a", "/src/main.rs"),
        make_file_result("r2", 0.85, "proj-a", "/src/main.rs"),
        make_file_result("r3", 0.80, "proj-a", "/src/main.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    // r1: no penalty -> 0.90
    // r2: same file, run=1 -> 0.85 * 0.85 = 0.7225
    // r3: same file, run=2 -> 0.80 * 0.85^2 = 0.578
    assert_eq!(penalized.len(), 3);
    assert_eq!(penalized[0].id, "r1");
    assert!((penalized[0].score - 0.90).abs() < 1e-4);
    assert_eq!(penalized[1].id, "r2");
    assert!((penalized[1].score - 0.7225).abs() < 1e-4);
    assert_eq!(penalized[2].id, "r3");
    assert!((penalized[2].score - 0.578).abs() < 1e-3);
}

#[test]
fn test_penalty_same_project_different_file() {
    let config = DiversityPenaltyConfig::default();
    let results = vec![
        make_file_result("r1", 0.90, "proj-a", "/src/main.rs"),
        make_file_result("r2", 0.85, "proj-a", "/src/lib.rs"),
        make_file_result("r3", 0.80, "proj-a", "/src/utils.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    // r1: no penalty -> 0.90
    // r2: same project, different file, run=1 -> 0.85 * 0.92 = 0.782
    assert_eq!(penalized.len(), 3);
    assert_eq!(penalized[0].id, "r1");
    assert!((penalized[0].score - 0.90).abs() < 1e-4);
    assert_eq!(penalized[1].id, "r2");
    assert!((penalized[1].score - 0.782).abs() < 1e-3);
}

#[test]
fn test_penalty_different_projects_no_penalty() {
    let config = DiversityPenaltyConfig::default();
    let results = vec![
        make_file_result("r1", 0.90, "proj-a", "/a/main.rs"),
        make_file_result("r2", 0.85, "proj-b", "/b/main.rs"),
        make_file_result("r3", 0.80, "proj-c", "/c/main.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    assert_eq!(penalized.len(), 3);
    assert!((penalized[0].score - 0.90).abs() < 1e-6);
    assert!((penalized[1].score - 0.85).abs() < 1e-6);
    assert!((penalized[2].score - 0.80).abs() < 1e-6);
}

#[test]
fn test_penalty_reorders_results() {
    let config = DiversityPenaltyConfig::default();
    let results = vec![
        make_file_result("same1", 0.90, "proj-a", "/src/main.rs"),
        make_file_result("same2", 0.89, "proj-a", "/src/main.rs"),
        make_file_result("same3", 0.88, "proj-a", "/src/main.rs"),
        make_file_result("diff1", 0.75, "proj-b", "/other/file.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    // same1: 0.90 (no penalty)
    // same2: 0.89 * 0.85 = 0.7565
    // same3: 0.88 * 0.85^2 = 0.6358
    // diff1: 0.75 (no penalty -- different project)
    //
    // Re-sorted: same1(0.90) > same2(0.7565) > diff1(0.75) > same3(0.6358)
    assert_eq!(penalized[0].id, "same1");
    assert_eq!(penalized[1].id, "same2");
    assert_eq!(penalized[2].id, "diff1");
    assert_eq!(penalized[3].id, "same3");
}

#[test]
fn test_penalty_improves_source_variety() {
    let config = DiversityPenaltyConfig::default();

    let results = vec![
        make_file_result("a1", 0.95, "proj-a", "/a/file.rs"),
        make_file_result("a2", 0.93, "proj-a", "/a/file.rs"),
        make_file_result("a3", 0.91, "proj-a", "/a/file.rs"),
        make_file_result("a4", 0.89, "proj-a", "/a/file.rs"),
        make_file_result("b1", 0.80, "proj-b", "/b/file.rs"),
        make_file_result("b2", 0.78, "proj-b", "/b/file.rs"),
    ];

    let before_diversity = file_diversity_score(&results);
    let penalized = apply_diversity_penalty(results, &config);
    let after_diversity = file_diversity_score(&penalized);

    // Verify that proj-b results moved up in the ranking.
    let top_3: Vec<&str> = penalized.iter().take(3).map(|r| r.id.as_str()).collect();
    assert!(
        top_3.contains(&"b1"),
        "proj-b should appear in top 3 after penalty reranking; got {:?}",
        top_3
    );

    assert!(before_diversity > 0.0);
    assert!(after_diversity > 0.0);
}

#[test]
fn test_penalty_file_penalty_stronger_than_project() {
    let config = DiversityPenaltyConfig::default();

    let same_file = vec![
        make_file_result("f1", 1.0, "proj-a", "/same.rs"),
        make_file_result("f2", 1.0, "proj-a", "/same.rs"),
    ];
    let same_project = vec![
        make_file_result("p1", 1.0, "proj-a", "/diff1.rs"),
        make_file_result("p2", 1.0, "proj-a", "/diff2.rs"),
    ];

    let penalized_file = apply_diversity_penalty(same_file, &config);
    let penalized_proj = apply_diversity_penalty(same_project, &config);

    assert!(
        penalized_file[1].score < penalized_proj[1].score,
        "Same-file penalty ({}) should be stronger than same-project penalty ({})",
        penalized_file[1].score,
        penalized_proj[1].score,
    );
}

#[test]
fn test_penalty_custom_config() {
    let config = DiversityPenaltyConfig {
        enabled: true,
        same_file_penalty: 0.50,
        same_project_penalty: 0.75,
    };
    let results = vec![
        make_file_result("r1", 1.0, "proj-a", "/src/main.rs"),
        make_file_result("r2", 1.0, "proj-a", "/src/main.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    assert!((penalized[0].score - 1.0).abs() < 1e-6);
    assert!((penalized[1].score - 0.50).abs() < 1e-6);
}

#[test]
fn test_penalty_compounding_three_same_file() {
    let config = DiversityPenaltyConfig {
        enabled: true,
        same_file_penalty: 0.80,
        same_project_penalty: 0.90,
    };
    let results = vec![
        make_file_result("r1", 1.0, "proj-a", "/src/main.rs"),
        make_file_result("r2", 1.0, "proj-a", "/src/main.rs"),
        make_file_result("r3", 1.0, "proj-a", "/src/main.rs"),
        make_file_result("r4", 1.0, "proj-a", "/src/main.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    // r1: 1.0, r2: 0.80, r3: 0.64, r4: 0.512
    assert!((penalized[0].score - 1.0).abs() < 1e-4);
    assert!((penalized[1].score - 0.80).abs() < 1e-4);
    assert!((penalized[2].score - 0.64).abs() < 1e-4);
    assert!((penalized[3].score - 0.512).abs() < 1e-4);
}

#[test]
fn test_penalty_run_resets_on_different_source() {
    let config = DiversityPenaltyConfig::default();
    let results = vec![
        make_file_result("a1", 0.90, "proj-a", "/a.rs"),
        make_file_result("a2", 0.85, "proj-a", "/a.rs"),
        make_file_result("b1", 0.80, "proj-b", "/b.rs"),
        make_file_result("a3", 0.75, "proj-a", "/a.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    // a3 is after a different-project break, so no penalty
    assert!((penalized.iter().find(|r| r.id == "a3").unwrap().score - 0.75).abs() < 1e-6);
}

#[test]
fn test_penalty_mixed_file_and_project_runs() {
    let config = DiversityPenaltyConfig::default();
    let results = vec![
        make_file_result("r1", 0.95, "proj-a", "/src/main.rs"),
        make_file_result("r2", 0.90, "proj-a", "/src/main.rs"),
        make_file_result("r3", 0.85, "proj-a", "/src/lib.rs"),
        make_file_result("r4", 0.80, "proj-a", "/src/utils.rs"),
    ];
    let penalized = apply_diversity_penalty(results, &config);

    // r1: 0.95 (no penalty)
    // r2: 0.90 * 0.85 = 0.765 (same file, file_run=1)
    // r3: 0.85 * 0.92 = 0.782 (same project, file_run resets, project_run=1)
    // r4: 0.80 * 0.92^2 = 0.677 (same project, project_run=2)
    let r1 = penalized.iter().find(|r| r.id == "r1").unwrap();
    let r2 = penalized.iter().find(|r| r.id == "r2").unwrap();
    let r3 = penalized.iter().find(|r| r.id == "r3").unwrap();
    let r4 = penalized.iter().find(|r| r.id == "r4").unwrap();

    assert!((r1.score - 0.95).abs() < 1e-4);
    assert!((r2.score - 0.765).abs() < 1e-3);
    assert!((r3.score - 0.782).abs() < 1e-3);
    assert!((r4.score - 0.677).abs() < 1e-2);
}

#[test]
fn test_penalty_default_config_values() {
    let config = DiversityPenaltyConfig::default();
    assert!(config.enabled);
    assert!((config.same_file_penalty - 0.85).abs() < 1e-6);
    assert!((config.same_project_penalty - 0.92).abs() < 1e-6);
}

// ─── File diversity score tests ────────────────────────────────────

#[test]
fn test_file_diversity_score_all_unique() {
    let results = vec![
        make_file_result("r1", 0.9, "proj-a", "/a.rs"),
        make_file_result("r2", 0.8, "proj-a", "/b.rs"),
        make_file_result("r3", 0.7, "proj-b", "/c.rs"),
    ];
    let score = file_diversity_score(&results);
    assert!((score - 1.0).abs() < 1e-6);
}

#[test]
fn test_file_diversity_score_all_same() {
    let results = vec![
        make_file_result("r1", 0.9, "proj-a", "/same.rs"),
        make_file_result("r2", 0.8, "proj-a", "/same.rs"),
        make_file_result("r3", 0.7, "proj-a", "/same.rs"),
    ];
    let score = file_diversity_score(&results);
    assert!((score - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_file_diversity_score_empty() {
    assert_eq!(file_diversity_score(&[]), 0.0);
}
