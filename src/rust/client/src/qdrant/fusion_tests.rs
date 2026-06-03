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

#[test]
fn source_key_empty_library_name_falls_through_to_tenant() {
    // Parity: TS `libraryName ? ... : tenantId ...` treats empty-string
    // library_name as falsy, so it falls through to tenant_id.
    let mut payload = HashMap::new();
    payload.insert(
        "library_name".to_string(),
        serde_json::Value::String(String::new()),
    );
    payload.insert(
        "tenant_id".to_string(),
        serde_json::Value::String("proj1".to_string()),
    );
    let r = TaggedResult {
        id: "id".to_string(),
        score: 0.9,
        collection: "libraries".to_string(),
        payload,
        search_type: SearchType::Semantic,
    };
    assert_eq!(r.source_key(), "libraries:proj1");
}

// The diversity / tiering / interleave / point_to_tagged tests live in the
// sibling `fusion_diversity_tests.rs` to keep each file under the line limit.
