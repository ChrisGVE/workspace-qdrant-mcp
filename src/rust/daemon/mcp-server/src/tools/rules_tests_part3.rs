//! Rules tool tests part 3: config_dup_threshold operator override tests.
//!
//! Included from `rules_tests_part2.rs` via
//! `#[path = "rules_tests_part3.rs"] mod part3;`.

use serde_json::json;

use super::super::super::rules_tool;
use super::super::super::types::RulesInput;
use super::super::{extract_json, make_args, MockRulesDaemon, MockRulesQdrant, MockRulesReader};

// ─────────────────────────────────────────────────────────────────────────────
// config_dup_threshold: operator override (finding #12)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_rule_config_threshold_blocks_when_score_above_threshold() {
    // find_similar_rules only runs the dup-check when embed_response is non-empty.
    // With config_dup_threshold=0.01 and a Qdrant match at score 0.9 (>= 0.01),
    // the add must be blocked.
    let mut daemon = MockRulesDaemon {
        embed_response: vec![0.1_f32, 0.2_f32, 0.3_f32], // non-empty → dup-check runs
        ..MockRulesDaemon::ingest_ok()
    };
    let reader = MockRulesReader::empty();
    // Qdrant returns a point with score 0.9 — above the 0.01 threshold.
    let qdrant = MockRulesQdrant::with_duplicates(vec![(
        "dup-id".to_string(),
        0.9_f32,
        "similar content".to_string(),
    )]);
    let args = make_args(
        json!({ "action": "add", "label": "l", "content": "similar", "scope": "global" }),
    );
    let input = RulesInput::from_args(&args).unwrap();
    // config_dup_threshold = 0.01 → score 0.9 >= 0.01 → duplicate detected → blocked.
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, Some(0.01)).await;
    let j = extract_json(&result);
    assert!(
        !j["success"].as_bool().unwrap_or(true),
        "score 0.9 >= config threshold 0.01 must block the add; got: {j}"
    );
}

#[tokio::test]
async fn add_rule_config_threshold_allows_when_score_below_threshold() {
    // With config_dup_threshold = 0.99 and a Qdrant match at score 0.5,
    // the application filter (pt.score >= threshold) drops the "duplicate"
    // since 0.5 < 0.99 → empty duplicates list → add proceeds.
    let mut daemon = MockRulesDaemon {
        embed_response: vec![0.1_f32, 0.2_f32], // non-empty → dup-check runs
        ..MockRulesDaemon::ingest_ok()
    };
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::with_duplicates(vec![(
        "low-score-id".to_string(),
        0.5_f32, // below config threshold → filtered out
        "somewhat similar".to_string(),
    )]);
    let args = make_args(
        json!({ "action": "add", "label": "l2", "content": "new content", "scope": "global" }),
    );
    let input = RulesInput::from_args(&args).unwrap();
    // config_dup_threshold = 0.99 → score 0.5 < 0.99 → filtered out → add proceeds.
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, Some(0.99)).await;
    let j = extract_json(&result);
    assert_eq!(
        j["success"],
        json!(true),
        "score 0.5 below config threshold 0.99 must allow the add; got: {j}"
    );
}
