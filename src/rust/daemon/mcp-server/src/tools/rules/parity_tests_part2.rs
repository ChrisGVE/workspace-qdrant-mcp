//! Parity tests part 2: FIX 2 (dup-check) and DEFAULT_DUPLICATION_THRESHOLD.
//!
//! Included from `parity_tests.rs` via
//! `#[cfg(test)] #[path = "parity_tests_part2.rs"] mod part2;`.

use serde_json::json;

use super::types::RulesInput;
use super::*;
use super::{
    args, get_json, get_text, qdrant_pt, qdrant_retrieved, top_keys, PaDaemon, PaQdrant, PaReader,
};

// ─────────────────────────────────────────────────────────────────────────────
// FIX 2 — add runs findSimilarRules dup-check before persisting
// (mirrors rules.ts:68-93 + rules.ts:119-158)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fix2_empty_embedding_skips_dup_check_and_proceeds() {
    // embed returns [] → dup-check skipped → add proceeds
    let mut d = PaDaemon::ok_no_embed();
    let q = PaQdrant::no_duplicates();
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(
        json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }),
    ))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    // embed was still called (we always call it)
    assert_eq!(d.embed_count(), 1);
    // but no Qdrant search was issued (embed returned empty → short-circuit)
    assert_eq!(q.search_count(), 0);
    // ingest did run
    assert_eq!(d.ingest_count(), 1);
}

#[tokio::test]
async fn fix2_with_embedding_no_duplicates_proceeds_to_add() {
    // embed returns vector, search returns empty → no dups → add proceeds
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let q = PaQdrant::no_duplicates();
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(
        json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }),
    ))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["message"], json!("Rule added successfully"));
    assert_eq!(q.search_count(), 1);
    assert_eq!(d.ingest_count(), 1);
}

#[tokio::test]
async fn fix2_duplicate_found_returns_refusal_not_add() {
    // embed returns vector, search returns a high-score hit → refusal
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let dup = qdrant_pt("dup-id", "Existing rule", 0.85);
    let q = PaQdrant::search_ok(vec![dup]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "new", "content": "Similar content", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    // Must be refusal (rules.ts:84-90)
    assert_eq!(j["success"], json!(false));
    assert_eq!(j["action"], json!("add"));
    // ingest must NOT have been called
    assert_eq!(d.ingest_count(), 0);
    // similar_rules must be present with the duplicate
    let similar = j["similar_rules"].as_array().unwrap();
    assert_eq!(similar.len(), 1);
    assert_eq!(similar[0]["id"], json!("dup-id"));
}

#[tokio::test]
async fn fix2_refusal_message_mentions_force_path() {
    // #104: the refusal must tell the caller how to complete the add.
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let dups = vec![qdrant_pt("d1", "r1", 0.9), qdrant_pt("d2", "r2", 0.8)];
    let q = PaQdrant::search_ok(dups);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "l", "content": "c", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(
        j["message"].as_str().unwrap(),
        "Found 2 similar rule(s). Review them; if the new rule is distinct, retry with force: true."
    );
}

/// #104: `force: true` skips the similarity gate so a reviewed-and-distinct
/// rule can be added even when similar rules exist.
#[tokio::test]
async fn force_true_bypasses_dup_gate_and_adds() {
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let dup = qdrant_pt("dup-id", "Existing rule", 0.85);
    let q = PaQdrant::search_ok(vec![dup]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "new", "content": "Similar content",
        "scope": "global", "force": true
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["message"], json!("Rule added successfully"));
    // Gate skipped entirely: no dup search issued, ingest ran.
    assert_eq!(q.search_count(), 0);
    assert_eq!(d.ingest_count(), 1);
}

/// #104: absent or false `force` keeps the gate (default behavior unchanged).
#[tokio::test]
async fn force_false_keeps_dup_gate() {
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let dup = qdrant_pt("dup-id", "Existing rule", 0.85);
    let q = PaQdrant::search_ok(vec![dup]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "new", "content": "Similar content",
        "scope": "global", "force": false
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(false));
    assert_eq!(d.ingest_count(), 0);
}

#[tokio::test]
async fn fix2_similarity_rounded_to_3_decimals() {
    // rules.ts:153: Math.round(point.score * 1000) / 1000
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    // score 0.85678 → should round to 0.857
    let pt = qdrant_pt("id1", "content", 0.85678);
    let q = PaQdrant::search_ok(vec![pt]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "l", "content": "c", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    let similar = j["similar_rules"].as_array().unwrap();
    let sim = similar[0]["similarity"].as_f64().unwrap();
    assert!((sim - 0.857).abs() < 1e-9, "expected 0.857, got {sim}");
}

#[tokio::test]
async fn fix2_search_error_allows_add_to_proceed() {
    // Any embed/search error → allow add (rules.ts:155-158)
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let q = PaQdrant::search_err("Qdrant down");
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "l", "content": "c", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    assert_eq!(d.ingest_count(), 1);
}

#[tokio::test]
async fn fix2_refusal_field_order() {
    // Refusal field order: success → action → similar_rules → message
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let pt = qdrant_pt("d1", "dup", 0.9);
    let q = PaQdrant::search_ok(vec![pt]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "l", "content": "c", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    assert_eq!(
        top_keys(get_text(&res)),
        vec!["success", "action", "similar_rules", "message"]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DEFAULT_DUPLICATION_THRESHOLD constant
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn default_duplication_threshold_is_0_7() {
    // Mirrors `DEFAULT_DUPLICATION_THRESHOLD = 0.7` in rules.ts:32
    use super::super::list::DEFAULT_DUPLICATION_THRESHOLD;
    assert!(
        (DEFAULT_DUPLICATION_THRESHOLD - 0.7).abs() < f64::EPSILON,
        "expected 0.7, got {DEFAULT_DUPLICATION_THRESHOLD}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact-content idempotency (alkmimm PR #134 salvage, 5e7497759 item 3)
// A byte-identical rule in the same scope is a no-op, not a new row, and the
// check is deterministic + fail-open.
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn exact_content_duplicate_is_idempotent_noop() {
    // An existing rule with byte-identical content → success no-op, no write,
    // and the fuzzy embed/search gate never runs (it short-circuits earlier).
    let mut d = PaDaemon::ok_with_embed(vec![0.1_f32; 384]);
    let existing = qdrant_retrieved("rule-existing", "remember to use uv");
    let q = PaQdrant::scroll_ok(vec![existing]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "use-uv", "content": "remember to use uv", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    assert!(
        j["message"]
            .as_str()
            .unwrap()
            .contains("Identical rule already exists"),
        "got: {}",
        j["message"]
    );
    assert_eq!(d.ingest_count(), 0, "no ingest on an idempotent re-add");
    assert_eq!(
        d.embed_count(),
        0,
        "exact match short-circuits before the fuzzy gate"
    );
    assert_eq!(q.search_count(), 0, "no fuzzy search on an exact match");
}

#[tokio::test]
async fn exact_check_fails_open_on_scroll_error() {
    // A scroll failure must NOT block the add (deterministic + fail-open).
    let mut d = PaDaemon::ok_no_embed(); // empty embed → fuzzy gate skipped
    let q = PaQdrant::scroll_err("qdrant down");
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "l", "content": "c", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["message"], json!("Rule added successfully"));
    assert_eq!(d.ingest_count(), 1, "scroll error must not block the add");
}

#[tokio::test]
async fn near_but_not_identical_content_is_not_a_noop() {
    // Only BYTE-identical content is a no-op; different content proceeds to add
    // (where the fuzzy gate, not this exact check, governs similarity).
    let mut d = PaDaemon::ok_no_embed(); // empty embed → fuzzy gate skipped
    let existing = qdrant_retrieved("rule-1", "remember to use uv");
    let q = PaQdrant::scroll_ok(vec![existing]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({
        "action": "add", "label": "l", "content": "remember to use poetry", "scope": "global"
    })))
    .unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["message"], json!("Rule added successfully"));
    assert_eq!(d.ingest_count(), 1);
}
