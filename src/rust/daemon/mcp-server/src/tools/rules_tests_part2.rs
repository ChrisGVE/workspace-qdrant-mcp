//! Rules tool tests part 2: add fallback, update, remove, resolve_tenant,
//! is_connectivity_error, RuleItem field order, and queue op type tests.
//!
//! Included from `rules_tests.rs` via
//! `#[path = "rules_tests_part2.rs"] mod part2;`.

use serde_json::json;

use super::super::helpers;
use super::super::rules_tool;
use super::super::types::RuleItem;
use super::super::types::RulesInput;
use super::{
    extract_json, extract_text, make_args, top_level_keys, MockRulesDaemon, MockRulesQdrant,
    MockRulesReader,
};

// ─────────────────────────────────────────────────────────────────────────────
// add_rule — fallback path (connectivity error)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_rule_connectivity_error_falls_back_to_queue() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["fallback_mode"], json!("unified_queue"));
    assert_eq!(j["queue_id"], json!("q-fallback"));
    assert_eq!(daemon.enqueue_call_count(), 1);
    assert_eq!(daemon.mirror_call_count("upsert"), 1);
}

#[tokio::test]
async fn add_rule_hard_ingest_error_returns_error_text() {
    let mut daemon = MockRulesDaemon::ingest_fails_hard();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    assert_eq!(result.is_error, Some(true));
    assert!(extract_text(&result).contains("Failed to add rule"));
}

#[tokio::test]
async fn add_rule_missing_content_returns_in_band_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({ "action": "add", "label": "l" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert_eq!(j["action"], json!("add"));
    assert!(result.is_error.is_none());
}

#[tokio::test]
async fn add_rule_missing_label_returns_in_band_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({ "action": "add", "content": "c" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"].as_str().unwrap().contains("Label is required"));
}

#[tokio::test]
async fn add_rule_project_scope_no_project_id_returns_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({ "action": "add", "label": "l", "content": "c" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"]
        .as_str()
        .unwrap()
        .contains("not a registered project"));
}

#[tokio::test]
async fn add_rule_project_scope_uses_session_project_id() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "project" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(
        input,
        &mut daemon,
        &reader,
        &qdrant,
        Some("session-proj-001"),
        None,
    )
    .await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    let ingest = daemon.last_ingest().unwrap();
    assert_eq!(ingest.tenant_id, "session-proj-001");
}

#[tokio::test]
async fn add_rule_global_scope_uses_tenant_global() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let ingest = daemon.last_ingest().unwrap();
    assert_eq!(ingest.tenant_id, "global");
}

#[tokio::test]
async fn add_rule_result_field_order() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let text = extract_text(&result);
    assert_eq!(
        top_level_keys(text),
        vec!["success", "action", "label", "message"]
    );
}

#[tokio::test]
async fn add_rule_enqueue_metadata_is_mcp_rules_tool() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let enq = daemon.last_enqueue().unwrap();
    assert!(enq.metadata_json.contains("mcp_rules_tool"));
}

// ─────────────────────────────────────────────────────────────────────────────
// update_rule
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn update_rule_uses_label_as_doc_id() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({
        "action": "update",
        "label": "my-rule",
        "content": "Updated content",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let ingest = daemon.last_ingest().unwrap();
    assert_eq!(ingest.document_id, "my-rule");
}

#[tokio::test]
async fn update_rule_success_via_direct_ingest() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({
        "action": "update",
        "label": "my-rule",
        "content": "Updated.",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["action"], json!("update"));
    assert_eq!(j["message"], json!("Rule updated successfully"));
    assert!(j.get("fallback_mode").is_none());
    assert_eq!(daemon.mirror_call_count("upsert"), 1);
}

#[tokio::test]
async fn update_rule_missing_label_returns_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({ "action": "update", "content": "c" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert_eq!(j["action"], json!("update"));
}

#[tokio::test]
async fn update_rule_connectivity_fallback_enqueues() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({
        "action": "update",
        "label": "my-rule",
        "content": "content",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["fallback_mode"], json!("unified_queue"));
    assert_eq!(daemon.enqueue_call_count(), 1);
    let enq = daemon.last_enqueue().unwrap();
    assert_eq!(enq.op, "update");
}

// ─────────────────────────────────────────────────────────────────────────────
// remove_rule
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn remove_rule_success_queues_delete_and_deletes_mirror() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({
        "action": "remove",
        "label": "old-rule",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["action"], json!("remove"));
    assert_eq!(j["label"], json!("old-rule"));
    assert_eq!(j["fallback_mode"], json!("unified_queue"));
    let enq = daemon.last_enqueue().unwrap();
    assert_eq!(enq.op, "delete");
    assert_eq!(daemon.mirror_call_count("delete"), 1);
    let calls = daemon.mirror_calls.lock().unwrap();
    let mc = calls.iter().find(|c| c.method == "delete").unwrap();
    assert_eq!(mc.rule_id, "old-rule");
}

#[tokio::test]
async fn remove_rule_missing_label_returns_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({ "action": "remove" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert_eq!(j["action"], json!("remove"));
    assert!(j["message"].as_str().unwrap().contains("Label is required"));
}

#[tokio::test]
async fn remove_rule_enqueue_failure_returns_error_text() {
    let mut daemon = MockRulesDaemon::enqueue_fails("queue offline");
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({ "action": "remove", "label": "l", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    assert_eq!(result.is_error, Some(true));
    assert!(extract_text(&result).contains("queue offline"));
}

#[tokio::test]
async fn remove_rule_no_mirror_delete_on_enqueue_failure() {
    let mut daemon = MockRulesDaemon::enqueue_fails("fail");
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({ "action": "remove", "label": "l", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    assert_eq!(daemon.mirror_call_count("delete"), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// resolve_tenant (unit)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn resolve_tenant_global_scope_returns_none() {
    let result = helpers::resolve_tenant("global", None, None);
    assert_eq!(result, Ok(None));
}

#[test]
fn resolve_tenant_project_scope_explicit_id() {
    let result = helpers::resolve_tenant("project", Some("explicit-proj"), None);
    assert_eq!(result, Ok(Some("explicit-proj".to_string())));
}

#[test]
fn resolve_tenant_project_scope_session_fallback() {
    let result = helpers::resolve_tenant("project", None, Some("session-proj"));
    assert_eq!(result, Ok(Some("session-proj".to_string())));
}

#[test]
fn resolve_tenant_project_scope_no_id_returns_error() {
    let result = helpers::resolve_tenant("project", None, None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not a registered project"));
}

// ─────────────────────────────────────────────────────────────────────────────
// is_connectivity_error (unit)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn connectivity_error_detects_unavailable() {
    assert!(helpers::is_connectivity_error(
        "UNAVAILABLE: connection lost"
    ));
}

#[test]
fn connectivity_error_detects_deadline() {
    assert!(helpers::is_connectivity_error("DEADLINE_EXCEEDED"));
}

#[test]
fn connectivity_error_non_connectivity_false() {
    assert!(!helpers::is_connectivity_error("internal server error"));
    assert!(!helpers::is_connectivity_error("not found"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fix 1 — RuleItem field order: id→content→scope→label?→projectId?→…→similarity?
// TS `pointToRule` (rules-list.ts:35-54) emits content+scope unconditionally
// (lines 37-38) then label conditionally (line 41) — label AFTER scope.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rule_item_field_order_in_list() {
    let item = RuleItem {
        id: "r1".to_string(),
        content: "c".to_string(),
        scope: "global".to_string(),
        label: None,
        project_id: None,
        title: None,
        tags: None,
        priority: None,
        created_at: Some("2024-01-01T00:00:00Z".to_string()),
        updated_at: Some("2024-01-02T00:00:00Z".to_string()),
        similarity: None,
        owner: None,
    };
    let text = serde_json::to_string_pretty(&item).unwrap();
    assert_eq!(
        top_level_keys(&text),
        vec!["id", "content", "scope", "createdAt", "updatedAt"]
    );
}

#[test]
fn rule_item_field_order_with_all_optional_fields() {
    // content and scope come before label — mirrors pointToRule key order
    let item = RuleItem {
        id: "r2".to_string(),
        content: "Always use tracing".to_string(),
        scope: "project".to_string(),
        label: Some("use-tracing".to_string()),
        project_id: Some("proj-001".to_string()),
        title: Some("Tracing rule".to_string()),
        tags: Some(vec!["rust".to_string()]),
        priority: Some(5),
        created_at: Some("2024-01-01T00:00:00Z".to_string()),
        updated_at: Some("2024-01-02T00:00:00Z".to_string()),
        similarity: None,
        owner: None,
    };
    let text = serde_json::to_string_pretty(&item).unwrap();
    let keys = top_level_keys(&text);
    // content and scope appear before label
    let content_pos = keys.iter().position(|k| k == "content").unwrap();
    let scope_pos = keys.iter().position(|k| k == "scope").unwrap();
    let label_pos = keys.iter().position(|k| k == "label").unwrap();
    assert!(
        content_pos < label_pos,
        "content ({content_pos}) must come before label ({label_pos})"
    );
    assert!(
        scope_pos < label_pos,
        "scope ({scope_pos}) must come before label ({label_pos})"
    );
    // Full order assertion
    assert_eq!(
        keys,
        vec![
            "id",
            "content",
            "scope",
            "label",
            "projectId",
            "title",
            "tags",
            "priority",
            "createdAt",
            "updatedAt"
        ]
    );
}

#[test]
fn rule_item_similarity_field_is_last() {
    // similarity field appears after all other optional fields
    let item = RuleItem {
        id: "s1".to_string(),
        content: "c".to_string(),
        scope: "global".to_string(),
        label: Some("lbl".to_string()),
        project_id: None,
        title: None,
        tags: None,
        priority: None,
        created_at: None,
        updated_at: None,
        similarity: Some(0.857),
        owner: None,
    };
    let text = serde_json::to_string_pretty(&item).unwrap();
    let keys = top_level_keys(&text);
    assert_eq!(keys.last().unwrap(), "similarity");
    // content before label
    let content_pos = keys.iter().position(|k| k == "content").unwrap();
    let label_pos = keys.iter().position(|k| k == "label").unwrap();
    assert!(content_pos < label_pos);
}

// ─────────────────────────────────────────────────────────────────────────────
// queue op uses correct item_type and collection
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_queue_fallback_uses_text_item_type_and_rules_collection() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    let enq = daemon.last_enqueue().unwrap();
    assert_eq!(enq.item_type, "text");
    assert_eq!(enq.collection, "rules");
    assert_eq!(enq.branch, "main");
}

// ─────────────────────────────────────────────────────────────────────────────
// config_dup_threshold tests — split to sibling for size compliance
// ─────────────────────────────────────────────────────────────────────────────

#[path = "rules_tests_part3.rs"]
mod part3;
