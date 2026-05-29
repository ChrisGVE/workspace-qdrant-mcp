//! Tests for the `rules` MCP tool.
//!
//! All tests are hermetic: they inject `MockRulesDaemon` and
//! `MockRulesReader` to avoid live gRPC or SQLite dependencies.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use crate::sqlite::rules_mirror::RulesMirrorEntry;

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// MockRulesDaemon
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EnqueueCall {
    item_type: String,
    op: String,
    tenant_id: String,
    collection: String,
    payload_json: String,
    branch: String,
    metadata_json: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct IngestCall {
    content: String,
    collection_basename: String,
    tenant_id: String,
    document_id: String,
}

#[derive(Debug, Clone)]
struct MirrorCall {
    method: String,
    rule_id: String,
}

#[derive(Debug)]
struct MockRulesDaemon {
    /// Controls ingest_text result: Ok(success), Err((is_conn, msg))
    pub ingest_result: Result<bool, (bool, String)>,
    /// Controls enqueue_item result
    pub enqueue_result: Result<String, String>,
    pub enqueue_calls: Arc<Mutex<Vec<EnqueueCall>>>,
    pub ingest_calls: Arc<Mutex<Vec<IngestCall>>>,
    pub mirror_calls: Arc<Mutex<Vec<MirrorCall>>>,
}

impl MockRulesDaemon {
    fn ingest_ok() -> Self {
        Self {
            ingest_result: Ok(true),
            enqueue_result: Ok("q-1".to_string()),
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn ingest_fails_connectivity() -> Self {
        Self {
            ingest_result: Err((true, "UNAVAILABLE".to_string())),
            enqueue_result: Ok("q-fallback".to_string()),
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn ingest_fails_hard() -> Self {
        Self {
            ingest_result: Err((false, "internal error".to_string())),
            enqueue_result: Ok("q-1".to_string()),
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn enqueue_fails(msg: &str) -> Self {
        Self {
            ingest_result: Err((true, "UNAVAILABLE".to_string())),
            enqueue_result: Err(msg.to_string()),
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn ingest_call_count(&self) -> usize {
        self.ingest_calls.lock().unwrap().len()
    }

    fn enqueue_call_count(&self) -> usize {
        self.enqueue_calls.lock().unwrap().len()
    }

    fn mirror_call_count(&self, method: &str) -> usize {
        self.mirror_calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.method == method)
            .count()
    }

    fn last_enqueue(&self) -> Option<EnqueueCall> {
        self.enqueue_calls.lock().unwrap().last().cloned()
    }

    fn last_ingest(&self) -> Option<IngestCall> {
        self.ingest_calls.lock().unwrap().last().cloned()
    }
}

impl RulesDaemon for MockRulesDaemon {
    async fn ingest_text(
        &mut self,
        content: String,
        collection_basename: String,
        tenant_id: String,
        document_id: String,
        _metadata: HashMap<String, String>,
    ) -> Result<bool, (bool, String)> {
        self.ingest_calls.lock().unwrap().push(IngestCall {
            content,
            collection_basename,
            tenant_id,
            document_id,
        });
        self.ingest_result.clone()
    }

    async fn enqueue_item(
        &mut self,
        item_type: &str,
        op: &str,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata_json: Option<&str>,
    ) -> Result<String, String> {
        self.enqueue_calls.lock().unwrap().push(EnqueueCall {
            item_type: item_type.to_string(),
            op: op.to_string(),
            tenant_id: tenant_id.to_string(),
            collection: collection.to_string(),
            payload_json: payload_json.to_string(),
            branch: branch.to_string(),
            metadata_json: metadata_json.unwrap_or("").to_string(),
        });
        self.enqueue_result.clone()
    }

    async fn upsert_rule_mirror(
        &mut self,
        rule_id: String,
        _rule_text: String,
        _scope: Option<String>,
        _tenant_id: Option<String>,
        _created_at: String,
        _updated_at: String,
    ) {
        self.mirror_calls.lock().unwrap().push(MirrorCall {
            method: "upsert".to_string(),
            rule_id,
        });
    }

    async fn delete_rule_mirror(&mut self, rule_id: String) {
        self.mirror_calls.lock().unwrap().push(MirrorCall {
            method: "delete".to_string(),
            rule_id,
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MockRulesReader
// ─────────────────────────────────────────────────────────────────────────────

struct MockRulesReader {
    pub rows: Vec<RulesMirrorEntry>,
}

impl MockRulesReader {
    fn empty() -> Self {
        Self { rows: Vec::new() }
    }

    fn with(rows: Vec<RulesMirrorEntry>) -> Self {
        Self { rows }
    }
}

impl RulesReader for MockRulesReader {
    fn list_from_mirror(
        &self,
        _scope: Option<&str>,
        _tenant_id: Option<&str>,
        limit: usize,
    ) -> Vec<RulesMirrorEntry> {
        self.rows.iter().take(limit).cloned().collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
    result
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str()
}

fn extract_json(result: &rmcp::model::CallToolResult) -> Value {
    serde_json::from_str(extract_text(result)).unwrap()
}

/// Extract top-level field key order from serde_json pretty-printed JSON.
/// Scans lines starting with exactly two spaces + quoted key.
fn top_level_keys(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.starts_with("  \"") && !l.starts_with("   "))
        .filter_map(|l| {
            let trimmed = l.trim_start_matches("  \"");
            trimmed.split_once('"').map(|(k, _)| k.to_string())
        })
        .collect()
}

fn make_args(obj: Value) -> Map<String, Value> {
    obj.as_object().unwrap().clone()
}

fn make_rule_row(id: &str, text: &str, scope: &str, tenant: Option<&str>) -> RulesMirrorEntry {
    RulesMirrorEntry {
        rule_id: id.to_string(),
        rule_text: text.to_string(),
        scope: Some(scope.to_string()),
        tenant_id: tenant.map(str::to_string),
        created_at: "2024-01-01T00:00:00Z".to_string(),
        updated_at: "2024-01-02T00:00:00Z".to_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RulesInput parsing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn input_invalid_action_returns_err() {
    let args = make_args(json!({ "action": "delete" }));
    let result = RulesInput::from_args(&args);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid rules action"));
}

#[test]
fn input_missing_action_returns_err() {
    let args = make_args(json!({}));
    let result = RulesInput::from_args(&args);
    assert!(result.is_err());
}

#[test]
fn input_add_parses_correctly() {
    let args = make_args(json!({
        "action": "add",
        "label": "use-tracing",
        "content": "Always use tracing macros.",
        "scope": "global",
        "tags": ["rust"],
        "priority": 5
    }));
    let input = RulesInput::from_args(&args).unwrap();
    assert_eq!(input.action, "add");
    assert_eq!(input.label.as_deref(), Some("use-tracing"));
    assert_eq!(input.scope, "global");
    assert_eq!(input.tags.as_deref(), Some(["rust".to_string()].as_slice()));
    assert_eq!(input.priority, Some(5));
}

#[test]
fn input_scope_defaults_to_project() {
    let args = make_args(json!({ "action": "list" }));
    let input = RulesInput::from_args(&args).unwrap();
    assert_eq!(input.scope, "project");
}

#[test]
fn input_limit_defaults_to_50() {
    let args = make_args(json!({ "action": "list" }));
    let input = RulesInput::from_args(&args).unwrap();
    assert_eq!(input.limit, 50);
}

// ─────────────────────────────────────────────────────────────────────────────
// add_rule — happy path (ingest direct)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_rule_success_via_ingest() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({
        "action": "add",
        "label": "use-tracing",
        "content": "Always use tracing macros.",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    assert!(result.is_error.is_none());
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["action"], json!("add"));
    assert_eq!(j["label"], json!("use-tracing"));
    assert_eq!(j["message"], json!("Rule added successfully"));
    // No fallback_mode or queue_id on direct ingest success
    assert!(j.get("fallback_mode").is_none());
    assert!(j.get("queue_id").is_none());
    // ingest called once, enqueue not called
    assert_eq!(daemon.ingest_call_count(), 1);
    assert_eq!(daemon.enqueue_call_count(), 0);
}

#[tokio::test]
async fn add_rule_ingest_uses_random_uuid_as_doc_id() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    let ingest = daemon.last_ingest().unwrap();
    // ADD: document_id is a UUID (36-char string), NOT the label
    // rules-mutation-helpers.ts:194
    assert_ne!(ingest.document_id, "l");
    assert_eq!(ingest.document_id.len(), 36); // UUID format
}

#[tokio::test]
async fn add_rule_calls_upsert_mirror_on_direct_success() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    assert_eq!(daemon.mirror_call_count("upsert"), 1);
    let calls = daemon.mirror_calls.lock().unwrap();
    let mc = calls.iter().find(|c| c.method == "upsert").unwrap();
    assert_eq!(mc.rule_id, "l");
}

// ─────────────────────────────────────────────────────────────────────────────
// add_rule — fallback path (connectivity error)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_rule_connectivity_error_falls_back_to_queue() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
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
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    assert_eq!(result.is_error, Some(true));
    assert!(extract_text(&result).contains("Failed to add rule"));
}

#[tokio::test]
async fn add_rule_missing_content_returns_in_band_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "add", "label": "l" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert_eq!(j["action"], json!("add"));
    assert!(result.is_error.is_none());
}

#[tokio::test]
async fn add_rule_missing_label_returns_in_band_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "add", "content": "c" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"].as_str().unwrap().contains("Label is required"));
}

#[tokio::test]
async fn add_rule_project_scope_no_project_id_returns_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "add", "label": "l", "content": "c" }));
    // scope defaults to "project", no project_id, no session project
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
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
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "project" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, Some("session-proj-001")).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    let ingest = daemon.last_ingest().unwrap();
    assert_eq!(ingest.tenant_id, "session-proj-001");
}

#[tokio::test]
async fn add_rule_global_scope_uses_tenant_global() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    let ingest = daemon.last_ingest().unwrap();
    assert_eq!(ingest.tenant_id, "global");
}

#[tokio::test]
async fn add_rule_result_field_order() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let text = extract_text(&result);
    // success → action → label → message (fallback_mode/queue_id absent)
    assert_eq!(
        top_level_keys(text),
        vec!["success", "action", "label", "message"]
    );
}

#[tokio::test]
async fn add_rule_enqueue_metadata_is_mcp_rules_tool() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    let enq = daemon.last_enqueue().unwrap();
    assert!(enq.metadata_json.contains("mcp_rules_tool"));
}

// ─────────────────────────────────────────────────────────────────────────────
// update_rule
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn update_rule_uses_label_as_doc_id() {
    // UPDATE: document_id = label (stable) — rules-mutation-helpers.ts:279
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({
        "action": "update",
        "label": "my-rule",
        "content": "Updated content",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    let ingest = daemon.last_ingest().unwrap();
    assert_eq!(ingest.document_id, "my-rule");
}

#[tokio::test]
async fn update_rule_success_via_direct_ingest() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({
        "action": "update",
        "label": "my-rule",
        "content": "Updated.",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
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
    let args = make_args(json!({ "action": "update", "content": "c" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert_eq!(j["action"], json!("update"));
}

#[tokio::test]
async fn update_rule_connectivity_fallback_enqueues() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({
        "action": "update",
        "label": "my-rule",
        "content": "content",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
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
    let args = make_args(json!({
        "action": "remove",
        "label": "old-rule",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["action"], json!("remove"));
    assert_eq!(j["label"], json!("old-rule"));
    assert_eq!(j["fallback_mode"], json!("unified_queue"));
    // enqueue called with op="delete"
    let enq = daemon.last_enqueue().unwrap();
    assert_eq!(enq.op, "delete");
    // mirror delete called
    assert_eq!(daemon.mirror_call_count("delete"), 1);
    let calls = daemon.mirror_calls.lock().unwrap();
    let mc = calls.iter().find(|c| c.method == "delete").unwrap();
    assert_eq!(mc.rule_id, "old-rule");
}

#[tokio::test]
async fn remove_rule_missing_label_returns_error() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "remove" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert_eq!(j["action"], json!("remove"));
    assert!(j["message"].as_str().unwrap().contains("Label is required"));
}

#[tokio::test]
async fn remove_rule_enqueue_failure_returns_error_text() {
    let mut daemon = MockRulesDaemon::enqueue_fails("queue offline");
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "remove", "label": "l", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    assert_eq!(result.is_error, Some(true));
    assert!(extract_text(&result).contains("queue offline"));
}

#[tokio::test]
async fn remove_rule_no_mirror_delete_on_enqueue_failure() {
    let mut daemon = MockRulesDaemon::enqueue_fails("fail");
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "remove", "label": "l", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    assert_eq!(daemon.mirror_call_count("delete"), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// list_rules
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn list_rules_empty_mirror_returns_empty_list() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "list", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["action"], json!("list"));
    assert_eq!(j["rules"], json!([]));
    assert!(j["message"].as_str().unwrap().contains("Found 0"));
}

#[tokio::test]
async fn list_rules_returns_mirror_rows() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let rows = vec![
        make_rule_row("rule-1", "Always use tracing.", "global", None),
        make_rule_row("rule-2", "Prefer composition.", "project", Some("proj-abc")),
    ];
    let reader = MockRulesReader::with(rows);
    let args = make_args(json!({ "action": "list" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    let rules = j["rules"].as_array().unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0]["id"], json!("rule-1"));
    assert_eq!(rules[0]["content"], json!("Always use tracing."));
    assert_eq!(rules[0]["scope"], json!("global"));
    assert_eq!(rules[1]["projectId"], json!("proj-abc"));
}

#[tokio::test]
async fn list_rules_result_field_order() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "list" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, None).await;
    let text = extract_text(&result);
    // success → action → rules → message
    assert_eq!(
        top_level_keys(text),
        vec!["success", "action", "rules", "message"]
    );
}

#[tokio::test]
async fn list_rules_does_not_call_daemon() {
    // list reads mirror only — no daemon calls
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let args = make_args(json!({ "action": "list" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    assert_eq!(daemon.ingest_call_count(), 0);
    assert_eq!(daemon.enqueue_call_count(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// resolve_tenant (unit)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn resolve_tenant_global_scope_returns_none() {
    let result = resolve_tenant("global", None, None);
    assert_eq!(result, Ok(None));
}

#[test]
fn resolve_tenant_project_scope_explicit_id() {
    let result = resolve_tenant("project", Some("explicit-proj"), None);
    assert_eq!(result, Ok(Some("explicit-proj".to_string())));
}

#[test]
fn resolve_tenant_project_scope_session_fallback() {
    let result = resolve_tenant("project", None, Some("session-proj"));
    assert_eq!(result, Ok(Some("session-proj".to_string())));
}

#[test]
fn resolve_tenant_project_scope_no_id_returns_error() {
    let result = resolve_tenant("project", None, None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not a registered project"));
}

// ─────────────────────────────────────────────────────────────────────────────
// is_connectivity_error (unit)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn connectivity_error_detects_unavailable() {
    assert!(is_connectivity_error("UNAVAILABLE: connection lost"));
}

#[test]
fn connectivity_error_detects_deadline() {
    assert!(is_connectivity_error("DEADLINE_EXCEEDED"));
}

#[test]
fn connectivity_error_non_connectivity_false() {
    assert!(!is_connectivity_error("internal server error"));
    assert!(!is_connectivity_error("not found"));
}

// ─────────────────────────────────────────────────────────────────────────────
// RuleItem field order
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rule_item_field_order_in_list() {
    let item = RuleItem {
        id: "r1".to_string(),
        label: None,
        content: "c".to_string(),
        scope: "global".to_string(),
        project_id: None,
        title: None,
        tags: None,
        priority: None,
        created_at: Some("2024-01-01T00:00:00Z".to_string()),
        updated_at: Some("2024-01-02T00:00:00Z".to_string()),
    };
    let text = serde_json::to_string_pretty(&item).unwrap();
    // id → content → scope → createdAt → updatedAt (optional absent fields omitted)
    assert_eq!(
        top_level_keys(&text),
        vec!["id", "content", "scope", "createdAt", "updatedAt"]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// queue op uses correct item_type and collection
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_queue_fallback_uses_text_item_type_and_rules_collection() {
    let mut daemon = MockRulesDaemon::ingest_fails_connectivity();
    let reader = MockRulesReader::empty();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, None).await;
    let enq = daemon.last_enqueue().unwrap();
    assert_eq!(enq.item_type, "text");
    assert_eq!(enq.collection, "rules");
    assert_eq!(enq.branch, "main");
}
