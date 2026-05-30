//! Tests for the `rules` MCP tool — core mutations and infrastructure.
//!
//! All tests are hermetic: they inject `MockRulesDaemon`, `MockRulesReader`,
//! and `MockRulesQdrant` to avoid live gRPC, SQLite, or Qdrant dependencies.
//!
//! FIX 1 (list Qdrant-first) and FIX 2 (add dup-check) parity tests live in
//! `rules/parity_tests.rs`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use crate::qdrant::client::{QdrantPoint, QdrantRetrievedPoint};
use crate::sqlite::rules_mirror::RulesMirrorEntry;

use super::traits::{RulesDaemon, RulesQdrant, RulesReader};
use super::types::{RuleItem, RulesInput};
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
    /// Embedding to return from embed_text (empty = signal failure)
    pub embed_response: Vec<f32>,
    pub enqueue_calls: Arc<Mutex<Vec<EnqueueCall>>>,
    pub ingest_calls: Arc<Mutex<Vec<IngestCall>>>,
    pub mirror_calls: Arc<Mutex<Vec<MirrorCall>>>,
    pub embed_calls: Arc<Mutex<Vec<String>>>,
}

impl MockRulesDaemon {
    fn ingest_ok() -> Self {
        Self {
            ingest_result: Ok(true),
            enqueue_result: Ok("q-1".to_string()),
            embed_response: Vec::new(), // empty → dup-check skipped
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
            embed_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn ingest_fails_connectivity() -> Self {
        Self {
            ingest_result: Err((true, "UNAVAILABLE".to_string())),
            enqueue_result: Ok("q-fallback".to_string()),
            embed_response: Vec::new(),
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
            embed_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn ingest_fails_hard() -> Self {
        Self {
            ingest_result: Err((false, "internal error".to_string())),
            enqueue_result: Ok("q-1".to_string()),
            embed_response: Vec::new(),
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
            embed_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn enqueue_fails(msg: &str) -> Self {
        Self {
            ingest_result: Err((true, "UNAVAILABLE".to_string())),
            enqueue_result: Err(msg.to_string()),
            embed_response: Vec::new(),
            enqueue_calls: Arc::new(Mutex::new(Vec::new())),
            ingest_calls: Arc::new(Mutex::new(Vec::new())),
            mirror_calls: Arc::new(Mutex::new(Vec::new())),
            embed_calls: Arc::new(Mutex::new(Vec::new())),
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

    async fn embed_text(&mut self, text: String) -> Vec<f32> {
        self.embed_calls.lock().unwrap().push(text);
        self.embed_response.clone()
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
// MockRulesQdrant — no_duplicates variant (for mutation tests)
// ─────────────────────────────────────────────────────────────────────────────

struct MockRulesQdrant {
    pub scroll_result: Result<Vec<QdrantRetrievedPoint>, String>,
    pub search_result: Result<Vec<QdrantPoint>, String>,
}

impl MockRulesQdrant {
    fn no_duplicates() -> Self {
        Self {
            scroll_result: Ok(Vec::new()),
            search_result: Ok(Vec::new()),
        }
    }
}

impl RulesQdrant for MockRulesQdrant {
    async fn scroll_rules(
        &self,
        _filter: Option<qdrant_client::qdrant::Filter>,
        _limit: u32,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        self.scroll_result.clone()
    }

    async fn search_rules(
        &self,
        _vector: Vec<f32>,
        _limit: u64,
        _score_threshold: f32,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<Vec<QdrantPoint>, String> {
        self.search_result.clone()
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
// add_rule — happy path (ingest direct, no duplicates)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_rule_success_via_ingest() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args = make_args(json!({
        "action": "add",
        "label": "use-tracing",
        "content": "Always use tracing macros.",
        "scope": "global"
    }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
    assert!(result.is_error.is_none());
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["action"], json!("add"));
    assert_eq!(j["label"], json!("use-tracing"));
    assert_eq!(j["message"], json!("Rule added successfully"));
    assert!(j.get("fallback_mode").is_none());
    assert!(j.get("queue_id").is_none());
    assert_eq!(daemon.ingest_call_count(), 1);
    assert_eq!(daemon.enqueue_call_count(), 0);
}

#[tokio::test]
async fn add_rule_ingest_uses_random_uuid_as_doc_id() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
    let ingest = daemon.last_ingest().unwrap();
    // ADD: document_id is a UUID (36-char string), NOT the label
    assert_ne!(ingest.document_id, "l");
    assert_eq!(ingest.document_id.len(), 36);
}

#[tokio::test]
async fn add_rule_calls_upsert_mirror_on_direct_success() {
    let mut daemon = MockRulesDaemon::ingest_ok();
    let reader = MockRulesReader::empty();
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let qdrant = MockRulesQdrant::no_duplicates();
    let args =
        make_args(json!({ "action": "add", "label": "l", "content": "c", "scope": "global" }));
    let input = RulesInput::from_args(&args).unwrap();
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
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
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None).await;
    let enq = daemon.last_enqueue().unwrap();
    assert_eq!(enq.item_type, "text");
    assert_eq!(enq.collection, "rules");
    assert_eq!(enq.branch, "main");
}
