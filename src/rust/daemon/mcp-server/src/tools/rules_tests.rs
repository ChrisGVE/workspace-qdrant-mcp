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
use super::types::RulesInput;
use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// MockRulesDaemon
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct EnqueueCall {
    pub item_type: String,
    pub op: String,
    pub tenant_id: String,
    pub collection: String,
    pub payload_json: String,
    pub branch: String,
    pub metadata_json: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct IngestCall {
    pub content: String,
    pub collection_basename: String,
    pub tenant_id: String,
    pub document_id: String,
}

#[derive(Debug, Clone)]
pub(super) struct MirrorCall {
    pub method: String,
    pub rule_id: String,
}

#[derive(Debug)]
pub(super) struct MockRulesDaemon {
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
    pub(super) fn ingest_ok() -> Self {
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

    pub(super) fn ingest_fails_connectivity() -> Self {
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

    pub(super) fn ingest_fails_hard() -> Self {
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

    pub(super) fn enqueue_fails(msg: &str) -> Self {
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

    pub(super) fn ingest_call_count(&self) -> usize {
        self.ingest_calls.lock().unwrap().len()
    }

    pub(super) fn enqueue_call_count(&self) -> usize {
        self.enqueue_calls.lock().unwrap().len()
    }

    pub(super) fn mirror_call_count(&self, method: &str) -> usize {
        self.mirror_calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.method == method)
            .count()
    }

    pub(super) fn last_enqueue(&self) -> Option<EnqueueCall> {
        self.enqueue_calls.lock().unwrap().last().cloned()
    }

    pub(super) fn last_ingest(&self) -> Option<IngestCall> {
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

pub(super) struct MockRulesReader {
    pub rows: Vec<RulesMirrorEntry>,
}

impl MockRulesReader {
    pub(super) fn empty() -> Self {
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

pub(super) struct MockRulesQdrant {
    pub scroll_result: Result<Vec<QdrantRetrievedPoint>, String>,
    pub search_result: Result<Vec<QdrantPoint>, String>,
}

impl MockRulesQdrant {
    pub(super) fn no_duplicates() -> Self {
        Self {
            scroll_result: Ok(Vec::new()),
            search_result: Ok(Vec::new()),
        }
    }

    /// Return a list of near-duplicate points from search.
    /// Each entry is (id, score, content) mirroring a real QdrantPoint.
    pub(super) fn with_duplicates(dups: Vec<(String, f32, String)>) -> Self {
        let points = dups
            .into_iter()
            .map(|(id, score, content)| {
                let mut payload = std::collections::HashMap::new();
                payload.insert("content".to_string(), serde_json::Value::String(content));
                payload.insert(
                    "label".to_string(),
                    serde_json::Value::String("dup-label".to_string()),
                );
                crate::qdrant::client::QdrantPoint {
                    id,
                    score: score.into(),
                    payload,
                }
            })
            .collect();
        Self {
            scroll_result: Ok(Vec::new()),
            search_result: Ok(points),
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

pub(super) fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
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

pub(super) fn extract_json(result: &rmcp::model::CallToolResult) -> Value {
    serde_json::from_str(extract_text(result)).unwrap()
}

/// Extract top-level field key order from serde_json pretty-printed JSON.
pub(super) fn top_level_keys(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.starts_with("  \"") && !l.starts_with("   "))
        .filter_map(|l| {
            let trimmed = l.trim_start_matches("  \"");
            trimmed.split_once('"').map(|(k, _)| k.to_string())
        })
        .collect()
}

pub(super) fn make_args(obj: Value) -> Map<String, Value> {
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
    let result = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
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
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
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
    let _ = rules_tool(input, &mut daemon, &reader, &qdrant, None, None).await;
    assert_eq!(daemon.mirror_call_count("upsert"), 1);
    let calls = daemon.mirror_calls.lock().unwrap();
    let mc = calls.iter().find(|c| c.method == "upsert").unwrap();
    assert_eq!(mc.rule_id, "l");
}

// ─────────────────────────────────────────────────────────────────────────────
// add_rule fallback, update, remove, unit tests — split into sibling
// ─────────────────────────────────────────────────────────────────────────────

#[path = "rules_tests_part2.rs"]
mod part2;
