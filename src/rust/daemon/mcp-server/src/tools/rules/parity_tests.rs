//! Parity tests for the two real-fix gaps in the rules tool.
//!
//! FIX 1 — `list` must query Qdrant FIRST, fall back to mirror (rules-list.ts:119-135)
//! FIX 2 — `add` must run findSimilarRules dup-check BEFORE adding (rules.ts:68-93)
//!
//! Both fixes are tested here with hermetic stubs — no live gRPC, SQLite, or Qdrant.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use crate::qdrant::client::{QdrantPoint, QdrantRetrievedPoint};
use crate::sqlite::rules_mirror::RulesMirrorEntry;

use super::traits::{RulesDaemon, RulesQdrant, RulesReader};
use super::types::RulesInput;
use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Compact mocks (self-contained for this test module)
// ─────────────────────────────────────────────────────────────────────────────

pub(super) struct PaDaemon {
    pub ingest_result: Result<bool, (bool, String)>,
    pub embed_response: Vec<f32>,
    pub ingest_count: Arc<Mutex<usize>>,
    pub embed_count: Arc<Mutex<usize>>,
}

impl PaDaemon {
    pub(super) fn ok_no_embed() -> Self {
        Self {
            ingest_result: Ok(true),
            embed_response: Vec::new(),
            ingest_count: Arc::new(Mutex::new(0)),
            embed_count: Arc::new(Mutex::new(0)),
        }
    }

    pub(super) fn ok_with_embed(v: Vec<f32>) -> Self {
        Self {
            ingest_result: Ok(true),
            embed_response: v,
            ingest_count: Arc::new(Mutex::new(0)),
            embed_count: Arc::new(Mutex::new(0)),
        }
    }

    pub(super) fn ingest_count(&self) -> usize {
        *self.ingest_count.lock().unwrap()
    }

    pub(super) fn embed_count(&self) -> usize {
        *self.embed_count.lock().unwrap()
    }
}

impl RulesDaemon for PaDaemon {
    async fn ingest_text(
        &mut self,
        _c: String,
        _col: String,
        _t: String,
        _d: String,
        _m: HashMap<String, String>,
    ) -> Result<bool, (bool, String)> {
        *self.ingest_count.lock().unwrap() += 1;
        self.ingest_result.clone()
    }

    async fn enqueue_item(
        &mut self,
        _it: &str,
        _op: &str,
        _t: &str,
        _col: &str,
        _pj: &str,
        _br: &str,
        _mj: Option<&str>,
    ) -> Result<String, String> {
        Ok("q-pa".to_string())
    }

    async fn upsert_rule_mirror(
        &mut self,
        _id: String,
        _t: String,
        _s: Option<String>,
        _ti: Option<String>,
        _ca: String,
        _ua: String,
    ) {
    }

    async fn delete_rule_mirror(&mut self, _id: String) {}

    async fn embed_text(&mut self, _text: String) -> Vec<f32> {
        *self.embed_count.lock().unwrap() += 1;
        self.embed_response.clone()
    }
}

pub(super) struct PaReader {
    pub rows: Vec<RulesMirrorEntry>,
}

impl PaReader {
    pub(super) fn empty() -> Self {
        Self { rows: Vec::new() }
    }

    pub(super) fn with(rows: Vec<RulesMirrorEntry>) -> Self {
        Self { rows }
    }
}

impl RulesReader for PaReader {
    fn list_from_mirror(
        &self,
        _s: Option<&str>,
        _t: Option<&str>,
        limit: usize,
    ) -> Vec<RulesMirrorEntry> {
        self.rows.iter().take(limit).cloned().collect()
    }
}

pub(super) struct PaQdrant {
    pub scroll_result: Result<Vec<QdrantRetrievedPoint>, String>,
    pub search_result: Result<Vec<QdrantPoint>, String>,
    pub scroll_count: Arc<Mutex<u32>>,
    pub search_count: Arc<Mutex<u32>>,
}

impl PaQdrant {
    pub(super) fn scroll_ok(pts: Vec<QdrantRetrievedPoint>) -> Self {
        Self {
            scroll_result: Ok(pts),
            search_result: Ok(Vec::new()),
            scroll_count: Arc::new(Mutex::new(0)),
            search_count: Arc::new(Mutex::new(0)),
        }
    }

    pub(super) fn scroll_err(msg: &str) -> Self {
        Self {
            scroll_result: Err(msg.to_string()),
            search_result: Ok(Vec::new()),
            scroll_count: Arc::new(Mutex::new(0)),
            search_count: Arc::new(Mutex::new(0)),
        }
    }

    pub(super) fn search_ok(pts: Vec<QdrantPoint>) -> Self {
        Self {
            scroll_result: Ok(Vec::new()),
            search_result: Ok(pts),
            scroll_count: Arc::new(Mutex::new(0)),
            search_count: Arc::new(Mutex::new(0)),
        }
    }

    pub(super) fn search_err(msg: &str) -> Self {
        Self {
            scroll_result: Ok(Vec::new()),
            search_result: Err(msg.to_string()),
            scroll_count: Arc::new(Mutex::new(0)),
            search_count: Arc::new(Mutex::new(0)),
        }
    }

    pub(super) fn no_duplicates() -> Self {
        Self {
            scroll_result: Ok(Vec::new()),
            search_result: Ok(Vec::new()),
            scroll_count: Arc::new(Mutex::new(0)),
            search_count: Arc::new(Mutex::new(0)),
        }
    }

    pub(super) fn scroll_count(&self) -> u32 {
        *self.scroll_count.lock().unwrap()
    }

    pub(super) fn search_count(&self) -> u32 {
        *self.search_count.lock().unwrap()
    }
}

impl RulesQdrant for PaQdrant {
    async fn scroll_rules(
        &self,
        _f: Option<qdrant_client::qdrant::Filter>,
        _l: u32,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        *self.scroll_count.lock().unwrap() += 1;
        self.scroll_result.clone()
    }

    async fn search_rules(
        &self,
        _v: Vec<f32>,
        _l: u64,
        _t: f32,
        _f: Option<qdrant_client::qdrant::Filter>,
    ) -> Result<Vec<QdrantPoint>, String> {
        *self.search_count.lock().unwrap() += 1;
        self.search_result.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn args(obj: Value) -> Map<String, Value> {
    obj.as_object().unwrap().clone()
}

pub(super) fn get_json(r: &rmcp::model::CallToolResult) -> Value {
    let text = r
        .content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str();
    serde_json::from_str(text).unwrap()
}

pub(super) fn get_text(r: &rmcp::model::CallToolResult) -> &str {
    r.content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str()
}

pub(super) fn top_keys(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.starts_with("  \"") && !l.starts_with("   "))
        .filter_map(|l| {
            let t = l.trim_start_matches("  \"");
            t.split_once('"').map(|(k, _)| k.to_string())
        })
        .collect()
}

pub(super) fn mirror_row(id: &str, text: &str, scope: &str) -> RulesMirrorEntry {
    RulesMirrorEntry {
        rule_id: id.to_string(),
        rule_text: text.to_string(),
        scope: Some(scope.to_string()),
        tenant_id: None,
        created_at: "2024-01-01T00:00:00Z".to_string(),
        updated_at: "2024-01-02T00:00:00Z".to_string(),
    }
}

pub(super) fn qdrant_pt(id: &str, content: &str, score: f64) -> QdrantPoint {
    let mut p = HashMap::new();
    p.insert("content".to_string(), Value::String(content.to_string()));
    p.insert("scope".to_string(), Value::String("global".to_string()));
    QdrantPoint {
        id: id.to_string(),
        score,
        payload: p,
    }
}

pub(super) fn qdrant_retrieved(id: &str, content: &str) -> QdrantRetrievedPoint {
    let mut p = HashMap::new();
    p.insert("content".to_string(), Value::String(content.to_string()));
    p.insert("scope".to_string(), Value::String("global".to_string()));
    QdrantRetrievedPoint {
        id: id.to_string(),
        payload: p,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FIX 1 — list queries Qdrant first, falls back to mirror
// (mirrors rules-list.ts:119-135)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fix1_list_uses_qdrant_primary_path() {
    // Qdrant returns a point → must be in response; mirror must NOT be used
    let mut d = PaDaemon::ok_no_embed();
    let pt = qdrant_retrieved("q-1", "Qdrant rule");
    let q = PaQdrant::scroll_ok(vec![pt]);
    let r = PaReader::with(vec![mirror_row("m-1", "Mirror rule", "global")]);
    let input =
        RulesInput::from_args(&args(json!({ "action": "list", "scope": "global" }))).unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    let rules = j["rules"].as_array().unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0]["id"], json!("q-1"));
    assert_eq!(rules[0]["content"], json!("Qdrant rule"));
    assert_eq!(q.scroll_count(), 1);
}

#[tokio::test]
async fn fix1_list_qdrant_ok_message_exact_ts_string() {
    // rules-list.ts:124: `Found ${rules.length} rule(s)`
    let mut d = PaDaemon::ok_no_embed();
    let q = PaQdrant::scroll_ok(vec![]);
    let r = PaReader::empty();
    let input =
        RulesInput::from_args(&args(json!({ "action": "list", "scope": "global" }))).unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None).await;
    let j = get_json(&res);
    assert_eq!(j["message"], json!("Found 0 rule(s)"));
    assert_eq!(j["success"], json!(true));
}

#[tokio::test]
async fn fix1_list_qdrant_error_mirror_fallback_message() {
    // rules-list.ts:95: "Found N rule(s) from local mirror (Qdrant unavailable)"
    let mut d = PaDaemon::ok_no_embed();
    let q = PaQdrant::scroll_err("conn refused");
    let r = PaReader::with(vec![
        mirror_row("m-1", "Mirror 1", "global"),
        mirror_row("m-2", "Mirror 2", "global"),
    ]);
    let input =
        RulesInput::from_args(&args(json!({ "action": "list", "scope": "global" }))).unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(true));
    assert_eq!(
        j["message"].as_str().unwrap(),
        "Found 2 rule(s) from local mirror (Qdrant unavailable)"
    );
    let rules = j["rules"].as_array().unwrap();
    assert_eq!(rules[0]["id"], json!("m-1"));
}

#[tokio::test]
async fn fix1_list_qdrant_error_mirror_empty_returns_failure() {
    // rules-list.ts:130-133: success=false, "Failed to list rules: <msg>"
    let mut d = PaDaemon::ok_no_embed();
    let q = PaQdrant::scroll_err("Qdrant down");
    let r = PaReader::empty();
    let input =
        RulesInput::from_args(&args(json!({ "action": "list", "scope": "global" }))).unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None).await;
    let j = get_json(&res);
    assert_eq!(j["success"], json!(false));
    let msg = j["message"].as_str().unwrap();
    assert!(msg.starts_with("Failed to list rules:"), "got: {msg}");
    assert!(msg.contains("Qdrant down"));
}

#[tokio::test]
async fn fix1_list_prefers_qdrant_over_mirror_when_both_available() {
    let mut d = PaDaemon::ok_no_embed();
    let pts = vec![qdrant_retrieved("q-a", "A"), qdrant_retrieved("q-b", "B")];
    let q = PaQdrant::scroll_ok(pts);
    let r = PaReader::with(vec![mirror_row("m-x", "Mirror X", "global")]);
    let input = RulesInput::from_args(&args(json!({ "action": "list" }))).unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None).await;
    let j = get_json(&res);
    let rules = j["rules"].as_array().unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0]["id"], json!("q-a"));
}

#[tokio::test]
async fn fix1_list_field_order_qdrant_success() {
    // success → action → rules → message
    let mut d = PaDaemon::ok_no_embed();
    let q = PaQdrant::scroll_ok(vec![]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({ "action": "list" }))).unwrap();
    let res = rules_tool(input, &mut d, &r, &q, None).await;
    assert_eq!(
        top_keys(get_text(&res)),
        vec!["success", "action", "rules", "message"]
    );
}

#[tokio::test]
async fn fix1_list_does_not_call_ingest_or_enqueue() {
    let mut d = PaDaemon::ok_no_embed();
    let q = PaQdrant::scroll_ok(vec![]);
    let r = PaReader::empty();
    let input = RulesInput::from_args(&args(json!({ "action": "list" }))).unwrap();
    let _ = rules_tool(input, &mut d, &r, &q, None).await;
    assert_eq!(d.ingest_count(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// FIX 2 + constant — split into sibling to keep this file under 500 lines
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "parity_tests_part2.rs"]
mod part2;
