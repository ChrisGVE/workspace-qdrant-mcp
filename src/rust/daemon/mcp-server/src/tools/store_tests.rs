//! Tests for the `store` MCP tool.
//!
//! All tests are hermetic: they inject a `MockStoreDaemon` to avoid live
//! gRPC or SQLite dependencies.

use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// MockStoreDaemon
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct Call {
    method: String,
    args: Vec<String>,
}

#[derive(Debug)]
pub(super) struct MockStoreDaemon {
    pub register_result: Result<ProjectRegisterResult, String>,
    pub enqueue_result: Result<String, String>,
    pub calls: Arc<Mutex<Vec<Call>>>,
}

impl MockStoreDaemon {
    pub(super) fn ok(queue_id: &str) -> Self {
        Self {
            register_result: Ok(ProjectRegisterResult {
                project_id: "proj-abc123".to_string(),
                newly_registered: false,
                is_active: true,
            }),
            enqueue_result: Ok(queue_id.to_string()),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(super) fn new_registered(queue_id: &str) -> Self {
        Self {
            register_result: Ok(ProjectRegisterResult {
                project_id: "proj-new001".to_string(),
                newly_registered: true,
                is_active: true,
            }),
            enqueue_result: Ok(queue_id.to_string()),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(super) fn fail_enqueue(msg: &str) -> Self {
        Self {
            register_result: Ok(ProjectRegisterResult {
                project_id: "p".to_string(),
                newly_registered: false,
                is_active: true,
            }),
            enqueue_result: Err(msg.to_string()),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(super) fn fail_register(msg: &str) -> Self {
        Self {
            register_result: Err(msg.to_string()),
            enqueue_result: Ok("q1".to_string()),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(super) fn call_count(&self, method: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.method == method)
            .count()
    }

    pub(super) fn last_call_args(&self, method: &str) -> Option<Vec<String>> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|c| c.method == method)
            .map(|c| c.args.clone())
    }
}

impl StoreDaemon for MockStoreDaemon {
    async fn register_project(
        &mut self,
        path: &str,
        name: &str,
        git_remote: Option<&str>,
    ) -> Result<ProjectRegisterResult, String> {
        self.calls.lock().unwrap().push(Call {
            method: "register_project".to_string(),
            args: vec![
                path.to_string(),
                name.to_string(),
                git_remote.unwrap_or("").to_string(),
            ],
        });
        self.register_result.clone()
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
        self.calls.lock().unwrap().push(Call {
            method: "enqueue_item".to_string(),
            args: vec![
                item_type.to_string(),
                op.to_string(),
                tenant_id.to_string(),
                collection.to_string(),
                payload_json.to_string(),
                branch.to_string(),
                metadata_json.unwrap_or("").to_string(),
            ],
        });
        self.enqueue_result.clone()
    }

    async fn upsert_scratchpad_mirror(
        &mut self,
        scratchpad_id: String,
        content: String,
        title: Option<String>,
        tags: Option<String>,
        tenant_id: String,
        _created_at: String,
        _updated_at: String,
    ) {
        self.calls.lock().unwrap().push(Call {
            method: "upsert_scratchpad_mirror".to_string(),
            args: vec![
                scratchpad_id,
                content,
                title.unwrap_or_default(),
                tags.unwrap_or_default(),
                tenant_id,
            ],
        });
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

/// Extract the top-level field key order from a serde_json pretty-printed
/// JSON object by scanning lines that start with exactly two spaces followed
/// by a quoted key. This avoids re-parsing into `serde_json::Value` which
/// uses BTreeMap (alphabetical) when compiled without `preserve_order`.
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
// StoreInput parsing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn input_default_type_is_library() {
    let args = make_args(json!({ "content": "hello", "libraryName": "my-lib" }));
    let input = StoreInput::from_args(&args, None);
    assert_eq!(input.store_type, "library");
}

#[test]
fn input_explicit_type_scratchpad() {
    let args = make_args(json!({ "type": "scratchpad", "content": "note" }));
    let input = StoreInput::from_args(&args, None);
    assert_eq!(input.store_type, "scratchpad");
}

#[test]
fn input_for_project_uses_session_project_id() {
    let args = make_args(json!({ "type": "library", "content": "c", "forProject": true }));
    let input = StoreInput::from_args(&args, Some("session-proj-001"));
    assert!(input.for_project);
    assert_eq!(input.project_id.as_deref(), Some("session-proj-001"));
}

#[test]
fn input_source_type_defaults_to_user_input() {
    let args = make_args(json!({ "content": "c" }));
    let input = StoreInput::from_args(&args, None);
    assert_eq!(input.source_type, "user_input");
}

#[test]
fn input_invalid_source_type_falls_back_to_user_input() {
    let args = make_args(json!({ "sourceType": "invalid_type", "content": "c" }));
    let input = StoreInput::from_args(&args, None);
    assert_eq!(input.source_type, "user_input");
}

#[test]
fn input_tags_parsed_as_vec() {
    let args = make_args(json!({ "type": "scratchpad", "content": "c",
        "tags": ["rust", "async"] }));
    let input = StoreInput::from_args(&args, None);
    assert_eq!(input.tags, vec!["rust", "async"]);
}

// ─────────────────────────────────────────────────────────────────────────────
// store type=project
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn project_missing_path_returns_error() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "project" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    assert_eq!(result.is_error, Some(true));
    assert!(extract_text(&result).contains("path is required"));
}

#[tokio::test]
async fn project_daemon_not_connected_returns_error() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "project", "path": "/home/user/proj" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, false).await;
    assert_eq!(result.is_error, Some(true));
    assert!(extract_text(&result).contains("Daemon is not connected"));
}

#[tokio::test]
async fn project_success_existing_project() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "project", "path": "/home/user/proj" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    assert!(result.is_error.is_none());
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["project_id"], json!("proj-abc123"));
    assert_eq!(j["created"], json!(false));
    assert_eq!(j["is_active"], json!(true));
    assert!(j["message"]
        .as_str()
        .unwrap()
        .contains("already registered"));
}

#[tokio::test]
async fn project_success_newly_registered() {
    let mut daemon = MockStoreDaemon::new_registered("q1");
    let args = make_args(json!({ "type": "project", "path": "/home/user/proj" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["created"], json!(true));
    assert!(j["message"]
        .as_str()
        .unwrap()
        .contains("registered and activated"));
    // should not contain "already"
    assert!(!j["message"].as_str().unwrap().contains("already"));
}

#[tokio::test]
async fn project_register_failure_returns_error() {
    let mut daemon = MockStoreDaemon::fail_register("connection refused");
    let args = make_args(json!({ "type": "project", "path": "/home/user/proj" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    assert_eq!(result.is_error, Some(true));
    assert!(extract_text(&result).contains("connection refused"));
}

#[tokio::test]
async fn project_result_field_order() {
    // Verify all five fields in order: success, project_id, created, is_active, message
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "project", "path": "/p" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let text = extract_text(&result);
    let keys = top_level_keys(text);
    assert_eq!(
        keys,
        vec!["success", "project_id", "created", "is_active", "message"]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// store type=url
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn url_missing_url_field_returns_error_json() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "url" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    // URL errors are in-band (success=false), NOT error_text
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(result.is_error.is_none());
}

#[tokio::test]
async fn url_invalid_scheme_returns_error_json() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "url", "url": "ftp://example.com/file.txt" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"].as_str().unwrap().contains("http://"));
}

#[tokio::test]
async fn url_success_enqueues_with_correct_item_type() {
    let mut daemon = MockStoreDaemon::ok("url-queue-1");
    let args = make_args(json!({
        "type": "url",
        "url": "https://example.com/page",
        "libraryName": "my-lib"
    }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["queue_id"], json!("url-queue-1"));
    assert_eq!(j["collection"], json!("libraries"));
    // Verify enqueue was called with item_type="url"
    let args_vec = daemon.last_call_args("enqueue_item").unwrap();
    assert_eq!(args_vec[0], "url");
    assert_eq!(args_vec[1], "add");
}

#[tokio::test]
async fn url_no_library_uses_scratchpad_collection() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "url", "url": "https://example.com" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, Some("proj-1"), true).await;
    let j = extract_json(&result);
    assert_eq!(j["collection"], json!("scratchpad"));
    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    assert_eq!(enqueue_args[3], "scratchpad");
    // tenant_id should be session project_id
    assert_eq!(enqueue_args[2], "proj-1");
}

#[tokio::test]
async fn url_result_field_order() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "url", "url": "https://example.com" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let text = extract_text(&result);
    // success → message → queue_id → collection
    assert_eq!(
        top_level_keys(text),
        vec!["success", "message", "queue_id", "collection"]
    );
}

#[tokio::test]
async fn url_enqueue_failure_returns_in_band_error() {
    let mut daemon = MockStoreDaemon::fail_enqueue("daemon down");
    let args = make_args(json!({ "type": "url", "url": "https://example.com" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"].as_str().unwrap().contains("daemon down"));
    assert!(result.is_error.is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// scratchpad, library, generate_document_id, validate_url — split into sibling
// ─────────────────────────────────────────────────────────────────────────────

#[path = "store_tests_part2.rs"]
mod part2;
