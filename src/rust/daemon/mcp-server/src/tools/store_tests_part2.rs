//! Store tool tests part 2: scratchpad, library, generate_document_id,
//! validate_url, and metadata source marker tests.
//!
//! Included from `store_tests.rs` via
//! `#[path = "store_tests_part2.rs"] mod part2;`.

use serde_json::json;

use super::super::{generate_document_id, store_tool, validate_url, StoreInput};
use super::{extract_json, extract_text, make_args, top_level_keys, MockStoreDaemon};

// ─────────────────────────────────────────────────────────────────────────────
// store type=scratchpad
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn scratchpad_missing_content_returns_error_json() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "scratchpad" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"]
        .as_str()
        .unwrap()
        .contains("content is required"));
    assert!(result.is_error.is_none());
}

#[tokio::test]
async fn scratchpad_success_enqueues_text_item() {
    let mut daemon = MockStoreDaemon::ok("scratchpad-q1");
    let args = make_args(json!({
        "type": "scratchpad",
        "content": "My idea for a new feature",
        "title": "Feature Idea"
    }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, Some("proj-123"), true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["queue_id"], json!("scratchpad-q1"));
    assert_eq!(j["collection"], json!("scratchpad"));
    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    assert_eq!(enqueue_args[0], "text");
    assert_eq!(enqueue_args[3], "scratchpad");
    assert_eq!(enqueue_args[2], "proj-123");
}

#[tokio::test]
async fn scratchpad_calls_mirror_upsert_on_success() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({
        "type": "scratchpad",
        "content": "test note",
        "tags": ["rust"]
    }));
    let input = StoreInput::from_args(&args, None);
    let _ = store_tool(input, &mut daemon, Some("proj-456"), true).await;
    // Mirror upsert should have been called exactly once
    assert_eq!(daemon.call_count("upsert_scratchpad_mirror"), 1);
    let mirror_args = daemon.last_call_args("upsert_scratchpad_mirror").unwrap();
    // args: scratchpad_id, content, title, tags, tenant_id
    assert_eq!(mirror_args[1], "test note");
    assert_eq!(mirror_args[4], "proj-456");
}

#[tokio::test]
async fn scratchpad_no_mirror_on_enqueue_failure() {
    let mut daemon = MockStoreDaemon::fail_enqueue("timeout");
    let args = make_args(json!({ "type": "scratchpad", "content": "note" }));
    let input = StoreInput::from_args(&args, None);
    let _ = store_tool(input, &mut daemon, None, true).await;
    assert_eq!(daemon.call_count("upsert_scratchpad_mirror"), 0);
}

#[tokio::test]
async fn scratchpad_global_tenant_when_no_session_project() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "scratchpad", "content": "note" }));
    let input = StoreInput::from_args(&args, None);
    let _ = store_tool(input, &mut daemon, None, true).await;
    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    assert_eq!(enqueue_args[2], "global");
}

#[tokio::test]
async fn scratchpad_result_field_order_no_queue_id_on_error() {
    let mut daemon = MockStoreDaemon::fail_enqueue("fail");
    let args = make_args(json!({ "type": "scratchpad", "content": "note" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    // queue_id must be absent on failure
    assert!(j.get("queue_id").is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// store type=library (default)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn library_missing_content_returns_error_json() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "libraryName": "my-lib" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"]
        .as_str()
        .unwrap()
        .contains("Content is required"));
}

#[tokio::test]
async fn library_missing_library_name_returns_error_json() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "content": "some content" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"]
        .as_str()
        .unwrap()
        .contains("libraryName is required"));
}

#[tokio::test]
async fn library_success_enqueues_tenant_item() {
    let mut daemon = MockStoreDaemon::ok("lib-q1");
    let args = make_args(json!({
        "content": "Reference content here",
        "libraryName": "rust-docs"
    }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    assert_eq!(j["collection"], json!("libraries"));
    assert_eq!(j["fallback_mode"], json!("unified_queue"));
    assert_eq!(j["queue_id"], json!("lib-q1"));
    // documentId must be present and 32 chars
    let doc_id = j["documentId"].as_str().unwrap();
    assert_eq!(doc_id.len(), 32);

    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    assert_eq!(enqueue_args[0], "tenant");
    assert_eq!(enqueue_args[1], "add");
    assert_eq!(enqueue_args[2], "rust-docs");
    assert_eq!(enqueue_args[3], "libraries");
}

#[tokio::test]
async fn library_result_field_order() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "content": "c", "libraryName": "lib" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let text = extract_text(&result);
    // success → documentId → collection → message → fallback_mode → queue_id
    assert_eq!(
        top_level_keys(text),
        vec![
            "success",
            "documentId",
            "collection",
            "message",
            "fallback_mode",
            "queue_id"
        ]
    );
}

#[tokio::test]
async fn library_error_result_no_document_id() {
    let mut daemon = MockStoreDaemon::fail_enqueue("err");
    let args = make_args(json!({ "content": "c", "libraryName": "lib" }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert!(j.get("documentId").is_none());
    assert!(j.get("queue_id").is_none());
}

#[tokio::test]
async fn library_metadata_source_marker_is_mcp_store_tool() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "content": "c", "libraryName": "lib" }));
    let input = StoreInput::from_args(&args, None);
    let _ = store_tool(input, &mut daemon, None, true).await;
    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    // metadata_json is last arg
    assert!(enqueue_args[6].contains("mcp_store_tool"));
}

#[tokio::test]
async fn library_for_project_uses_session_project_id() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "content": "c", "forProject": true }));
    let input = StoreInput::from_args(&args, Some("proj-xyz"));
    let result = store_tool(input, &mut daemon, Some("proj-xyz"), true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(true));
    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    assert_eq!(enqueue_args[2], "proj-xyz");
}

#[tokio::test]
async fn library_for_project_without_session_id_returns_error() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "content": "c", "forProject": true }));
    let input = StoreInput::from_args(&args, None);
    let result = store_tool(input, &mut daemon, None, true).await;
    let j = extract_json(&result);
    assert_eq!(j["success"], json!(false));
    assert!(j["message"].as_str().unwrap().contains("No active project"));
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_document_id
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn document_id_is_32_hex_chars() {
    let id = generate_document_id("hello world", "my-tenant");
    assert_eq!(id.len(), 32);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn document_id_is_deterministic() {
    let id1 = generate_document_id("content", "tenant");
    let id2 = generate_document_id("content", "tenant");
    assert_eq!(id1, id2);
}

#[test]
fn document_id_differs_by_content() {
    let id1 = generate_document_id("content A", "tenant");
    let id2 = generate_document_id("content B", "tenant");
    assert_ne!(id1, id2);
}

#[test]
fn document_id_differs_by_tenant() {
    let id1 = generate_document_id("content", "tenant-A");
    let id2 = generate_document_id("content", "tenant-B");
    assert_ne!(id1, id2);
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_url
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn validate_url_empty_fails() {
    assert!(validate_url("").is_err());
    assert!(validate_url("   ").is_err());
}

#[test]
fn validate_url_non_http_scheme_fails() {
    assert!(validate_url("ftp://example.com").is_err());
}

#[test]
fn validate_url_http_ok() {
    assert!(validate_url("http://example.com").is_ok());
}

#[test]
fn validate_url_https_ok() {
    assert!(validate_url("https://example.com/path").is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// scratchpad metadata source marker
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn scratchpad_metadata_source_is_mcp_store_scratchpad() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "scratchpad", "content": "note" }));
    let input = StoreInput::from_args(&args, None);
    let _ = store_tool(input, &mut daemon, None, true).await;
    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    assert!(enqueue_args[6].contains("mcp_store_scratchpad"));
}

// ─────────────────────────────────────────────────────────────────────────────
// url metadata source marker
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn url_metadata_source_is_mcp_store_url() {
    let mut daemon = MockStoreDaemon::ok("q1");
    let args = make_args(json!({ "type": "url", "url": "https://example.com" }));
    let input = StoreInput::from_args(&args, None);
    let _ = store_tool(input, &mut daemon, None, true).await;
    let enqueue_args = daemon.last_call_args("enqueue_item").unwrap();
    assert!(enqueue_args[6].contains("mcp_store_url"));
}
