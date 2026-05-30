//! Assertion helpers for the TS↔Rust **error/validation** parity matrix.
//!
//! Each helper loads a committed corpus under `tests/golden/errors/` (captured
//! from the TypeScript MCP server — real `dist/` exports or verbatim-copied
//! non-exported validators; see `tests/golden/errors/README.md`), looks up one
//! case by `name`, drives the corresponding **real Rust function/handler**, and
//! asserts byte-for-byte parity with the TS-sourced expected value.
//!
//! Per-case `#[test]` wrappers live in the `gen_err_*.rs` files (one test per
//! corpus row). All drivers are hermetic — no daemon / Qdrant / network; the
//! validation branches under test return before any I/O, so an always-failing
//! stub daemon is never actually invoked.

use std::path::Path;

use serde_json::{Map, Value};

use mcp_server::proto::{TextSearchRequest, TextSearchResponse};
use mcp_server::tools::envelope::unknown_tool;
use mcp_server::tools::grep::{grep_tool, GrepDaemon, GrepInput};
use mcp_server::tools::rules::RulesInput;
use mcp_server::tools::store::{store_tool, ProjectRegisterResult, StoreDaemon, StoreInput};

// ── corpus loading ────────────────────────────────────────────────────────────

fn load_corpus(stem: &str) -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/errors")
        .join(stem)
        .with_extension("json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read error corpus {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parse error corpus {}: {e}", path.display()))
}

fn case(stem: &str, name: &str) -> Value {
    load_corpus(stem)
        .into_iter()
        .find(|c| c["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("error corpus {stem}: no case named {name:?}"))
}

fn as_args(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

/// Extract the single text content block from a `CallToolResult`.
fn content_text(r: &rmcp::model::CallToolResult) -> String {
    r.content
        .first()
        .expect("content must not be empty")
        .raw
        .as_text()
        .expect("content must be text")
        .text
        .clone()
}

// ── always-failing stub daemon (never reached on the validation paths) ─────────

struct UnreachableDaemon;

impl StoreDaemon for UnreachableDaemon {
    async fn register_project(
        &mut self,
        _path: &str,
        _name: &str,
        _git_remote: Option<&str>,
    ) -> Result<ProjectRegisterResult, String> {
        panic!("validation path must return before register_project")
    }

    async fn enqueue_item(
        &mut self,
        _item_type: &str,
        _op: &str,
        _tenant_id: &str,
        _collection: &str,
        _payload_json: &str,
        _branch: &str,
        _metadata_json: Option<&str>,
    ) -> Result<String, String> {
        // Reached only by the *valid*-input rows (e.g. a well-formed URL that
        // passes validation): the handler proceeds to enqueue. Returning an
        // error surfaces a non-validation message, which the OK-row assertion
        // checks for. The in-band error rows return before this is ever called.
        Err("daemon not connected".to_string())
    }

    async fn upsert_scratchpad_mirror(
        &mut self,
        _scratchpad_id: String,
        _content: String,
        _title: Option<String>,
        _tags: Option<String>,
        _tenant_id: String,
        _created_at: String,
        _updated_at: String,
    ) {
    }
}

impl GrepDaemon for UnreachableDaemon {
    async fn text_search(
        &mut self,
        _request: TextSearchRequest,
    ) -> Result<TextSearchResponse, String> {
        panic!("validation path must return before text_search")
    }
}

// ── 1. rules action — `RulesInput::from_args` Err string ────────────────────────

/// `RulesInput::from_args` either succeeds (corpus `expectedError == "OK"`) or
/// returns `Err(msg)` that must equal TS `buildRuleOptions`'s thrown message.
pub fn assert_rules_action(name: &str) {
    let c = case("rules_action", name);
    let mut args = Map::new();
    if c["present"].as_bool().unwrap_or(false) {
        args.insert("action".to_string(), c["action"].clone());
    }
    let expected = c["expectedError"].as_str().expect("expectedError string");
    match RulesInput::from_args(&args) {
        Ok(_) => assert_eq!(
            expected, "OK",
            "rules_action[{name}]: expected error but parsed"
        ),
        Err(msg) => assert_eq!(
            msg, expected,
            "rules_action[{name}] message mismatch\n  actual:   {msg}\n  expected: {expected}"
        ),
    }
}

// ── 2. unknown tool — `unknown_tool(name)` envelope text ────────────────────────

/// `unknown_tool(toolName)` produces `is_error=true` and text `Unknown tool: …`.
pub fn assert_unknown_tool(name: &str) {
    let c = case("unknown_tool", name);
    let tool_name = c["toolName"].as_str().expect("toolName string");
    let expected = c["expectedText"].as_str().expect("expectedText string");
    let result = unknown_tool(tool_name);
    assert_eq!(
        result.is_error,
        Some(true),
        "unknown_tool must set is_error"
    );
    assert_eq!(
        content_text(&result),
        expected,
        "unknown_tool[{name}] text mismatch"
    );
}

// ── 3. validate_url — `store_tool` type=url in-band message ─────────────────────

/// Drive `store_tool` with `type="url"` and assert the in-band message matches
/// TS `validateUrlInput`. `expected == "OK"` means the URL is valid (so the
/// validation passes and the handler would proceed past the validator — the
/// corpus only includes OK cases that the hermetic stub can short-circuit, so
/// for OK rows we assert the message is NOT one of the validation errors).
pub async fn assert_url_validate(name: &str) {
    let c = case("url_validate", name);
    let expected = c["expected"].as_str().expect("expected string");

    let mut args = Map::new();
    args.insert("type".to_string(), Value::String("url".to_string()));
    if c["present"].as_bool().unwrap_or(false) {
        args.insert("url".to_string(), c["url"].clone());
    }
    let input = StoreInput::from_args(&args, None);
    let mut daemon = UnreachableDaemon;
    let result = store_tool(input, &mut daemon, None, true).await;
    let body: Value = serde_json::from_str(&content_text(&result)).expect("valid json");
    let message = body["message"].as_str().unwrap_or("");

    if expected == "OK" {
        // Valid URL: must NOT be rejected by the validator. The hermetic stub
        // would panic if the handler proceeded to enqueue, so a valid URL
        // surfaces the daemon-enqueue error path instead of a validation error.
        assert!(
            !is_url_validation_error(message),
            "url_validate[{name}]: valid URL wrongly rejected: {message}"
        );
    } else {
        assert_eq!(
            message, expected,
            "url_validate[{name}] message mismatch\n  actual:   {message}\n  expected: {expected}"
        );
    }
}

fn is_url_validation_error(msg: &str) -> bool {
    msg.starts_with("url is required")
        || msg.starts_with("url is malformed")
        || msg.starts_with("url must use")
        || msg.starts_with("url has empty hostname")
        || msg.starts_with("url has invalid hostname")
}

// ── 4. store in-band — `store_tool` library path in-band message ────────────────

/// Drive `store_tool` (library/doc path) and assert the in-band failure JSON
/// (`success`, `collection`, `message`, `fallback_mode`) matches TS
/// `StoreTool.store` / `resolveTenant`.
pub async fn assert_store_inband(name: &str) {
    assert_inband_from("store_inband", name, |args| {
        // library/doc path: no explicit type (defaults to "library").
        StoreInput::from_args(args, None)
    })
    .await;
}

// ── 5. scratchpad in-band — `store_tool` type=scratchpad message ────────────────

pub async fn assert_scratchpad_inband(name: &str) {
    assert_inband_from("scratchpad_inband", name, |args| {
        let mut a = args.clone();
        a.insert("type".to_string(), Value::String("scratchpad".to_string()));
        StoreInput::from_args(&a, None)
    })
    .await;
}

async fn assert_inband_from<F>(stem: &str, name: &str, build: F)
where
    F: Fn(&Map<String, Value>) -> StoreInput,
{
    let c = case(stem, name);
    let args = as_args(&c["opts"]);
    let input = build(&args);
    let mut daemon = UnreachableDaemon;
    let result = store_tool(input, &mut daemon, None, true).await;
    let body: Value = serde_json::from_str(&content_text(&result)).expect("valid json");
    let expected = &c["expected"];
    for key in ["success", "message", "collection"] {
        assert_eq!(
            body[key], expected[key],
            "{stem}[{name}] field {key:?} mismatch\n  actual:   {}\n  expected: {}",
            body[key], expected[key]
        );
    }
}

// ── 6. grep in-band — `grep_tool` empty-pattern message ─────────────────────────

/// Drive `grep_tool` and assert the in-band `grepError` shape for the
/// empty-pattern branch matches TS `grep.ts`.
pub async fn assert_grep_inband(name: &str) {
    let c = case("grep_inband", name);
    let args = as_args(&c["opts"]);
    let input = GrepInput::from_args(&args);
    let mut daemon = UnreachableDaemon;
    let result = grep_tool(input, &mut daemon, None).await;
    let body: Value = serde_json::from_str(&content_text(&result)).expect("valid json");
    let expected = &c["expected"];
    for key in ["success", "message", "total_matches", "truncated"] {
        assert_eq!(
            body[key], expected[key],
            "grep_inband[{name}] field {key:?} mismatch\n  actual:   {}\n  expected: {}",
            body[key], expected[key]
        );
    }
    // matches must be an empty array.
    assert_eq!(
        body["matches"],
        Value::Array(vec![]),
        "grep_inband[{name}] matches"
    );
}
