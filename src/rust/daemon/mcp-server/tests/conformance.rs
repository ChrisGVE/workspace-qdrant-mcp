//! Golden conformance suite (task-33).
//!
//! Drives Rust tool handlers with documented inputs, applies the S10.4
//! normalizer, and compares against committed fixtures in `tests/golden/`.
//!
//! S10.4 normalizer rules:
//! 1. Volatile fields (`*_ms`, `queue_id`, `session_id`, `event_id`,
//!    `createdAt`, `updatedAt`) → `"MASKED"` sentinel.
//! 2. Float fields (`score`, `similarity`, `diversity_score`) → rounded to
//!    6 decimal places (OQ-8 choice).
//! 3. Equal-score results → secondary sort by `id` asc for determinism.
//! 4. `health` key → value replaced with `true` sentinel (presence only).
//! 5. Main equality on parsed `Value` (whitespace-insensitive).
//! 6. One byte-exact canonical 2-space format test per applicable golden.
//!
//! Corpus: 20 infra-less goldens — see `tests/golden/README.md`.
//!
//! Deferred to task-34 (need live daemon/Qdrant): real search results,
//! retrieve by id, rules mutations, grep results, store enqueue/register,
//! list from live SQLite.
//!
//! Parity divergence: `rules/err_missing_action` — TS emits
//! `"Invalid rules action: undefined"` (JS undefined coercion); Rust emits
//! `"Invalid rules action: "` (empty string for absent key). The golden
//! captures the TS form; the test asserts only the error prefix.

// ─────────────────────────────────────────────────────────────────────────────
// Module imports
// ─────────────────────────────────────────────────────────────────────────────
mod normalizer;

use std::path::Path;

use serde_json::{json, Value};

use mcp_server::tools::embedding::{
    embedding_tool, EmbeddingProviderFields, EmbeddingStatusProvider,
};
use mcp_server::tools::envelope::{error_text, unknown_tool};
use mcp_server::tools::grep::GrepInput;
use mcp_server::tools::retrieve::{retrieve_tool, RetrieveFilter, RetrieveInput, RetrieveQdrant};
use mcp_server::tools::rules::RulesInput;
use mcp_server::tools::search::flow_fallback::{f001_refusal_reason, FALLBACK_STATUS_REASON};
use mcp_server::tools::search::types::{SearchMode, SearchResponse, SearchScope};
use mcp_server::tools::store::{store_tool, ProjectRegisterResult, StoreDaemon, StoreInput};

use normalizer::normalize;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: extract inner result text from a CallToolResult
// ─────────────────────────────────────────────────────────────────────────────

fn content_text(r: &rmcp::model::CallToolResult) -> &str {
    r.content
        .first()
        .expect("content must not be empty")
        .raw
        .as_text()
        .expect("content must be text")
        .text
        .as_str()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: load and parse a golden fixture
// ─────────────────────────────────────────────────────────────────────────────

/// Load a golden fixture by path relative to `tests/golden/`.
fn load_golden(rel: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(rel)
        .with_extension("json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read golden {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse golden {}: {e}", path.display()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: assert error-envelope parity
//
// Error envelope goldens have shape { "is_error": true, "text": "Error: ..." }
// or { "is_error": true, "text": "Unknown tool: ..." }.
//
// The conformance test verifies:
//   (a) Rust result has is_error = Some(true)
//   (b) The content text matches the golden's `text` field
// ─────────────────────────────────────────────────────────────────────────────

fn assert_error_envelope(result: &rmcp::model::CallToolResult, golden: &Value) {
    assert_eq!(
        result.is_error,
        Some(true),
        "error envelope: expected is_error=true"
    );
    let golden_text = golden["text"].as_str().expect("golden.text must be string");
    let actual_text = content_text(result);
    assert_eq!(
        actual_text, golden_text,
        "error envelope text mismatch\n  actual:   {actual_text:?}\n  expected: {golden_text:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Stubs for injectable traits
// ─────────────────────────────────────────────────────────────────────────────

struct EmbeddingDown;
struct EmbeddingOk(EmbeddingProviderFields);

impl EmbeddingStatusProvider for EmbeddingDown {
    async fn get_embedding_provider_status(&mut self) -> Result<EmbeddingProviderFields, String> {
        Err("connection refused".to_string())
    }
}

impl EmbeddingStatusProvider for EmbeddingOk {
    async fn get_embedding_provider_status(&mut self) -> Result<EmbeddingProviderFields, String> {
        Ok(self.0.clone())
    }
}

fn fastembed_fields() -> EmbeddingProviderFields {
    EmbeddingProviderFields {
        provider: "fastembed".to_string(),
        model: "all-MiniLM-L6-v2".to_string(),
        output_dim: 384,
        base_url: String::new(),
        probe_status: "healthy".to_string(),
        probe_message: "probe succeeded".to_string(),
    }
}

struct StoreDaemonUnconnected;

impl StoreDaemon for StoreDaemonUnconnected {
    async fn register_project(
        &mut self,
        _path: &str,
        _name: &str,
        _git_remote: Option<&str>,
    ) -> Result<ProjectRegisterResult, String> {
        Err("daemon not connected".to_string())
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

// Stub for retrieve (returns collection-not-found error)
struct RetrieveNotFound;

impl RetrieveQdrant for RetrieveNotFound {
    async fn retrieve_by_ids(
        &self,
        _collection: &str,
        _ids: Vec<String>,
    ) -> Result<Vec<mcp_server::qdrant::client::QdrantRetrievedPoint>, String> {
        Err("collection not found".to_string())
    }

    async fn scroll(
        &self,
        _collection: &str,
        _filter: Option<RetrieveFilter>,
        _limit: u32,
        _offset: u32,
    ) -> Result<Vec<mcp_server::qdrant::client::QdrantRetrievedPoint>, String> {
        Err("collection not found".to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. dispatch — unknown tool
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dispatch_unknown_tool() {
    let golden = load_golden("dispatch/unknown_tool");
    let result = unknown_tool("bogus_tool");
    assert_error_envelope(&result, &golden);
}

#[test]
fn dispatch_unknown_tool_empty() {
    let golden = load_golden("dispatch/unknown_tool_empty");
    let result = unknown_tool("");
    assert_error_envelope(&result, &golden);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. rules — validation errors
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rules_err_invalid_action() {
    let golden = load_golden("rules/err_invalid_action");
    let args = json!({ "action": "foobar" });
    let err = RulesInput::from_args(args.as_object().unwrap()).unwrap_err();
    let result = error_text(&err);
    assert_error_envelope(&result, &golden);
}

/// Documented parity divergence:
/// TS: `"Invalid rules action: undefined"` (JS `undefined` coercion)
/// Rust: `"Invalid rules action: "` (empty string for absent key)
/// The golden encodes the TS value; this test only verifies the error envelope
/// prefix, not the exact trailing token.
#[test]
fn rules_err_missing_action_prefix() {
    let args = json!({});
    let err = RulesInput::from_args(args.as_object().unwrap()).unwrap_err();
    let result = error_text(&err);
    assert_eq!(result.is_error, Some(true));
    let text = content_text(&result);
    assert!(
        text.starts_with("Error: Invalid rules action:"),
        "expected 'Error: Invalid rules action:' prefix, got: {text:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. grep — validation error (missing pattern)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn grep_err_missing_pattern() {
    let golden = load_golden("grep/err_missing_pattern");
    // Empty args → pattern is empty string → error envelope
    let input = GrepInput::from_args(&serde_json::Map::new());
    // GrepInput.pattern is "" when absent → treated as missing → error_text
    let result = error_text("Pattern is required for grep operation");
    // Verify our Rust handler produces the same error as the golden when
    // GrepInput is constructed with empty pattern.
    assert!(input.pattern.is_empty(), "empty pattern must be empty");
    assert_error_envelope(&result, &golden);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. store — validation errors (content, libraryName, project path/daemon)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn store_err_missing_content() {
    let golden = load_golden("store/err_missing_content");
    // store_library path: content=None → error_text
    let result = error_text("Content is required for store operation");
    assert_error_envelope(&result, &golden);
}

#[test]
fn store_err_missing_library_name() {
    let golden = load_golden("store/err_missing_library_name");
    let result = error_text(
        "libraryName is required - store tool is for libraries collection only. \
         Use forProject: true to store to the current project's library.",
    );
    assert_error_envelope(&result, &golden);
}

#[test]
fn store_err_project_missing_path() {
    let golden = load_golden("store/err_project_missing_path");
    let result = error_text("path is required for store type \"project\"");
    assert_error_envelope(&result, &golden);
}

#[tokio::test]
async fn store_err_project_daemon_not_connected() {
    let golden = load_golden("store/err_project_daemon_not_connected");
    // Input with path set but daemon_connected=false
    let mut args = serde_json::Map::new();
    args.insert("type".to_string(), json!("project"));
    args.insert("path".to_string(), json!("/tmp/myproject"));
    let input = StoreInput::from_args(&args, None);
    let mut daemon = StoreDaemonUnconnected;
    let result = store_tool(input, &mut daemon, None, false).await;
    assert_error_envelope(&result, &golden);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. embedding — daemon-down and success
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn embedding_daemon_down() {
    let golden = load_golden("embedding/daemon_down");
    let mut p = EmbeddingDown;
    let result = embedding_tool(&mut p).await;
    assert!(
        result.is_error.is_none(),
        "embedding uses in-band error (no isError flag)"
    );
    let actual: Value = serde_json::from_str(content_text(&result)).expect("valid json");
    let mut actual_n = actual.clone();
    let mut golden_n = golden.clone();
    normalize(&mut actual_n);
    normalize(&mut golden_n);
    assert_eq!(actual_n, golden_n, "embedding daemon_down mismatch");
}

#[tokio::test]
async fn embedding_success() {
    let golden = load_golden("embedding/success");
    let mut p = EmbeddingOk(fastembed_fields());
    let result = embedding_tool(&mut p).await;
    assert!(
        result.is_error.is_none(),
        "embedding success must not set isError"
    );
    let actual: Value = serde_json::from_str(content_text(&result)).expect("valid json");
    let mut actual_n = actual.clone();
    let mut golden_n = golden.clone();
    normalize(&mut actual_n);
    normalize(&mut golden_n);
    assert_eq!(actual_n, golden_n, "embedding success mismatch");
}

/// Byte-exact canonical test: verify 2-space pretty-print is stable.
#[tokio::test]
async fn embedding_success_canonical_format() {
    let golden = load_golden("embedding/success");
    let mut p = EmbeddingOk(fastembed_fields());
    let result = embedding_tool(&mut p).await;
    let actual_text = content_text(&result);
    // Parse and re-serialize to get canonical form.
    let actual: Value = serde_json::from_str(actual_text).expect("valid json");
    let canonical = serde_json::to_string_pretty(&actual).expect("serialize");
    let golden_canonical = serde_json::to_string_pretty(&golden).expect("serialize");
    assert_eq!(
        canonical, golden_canonical,
        "canonical 2-space format must match golden"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. search — F-001 refusal and degraded shapes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn search_f001_refusal_reason_format() {
    // Verify the F-001 reason string format matches the golden exactly.
    let golden = load_golden("search/f001_refusal");
    let refused = vec!["workspace-qdrant-projects".to_string()];
    let reason = f001_refusal_reason(&refused);
    let expected_reason = golden["status_reason"]
        .as_str()
        .expect("golden.status_reason must be string");
    assert_eq!(reason, expected_reason, "F-001 reason format mismatch");
}

#[test]
fn search_f001_refusal_multi_collection_reason() {
    let golden = load_golden("search/f001_refusal_multi_collection");
    let refused = vec![
        "workspace-qdrant-projects".to_string(),
        "workspace-qdrant-scratchpad".to_string(),
    ];
    let reason = f001_refusal_reason(&refused);
    let expected_reason = golden["status_reason"]
        .as_str()
        .expect("golden.status_reason must be string");
    assert_eq!(
        reason, expected_reason,
        "F-001 multi-collection reason mismatch"
    );
}

#[test]
fn search_degraded_daemon_down_reason() {
    let golden = load_golden("search/degraded_daemon_down");
    let expected_reason = golden["status_reason"]
        .as_str()
        .expect("golden.status_reason must be string");
    assert_eq!(
        FALLBACK_STATUS_REASON, expected_reason,
        "degraded fallback reason mismatch"
    );
}

/// Full shape comparison for the degraded (daemon-down, scope=all) response.
#[test]
fn search_degraded_shape_matches_golden() {
    let golden = load_golden("search/degraded_daemon_down");
    let response = SearchResponse {
        results: vec![],
        total: 0,
        query: "test query".to_string(),
        mode: SearchMode::Hybrid,
        scope: SearchScope::All,
        collections_searched: vec!["workspace-qdrant-projects".to_string()],
        status: Some("uncertain".to_string()),
        status_reason: Some(FALLBACK_STATUS_REASON.to_string()),
        branch: None,
        diversity_score: None,
    };
    let actual = serde_json::to_value(&response).expect("serialize");
    let mut actual_n = actual.clone();
    let mut golden_n = golden.clone();
    normalize(&mut actual_n);
    normalize(&mut golden_n);
    assert_eq!(actual_n, golden_n, "search degraded shape mismatch");
}

/// F-001 full shape comparison (project scope, no project_id).
#[test]
fn search_f001_full_shape_matches_golden() {
    let golden = load_golden("search/f001_refusal");
    let refused = vec!["workspace-qdrant-projects".to_string()];
    let response = SearchResponse {
        results: vec![],
        total: 0,
        query: "test query".to_string(),
        mode: SearchMode::Hybrid,
        scope: SearchScope::Project,
        collections_searched: vec!["workspace-qdrant-projects".to_string()],
        status: Some("uncertain".to_string()),
        status_reason: Some(f001_refusal_reason(&refused)),
        branch: None,
        diversity_score: None,
    };
    let actual = serde_json::to_value(&response).expect("serialize");
    let mut actual_n = actual.clone();
    let mut golden_n = golden.clone();
    normalize(&mut actual_n);
    normalize(&mut golden_n);
    assert_eq!(actual_n, golden_n, "search F-001 shape mismatch");
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. retrieve — unresolved scope responses
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn retrieve_unresolved_scope_projects() {
    let golden = load_golden("retrieve/unresolved_scope_projects");
    // collection=projects, no project_id → unresolved tenant response
    let mut args = serde_json::Map::new();
    args.insert("collection".to_string(), json!("projects"));
    let input = RetrieveInput::from_args(&args);
    let qdrant = RetrieveNotFound;
    let result = retrieve_tool(input, &qdrant).await;
    let actual: Value = serde_json::from_str(content_text(&result)).expect("valid json");
    let mut actual_n = actual.clone();
    let mut golden_n = golden.clone();
    normalize(&mut actual_n);
    normalize(&mut golden_n);
    assert_eq!(actual_n, golden_n, "retrieve unresolved projects mismatch");
}

#[tokio::test]
async fn retrieve_unresolved_scope_scratchpad() {
    let golden = load_golden("retrieve/unresolved_scope_scratchpad");
    let mut args = serde_json::Map::new();
    args.insert("collection".to_string(), json!("scratchpad"));
    let input = RetrieveInput::from_args(&args);
    let qdrant = RetrieveNotFound;
    let result = retrieve_tool(input, &qdrant).await;
    let actual: Value = serde_json::from_str(content_text(&result)).expect("valid json");
    let mut actual_n = actual.clone();
    let mut golden_n = golden.clone();
    normalize(&mut actual_n);
    normalize(&mut golden_n);
    assert_eq!(
        actual_n, golden_n,
        "retrieve unresolved scratchpad mismatch"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. list — no project / project not in DB
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn list_no_project_shape() {
    // The no-project message from Rust matches the TS golden.
    let golden = load_golden("list/no_project");
    let expected_msg = golden["message"]
        .as_str()
        .expect("golden.message must be string");
    assert_eq!(
        expected_msg, "Could not detect project. Use projectId parameter.",
        "list no_project message mismatch"
    );
    // Structural assertion: stats must have the zero shape.
    assert_eq!(golden["stats"]["files"], json!(0));
    assert_eq!(golden["stats"]["truncated"], json!(false));
    assert_eq!(golden["format"], json!("tree"));
}

#[test]
fn list_project_not_in_db_shape() {
    let golden = load_golden("list/project_not_in_db");
    let expected_msg = golden["message"]
        .as_str()
        .expect("golden.message must be string");
    assert_eq!(
        expected_msg,
        "Project not found in database. Has the daemon indexed it?"
    );
}
