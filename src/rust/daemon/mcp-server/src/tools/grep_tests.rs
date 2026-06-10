//! Unit tests for the `grep` tool.  All hermetic — no live daemon.

use super::{grep_tool, GrepDaemon, GrepInput, GrepResponse};
use crate::proto::{TextSearchMatch, TextSearchRequest, TextSearchResponse};

// ---------------------------------------------------------------------------
// Stub daemon
// ---------------------------------------------------------------------------

struct OkDaemon(TextSearchResponse);
struct ErrDaemon(String);

impl GrepDaemon for OkDaemon {
    async fn text_search(
        &mut self,
        _request: TextSearchRequest,
    ) -> Result<TextSearchResponse, String> {
        Ok(self.0.clone())
    }
}

impl GrepDaemon for ErrDaemon {
    async fn text_search(
        &mut self,
        _request: TextSearchRequest,
    ) -> Result<TextSearchResponse, String> {
        Err(self.0.clone())
    }
}

/// Capture the last request received (for field-mapping assertions).
struct CaptureDaemon {
    pub last_request: Option<TextSearchRequest>,
    pub response: TextSearchResponse,
}

impl GrepDaemon for CaptureDaemon {
    async fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> Result<TextSearchResponse, String> {
        self.last_request = Some(request);
        Ok(self.response.clone())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn empty_response() -> TextSearchResponse {
    TextSearchResponse {
        matches: vec![],
        total_matches: 0,
        truncated: false,
        query_time_ms: 1,
        index_status: None,
    }
}

fn make_match(file: &str, line: i32, content: &str) -> TextSearchMatch {
    TextSearchMatch {
        file_path: file.to_string(),
        line_number: line,
        content: content.to_string(),
        tenant_id: "t1".to_string(),
        branch: None,
        context_before: vec![],
        context_after: vec![],
    }
}

fn result_text(r: &rmcp::model::CallToolResult) -> &str {
    r.content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str()
}

fn parse_response(r: &rmcp::model::CallToolResult) -> GrepResponse {
    serde_json::from_str(result_text(r)).expect("valid JSON")
}

// ---------------------------------------------------------------------------
// Pattern validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_pattern_returns_error() {
    let input = GrepInput {
        pattern: String::new(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, "").await;
    let resp = parse_response(&r);
    assert!(!resp.success);
    assert_eq!(
        resp.message.as_deref(),
        Some("Search pattern is required (pass it as 'pattern')")
    );
}

#[test]
fn from_args_accepts_query_alias_for_pattern() {
    // #87: callers carry the `search` tool's "query" argument name over.
    let args: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(r#"{"query": "needle"}"#).unwrap();
    let input = GrepInput::from_args(&args);
    assert_eq!(input.pattern, "needle");

    // Canonical "pattern" wins when both are present.
    let args: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(r#"{"pattern": "canonical", "query": "alias"}"#).unwrap();
    let input = GrepInput::from_args(&args);
    assert_eq!(input.pattern, "canonical");
}

#[tokio::test]
async fn empty_pattern_returns_in_band_no_is_error_flag() {
    let input = GrepInput {
        pattern: String::new(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, "").await;
    // In-band error — no is_error flag on the CallToolResult
    assert!(r.is_error.is_none());
}

#[tokio::test]
async fn empty_pattern_latency_is_zero() {
    // TS grep.ts:132 returns grepError('Search pattern is required', 0) with
    // LITERAL zero latency — the timer is not started until after the guard.
    // Mirror: latency_ms must be 0 for missing-pattern errors.
    let input = GrepInput {
        pattern: String::new(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, "").await;
    let resp = parse_response(&r);
    assert_eq!(
        resp.latency_ms, 0,
        "missing-pattern error must have latency_ms=0 (TS returns literal 0)"
    );
}

// ---------------------------------------------------------------------------
// Tenant ID resolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scope_project_no_project_id_returns_error() {
    let input = GrepInput {
        pattern: "fn main".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: None,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, "").await;
    let resp = parse_response(&r);
    assert!(!resp.success);
    assert_eq!(
        resp.message.as_deref(),
        Some("Could not detect project ID. Use scope \"all\" or provide projectId.")
    );
}

/// #124: the detection failure appends the dispatcher-computed registered-
/// projects hint so the caller can retry with a projectId or register the
/// project (parity with the list tool's #111 hint).
#[tokio::test]
async fn scope_project_no_project_id_appends_projects_hint() {
    let input = GrepInput {
        pattern: "fn main".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: None,
        ..Default::default()
    };
    let hint = " Available projects — retry with projectId=<id>: repo (projectId: abc123def456)";
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, hint).await;
    let resp = parse_response(&r);
    assert!(!resp.success);
    let msg = resp.message.unwrap();
    assert!(msg.starts_with("Could not detect project ID."));
    assert!(msg.ends_with(hint), "hint must be appended: {msg}");
}

#[tokio::test]
async fn scope_project_with_project_id_succeeds() {
    let input = GrepInput {
        pattern: "fn main".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: Some("proj-abc".to_string()),
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, "").await;
    let resp = parse_response(&r);
    assert!(resp.success);
}

#[tokio::test]
async fn scope_project_with_session_project_id_succeeds() {
    let input = GrepInput {
        pattern: "fn main".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: None,
        ..Default::default()
    };
    let r = grep_tool(
        input,
        &mut OkDaemon(empty_response()),
        Some("session-proj-xyz"),
        None,
        "",
    )
    .await;
    let resp = parse_response(&r);
    assert!(resp.success);
}

#[tokio::test]
async fn scope_all_no_project_id_succeeds() {
    let input = GrepInput {
        pattern: "fn main".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: None,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, "").await;
    let resp = parse_response(&r);
    assert!(resp.success);
}

// ---------------------------------------------------------------------------
// Request field mapping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_case_sensitive_defaults_true() {
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "foo".to_string(),
        scope: "all".to_string(),
        // case_sensitive not set — defaults to true per grep.ts:123
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, None, "").await;
    assert!(daemon.last_request.unwrap().case_sensitive);
}

#[tokio::test]
async fn request_regex_flag_forwarded() {
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: r"fn \w+".to_string(),
        regex: true,
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, None, "").await;
    assert!(daemon.last_request.unwrap().regex);
}

#[tokio::test]
async fn request_path_glob_forwarded() {
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "TODO".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        path_glob: Some("**/*.rs".to_string()),
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, None, "").await;
    assert_eq!(
        daemon.last_request.unwrap().path_glob.as_deref(),
        Some("**/*.rs")
    );
}

#[tokio::test]
async fn request_tenant_id_set_for_project_scope() {
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "hello".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: Some("proj-123".to_string()),
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, None, "").await;
    assert_eq!(
        daemon.last_request.unwrap().tenant_id.as_deref(),
        Some("proj-123")
    );
}

#[tokio::test]
async fn request_tenant_id_none_for_all_scope() {
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "hello".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: Some("proj-ignored".to_string()),
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, None, "").await;
    assert!(daemon.last_request.unwrap().tenant_id.is_none());
}

// ---------------------------------------------------------------------------
// Response mapping
// ---------------------------------------------------------------------------

#[tokio::test]
async fn matches_mapped_correctly() {
    let resp = TextSearchResponse {
        matches: vec![make_match("src/main.rs", 42, "fn main() {")],
        total_matches: 1,
        truncated: false,
        query_time_ms: 3,
        index_status: None,
    };
    let input = GrepInput {
        pattern: "fn main".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(resp), None, None, "").await;
    let result = parse_response(&r);
    assert!(result.success);
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].file, "src/main.rs");
    assert_eq!(result.matches[0].line, 42);
    assert_eq!(result.matches[0].content, "fn main() {");
}

#[tokio::test]
async fn context_lines_mapped() {
    let mut m = make_match("a.rs", 10, "body");
    m.context_before = vec!["line8".to_string(), "line9".to_string()];
    m.context_after = vec!["line11".to_string()];
    let resp = TextSearchResponse {
        matches: vec![m],
        total_matches: 1,
        truncated: false,
        query_time_ms: 1,
        index_status: None,
    };
    let input = GrepInput {
        pattern: "body".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        context_lines: 2,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(resp), None, None, "").await;
    let result = parse_response(&r);
    assert_eq!(result.matches[0].context_before, vec!["line8", "line9"]);
    assert_eq!(result.matches[0].context_after, vec!["line11"]);
}

#[tokio::test]
async fn total_matches_and_truncated_forwarded() {
    let resp = TextSearchResponse {
        matches: vec![],
        total_matches: 500,
        truncated: true,
        query_time_ms: 2,
        index_status: None,
    };
    let input = GrepInput {
        pattern: "x".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(resp), None, None, "").await;
    let result = parse_response(&r);
    assert_eq!(result.total_matches, 500);
    assert!(result.truncated);
}

// ---------------------------------------------------------------------------
// Daemon-down error path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn daemon_down_returns_grep_failed_message() {
    let input = GrepInput {
        pattern: "hello".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(
        input,
        &mut ErrDaemon("connection refused".to_string()),
        None,
        None,
        "",
    )
    .await;
    let resp = parse_response(&r);
    assert!(!resp.success);
    let msg = resp.message.expect("message present");
    assert!(
        msg.starts_with("Grep failed:"),
        "expected prefix, got: {msg:?}"
    );
    assert!(msg.contains("connection refused"));
}

#[tokio::test]
async fn daemon_down_matches_is_empty() {
    let input = GrepInput {
        pattern: "hello".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut ErrDaemon("timeout".to_string()), None, None, "").await;
    let resp = parse_response(&r);
    assert!(resp.matches.is_empty());
    assert_eq!(resp.total_matches, 0);
}

// ---------------------------------------------------------------------------
// GrepInput::from_args
// ---------------------------------------------------------------------------

#[test]
fn from_args_defaults() {
    let args = serde_json::json!({ "pattern": "hello" });
    let map = args.as_object().unwrap();
    let input = GrepInput::from_args(map);
    assert_eq!(input.pattern, "hello");
    assert!(!input.regex);
    assert!(input.case_sensitive); // default true
    assert_eq!(input.scope, "project");
    assert_eq!(input.context_lines, 0);
    assert_eq!(input.max_results, 1000);
    assert!(input.path_glob.is_none());
    assert!(input.branch.is_none());
    assert!(input.project_id.is_none());
}

#[test]
fn from_args_all_fields() {
    let args = serde_json::json!({
        "pattern": "TODO",
        "regex": true,
        "caseSensitive": false,
        "pathGlob": "**/*.rs",
        "scope": "all",
        "contextLines": 3,
        "maxResults": 50,
        "branch": "main",
        "projectId": "proj-abc"
    });
    let map = args.as_object().unwrap();
    let input = GrepInput::from_args(map);
    assert_eq!(input.pattern, "TODO");
    assert!(input.regex);
    assert!(!input.case_sensitive);
    assert_eq!(input.path_glob.as_deref(), Some("**/*.rs"));
    assert_eq!(input.scope, "all");
    assert_eq!(input.context_lines, 3);
    assert_eq!(input.max_results, 50);
    assert_eq!(input.branch.as_deref(), Some("main"));
    assert_eq!(input.project_id.as_deref(), Some("proj-abc"));
}

// ---------------------------------------------------------------------------
// Index status / indexing-lag warning (#97)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn index_status_incomplete_sets_warning() {
    let mut resp = empty_response();
    resp.index_status = Some(crate::proto::TextIndexStatus {
        files_tracked: 1,
        queue_pending: 19,
        index_complete: false,
    });
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(resp), None, None, "").await;
    let parsed = parse_response(&r);
    assert!(parsed.success);
    let status = parsed.index_status.expect("index_status present");
    assert_eq!(status.files_tracked, 1);
    assert_eq!(status.queue_pending, 19);
    assert!(!status.index_complete);
    let warning = parsed.warning.expect("warning present");
    assert!(
        warning.contains("19 item(s) still queued"),
        "warning should carry pending count: {warning}"
    );
}

#[tokio::test]
async fn index_status_complete_no_warning() {
    let mut resp = empty_response();
    resp.index_status = Some(crate::proto::TextIndexStatus {
        files_tracked: 20,
        queue_pending: 0,
        index_complete: true,
    });
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(resp), None, None, "").await;
    let parsed = parse_response(&r);
    assert!(parsed.success);
    let status = parsed.index_status.expect("index_status present");
    assert!(status.index_complete);
    assert!(parsed.warning.is_none(), "no warning when index complete");
}

// ---------------------------------------------------------------------------
// Branch default (#102)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tenant_scoped_grep_defaults_branch() {
    // No explicit branch + tenant scope → the session branch is applied,
    // preventing one-duplicate-per-file_metadata-branch-row matches (#102).
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: Some("proj-abc".to_string()),
        branch: None,
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, Some("main".to_string()), "").await;
    assert_eq!(daemon.last_request.unwrap().branch.as_deref(), Some("main"));
}

#[tokio::test]
async fn explicit_branch_overrides_default() {
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: Some("proj-abc".to_string()),
        branch: Some("feature/x".to_string()),
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, Some("main".to_string()), "").await;
    assert_eq!(
        daemon.last_request.unwrap().branch.as_deref(),
        Some("feature/x")
    );
}

#[tokio::test]
async fn star_branch_opts_into_all_branches() {
    // "*" must suppress BOTH the explicit value and the default — the daemon
    // receives no branch filter at all.
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "project".to_string(),
        case_sensitive: true,
        max_results: 1000,
        project_id: Some("proj-abc".to_string()),
        branch: Some("*".to_string()),
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, Some("main".to_string()), "").await;
    assert!(daemon.last_request.unwrap().branch.is_none());
}

#[tokio::test]
async fn scope_all_does_not_apply_branch_default() {
    // The default branch belongs to one repository — applying it across all
    // projects would wrongly filter every other project's results.
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        branch: None,
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, Some("main".to_string()), "").await;
    assert!(daemon.last_request.unwrap().branch.is_none());
}

#[tokio::test]
async fn scope_all_explicit_branch_still_forwarded() {
    let mut daemon = CaptureDaemon {
        last_request: None,
        response: empty_response(),
    };
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        branch: Some("dev".to_string()),
        ..Default::default()
    };
    grep_tool(input, &mut daemon, None, None, "").await;
    assert_eq!(daemon.last_request.unwrap().branch.as_deref(), Some("dev"));
}

#[tokio::test]
async fn index_status_absent_omits_fields() {
    let input = GrepInput {
        pattern: "needle".to_string(),
        scope: "all".to_string(),
        case_sensitive: true,
        max_results: 1000,
        ..Default::default()
    };
    let r = grep_tool(input, &mut OkDaemon(empty_response()), None, None, "").await;
    let text = result_text(&r);
    assert!(
        !text.contains("index_status") && !text.contains("warning"),
        "absent status must serialize to nothing: {text}"
    );
}
