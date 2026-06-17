//! `grep` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/grep.ts`.
//!
//! Uses `TextSearchService` (gRPC) via [`GrepDaemon`] to perform FTS5-based
//! code search and returns results as a [`GrepResponse`].
//!
//! # Result shape (field order matches TS `GrepResponse` declaration)
//!
//! ```json
//! // success
//! { "success": true, "matches": [...], "total_matches": 2,
//!   "truncated": false, "latency_ms": 12 }
//!
//! // error (daemon down)
//! { "success": false, "matches": [], "total_matches": 0,
//!   "truncated": false, "latency_ms": 5,
//!   "message": "Grep failed: daemon not reachable (tcp connect error: …) —
//!               is memexd running? Start it with `wqm service start`" }
//! ```
//!
//! `latency_ms` is wall-clock and non-deterministic; golden comparisons
//! must exclude it (the task-33 golden suite masks it).

use std::time::Instant;

use rmcp::model::CallToolResult;
use serde::Serialize;

use crate::proto::{TextSearchMatch, TextSearchRequest, TextSearchResponse};
use crate::tools::envelope::ok_text;

// ---------------------------------------------------------------------------
// Public trait — injectable for tests
// ---------------------------------------------------------------------------

/// Abstraction over the single gRPC call needed by the grep tool.
pub trait GrepDaemon {
    fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> impl std::future::Future<Output = Result<TextSearchResponse, String>> + Send;
}

impl GrepDaemon for crate::grpc::DaemonClient {
    async fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> Result<TextSearchResponse, String> {
        self.text_search(request)
            .await
            .map_err(|s| wqm_client::grpc::status_user_message(&s))
    }
}

// ---------------------------------------------------------------------------
// Input struct
// ---------------------------------------------------------------------------

/// Input arguments for the `grep` tool.
///
/// Field names match the MCP input schema from `definitions.rs` (camelCase
/// in JSON; parsed from `CallToolRequestParams.arguments`).
#[derive(Debug, Default)]
pub struct GrepInput {
    pub pattern: String,
    pub regex: bool,
    /// Default `true` — matches TS grep.ts line 123.
    pub case_sensitive: bool,
    pub path_glob: Option<String>,
    /// "project" | "all" — default "project"
    pub scope: String,
    pub context_lines: u32,
    pub max_results: u32,
    pub branch: Option<String>,
    pub project_id: Option<String>,
}

impl GrepInput {
    /// Parse from the JSON `arguments` map of a `CallToolRequestParams`.
    ///
    /// Mirrors the destructuring defaults in grep.ts lines 120-130.
    pub fn from_args(args: &serde_json::Map<String, serde_json::Value>) -> Self {
        // "query" is accepted as an alias: the sibling `search` tool names its
        // text argument "query", and callers regularly carry that habit over
        // (#87). The schema documents "pattern" as canonical.
        let pattern = args
            .get("pattern")
            .or_else(|| args.get("query"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let regex = args.get("regex").and_then(|v| v.as_bool()).unwrap_or(false);

        // TS default: caseSensitive = true  (grep.ts:123)
        let case_sensitive = args
            .get("caseSensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let path_glob = args
            .get("pathGlob")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let scope = args
            .get("scope")
            .and_then(|v| v.as_str())
            .unwrap_or("project")
            .to_string();

        let context_lines = args
            .get("contextLines")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let max_results = args
            .get("maxResults")
            .and_then(|v| v.as_u64())
            .unwrap_or(1000) as u32;

        let branch = args
            .get("branch")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let project_id = args
            .get("projectId")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        Self {
            pattern,
            regex,
            case_sensitive,
            path_glob,
            scope,
            context_lines,
            max_results,
            branch,
            project_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Result structs — field ORDER matches TS declarations for JSON parity
// ---------------------------------------------------------------------------

/// Single match — mirrors TS `GrepMatch` (grep.ts lines 29-35).
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct GrepMatch {
    pub file: String,
    pub line: i64,
    pub content: String,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

/// Tool response — mirrors TS `GrepResponse` (grep.ts lines 37-44).
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct GrepResponse {
    pub success: bool,
    pub matches: Vec<GrepMatch>,
    pub total_matches: i64,
    pub truncated: bool,
    pub latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Tenant indexing state (#97); present for tenant-scoped requests when
    /// the daemon reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_status: Option<GrepIndexStatus>,
    /// Set when the index is incomplete: results (including zero matches) may
    /// reflect indexing lag rather than pattern absence (#97).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// Tenant indexing state attached to grep responses (#97).
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct GrepIndexStatus {
    pub files_tracked: u64,
    pub queue_pending: u64,
    pub index_complete: bool,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a proto [`TextSearchMatch`] to a [`GrepMatch`].
fn map_match(m: TextSearchMatch) -> GrepMatch {
    GrepMatch {
        file: m.file_path,
        line: m.line_number as i64,
        content: m.content,
        context_before: m.context_before,
        context_after: m.context_after,
    }
}

/// Build the indexing-completeness warning for a grep response (#97, #137, #141).
///
/// Returns `None` when the queried branch is fully searchable. Otherwise it
/// distinguishes the two ways a result can be incomplete so a zero-match result
/// is never silently misread as pattern absence:
///
/// - **Items still queued** (`queue_pending > 0`): the branch is mid-index, so
///   results reflect indexing lag.
/// - **Branch not indexed** (`queue_pending == 0` but `files_tracked == 0`): no
///   files are indexed for this branch yet — a file indexed on another branch is
///   invisible here because `file_metadata` is per-branch (#137). This otherwise
///   looked like a healthy index returning an honest "not found".
fn build_index_warning(status: &GrepIndexStatus) -> Option<String> {
    if status.index_complete {
        return None;
    }
    if status.queue_pending > 0 {
        Some(format!(
            "Index incomplete for this branch: {} item(s) still queued — \
             results may reflect indexing lag rather than pattern absence",
            status.queue_pending
        ))
    } else {
        Some(
            "No files are indexed for this branch yet — results may reflect a \
             not-yet-indexed branch rather than pattern absence. Content indexed \
             on another branch is not visible to a branch-scoped grep."
                .to_string(),
        )
    }
}

/// Build a failure [`GrepResponse`] (daemon down or validation error).
fn grep_error(message: String, latency_ms: u64) -> GrepResponse {
    GrepResponse {
        success: false,
        matches: vec![],
        total_matches: 0,
        truncated: false,
        latency_ms,
        message: Some(message),
        index_status: None,
        warning: None,
    }
}

// ---------------------------------------------------------------------------
// Tool function
// ---------------------------------------------------------------------------

/// Execute the `grep` tool.
///
/// Mirrors `GrepTool.grep()` in grep.ts. Always returns a `CallToolResult`;
/// errors (daemon down, missing pattern) are returned **in-band** with
/// `success: false`.
///
/// `default_branch` is the branch filter applied when the caller did not pass
/// one explicitly (#102): the session's current branch, or the TARGET
/// project's branch for explicit cross-project calls (resolved by the
/// dispatcher, mirroring search/list #99).
///
/// `projects_hint` is the registered-projects retry hint (#111/#124), computed
/// by the dispatcher only when project detection is about to fail; appended to
/// the "Could not detect project ID" error so the caller can retry with a
/// projectId or register the project instead of dead-ending.
pub async fn grep_tool<D>(
    input: GrepInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
    default_branch: Option<String>,
    projects_hint: &str,
) -> CallToolResult
where
    D: GrepDaemon,
{
    // Pattern is required — grep.ts:132.
    // TS returns `grepError('Search pattern is required', 0)` BEFORE starting
    // the timer (line 132 precedes `const startTime = Date.now()` at line 134).
    // Mirror: check pattern BEFORE creating the timer; use literal 0 latency.
    if input.pattern.is_empty() {
        return ok_text(&grep_error(
            "Search pattern is required (pass it as 'pattern')".to_string(),
            0, // TS: grepError(..., 0) — literal zero, timer not yet started
        ));
    }

    let start = Instant::now();

    // Resolve tenant_id for scope=project — grep.ts:137-145
    let tenant_id: Option<String> = if input.scope == "project" {
        let resolved = input
            .project_id
            .clone()
            .or_else(|| session_project_id.map(str::to_string));
        if resolved.is_none() {
            return ok_text(&grep_error(
                format!(
                    "Could not detect project ID. Use scope \"all\" or provide \
                     projectId.{projects_hint}"
                ),
                elapsed_ms(start),
            ));
        }
        resolved
    } else {
        // scope=all: no tenant scoping
        None
    };

    // Branch default (#102): a tenant-scoped grep with no branch filter joins
    // code_lines × ALL file_metadata branch rows and emits one duplicate match
    // per branch row. Default to the session/target branch like search/list
    // (#99). "*" explicitly opts into all branches (no filter sent). The
    // default is NOT applied for scope "all": it belongs to one repository
    // and would wrongly filter every other project.
    let branch = input
        .branch
        .or(if tenant_id.is_some() {
            default_branch
        } else {
            None
        })
        .filter(|b| b != "*");

    let request = TextSearchRequest {
        pattern: input.pattern,
        regex: input.regex,
        case_sensitive: input.case_sensitive,
        context_lines: input.context_lines as i32,
        max_results: input.max_results as i32,
        tenant_id,
        branch,
        path_glob: input.path_glob,
        path_prefix: None,
    };

    match daemon.text_search(request).await {
        Ok(resp) => {
            let matches: Vec<GrepMatch> = resp.matches.into_iter().map(map_match).collect();
            // Indexing state (#97): surface a warning when items are still
            // queued so a zero-match result is not misread as pattern absence.
            let index_status = resp.index_status.map(|s| GrepIndexStatus {
                files_tracked: s.files_tracked,
                queue_pending: s.queue_pending,
                index_complete: s.index_complete,
            });
            let warning = index_status.as_ref().and_then(build_index_warning);
            let response = GrepResponse {
                success: true,
                matches,
                total_matches: resp.total_matches as i64,
                truncated: resp.truncated,
                latency_ms: elapsed_ms(start),
                message: None,
                index_status,
                warning,
            };
            ok_text(&response)
        }
        Err(err) => ok_text(&grep_error(
            format!("Grep failed: {err}"),
            elapsed_ms(start),
        )),
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "grep_tests.rs"]
mod tests;
