//! TextSearchService RPC wrappers for [`DaemonClient`].
//!
//! Mirrors `DaemonClientService.textSearch` and `textSearchCount` from
//! `service-methods.ts` lines 71-92.
//!
//! | Rust method      | Proto RPC                        | TS equivalent       |
//! |------------------|----------------------------------|---------------------|
//! | `text_search`    | `TextSearchService::Search`      | `textSearch()`      |
//! | `count_matches`  | `TextSearchService::CountMatches`| `textSearchCount()` |
//!
//! The wire method name for `text_search` is `"search"` (used in the TS
//! `grpcUnaryWithTimeout` call), which triggers the 10 s budget via
//! [`super::timeouts::resolve_timeout`].  `count_matches` uses the default
//! 5 s budget.
//!
//! ## Internal result type
//!
//! [`TextSearchMatchResult`] is a thin Rust struct that mirrors the proto
//! [`TextSearchMatch`] fields exactly — it provides a typed interface for
//! callers that do not want to work with prost-generated types directly.

use tonic::Status;

use wqm_proto::workspace_daemon::{TextSearchMatch, TextSearchRequest, TextSearchResponse};

use super::client::DaemonClient;

/// Internal representation of a single text-search match.
///
/// Field names match the proto `TextSearchMatch` message (and the TS
/// `TextSearchMatch` interface in `grpc-types-search-graph.ts` lines 29-37)
/// exactly:
/// - `file_path`: relative path from project root
/// - `line_number`: 1-based line number
/// - `content`: full content of the matching line
/// - `tenant_id`: project tenant identifier
/// - `branch`: optional branch scope
/// - `context_before`: lines before the match (up to `context_lines`)
/// - `context_after`: lines after the match (up to `context_lines`)
#[derive(Debug, Clone, PartialEq)]
pub struct TextSearchMatchResult {
    pub file_path: String,
    pub line_number: i32,
    pub content: String,
    pub tenant_id: String,
    pub branch: Option<String>,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

impl From<TextSearchMatch> for TextSearchMatchResult {
    fn from(m: TextSearchMatch) -> Self {
        Self {
            file_path: m.file_path,
            line_number: m.line_number,
            content: m.content,
            tenant_id: m.tenant_id,
            branch: m.branch,
            context_before: m.context_before,
            context_after: m.context_after,
        }
    }
}

impl DaemonClient {
    /// Full-text search across indexed code — mirrors TS `textSearch()`.
    ///
    /// The wire method name is `"search"` (see service-methods.ts line 77),
    /// which causes [`super::timeouts::resolve_timeout`] to apply the 10 s
    /// search budget.
    ///
    /// # Request fields (mirrors [`TextSearchRequest`] proto)
    /// - `pattern`: exact substring or regex pattern
    /// - `regex`: treat `pattern` as regex
    /// - `case_sensitive`: case-sensitive matching
    /// - `tenant_id`: optional project scope
    /// - `branch`: optional branch scope
    /// - `path_glob`: optional glob filter (e.g. `"**/*.rs"`)
    /// - `context_lines`: lines of context before/after (0 = none)
    /// - `max_results`: 0 = unlimited
    ///
    /// # Errors
    /// Returns `Err(Status)` on transport, timeout, or daemon error.
    pub async fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> Result<TextSearchResponse, Status> {
        // Wire method name "search" → 10 s budget (matches TS line 77)
        let client = self.text_search.clone();
        self.call("search", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.search(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Count text-search matches without returning results — mirrors TS `textSearchCount()`.
    ///
    /// Uses the same [`TextSearchRequest`] as [`DaemonClient::text_search`] but
    /// calls the `CountMatches` RPC, returning only the count and query time.
    ///
    /// Uses the default 5 s budget (`"countMatches"` does not contain `"search"`
    /// in the method-name check).
    ///
    /// # Errors
    /// Returns `Err(Status)` on transport, timeout, or daemon error.
    pub async fn count_matches(
        &mut self,
        request: TextSearchRequest,
    ) -> Result<wqm_proto::workspace_daemon::TextSearchCountResponse, Status> {
        let client = self.text_search.clone();
        self.call("countMatches", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.count_matches(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wqm_proto::workspace_daemon::{TextSearchCountResponse, TextSearchMatch};

    // Helpers: build proto types directly (no live daemon needed).

    fn make_search_request(pattern: &str) -> TextSearchRequest {
        TextSearchRequest {
            pattern: pattern.to_string(),
            regex: false,
            case_sensitive: true,
            tenant_id: None,
            branch: None,
            path_glob: None,
            path_prefix: None,
            context_lines: 0,
            max_results: 100,
        }
    }

    fn make_search_match(file: &str, line: i32, content: &str, tenant: &str) -> TextSearchMatch {
        TextSearchMatch {
            file_path: file.to_string(),
            line_number: line,
            content: content.to_string(),
            tenant_id: tenant.to_string(),
            branch: None,
            context_before: vec![],
            context_after: vec![],
        }
    }

    fn make_search_response(matches: Vec<TextSearchMatch>, truncated: bool) -> TextSearchResponse {
        let total = matches.len() as i32;
        TextSearchResponse {
            matches,
            total_matches: total,
            truncated,
            query_time_ms: 5,
        }
    }

    fn make_count_response(count: i32) -> TextSearchCountResponse {
        TextSearchCountResponse {
            count,
            query_time_ms: 2,
        }
    }

    // ── TextSearchRequest field mapping ───────────────────────────────────────

    #[test]
    fn search_request_pattern_field() {
        let req = make_search_request("fn main");
        assert_eq!(req.pattern, "fn main");
    }

    #[test]
    fn search_request_defaults() {
        let req = make_search_request("foo");
        assert!(!req.regex);
        assert!(req.case_sensitive);
        assert!(req.tenant_id.is_none());
        assert!(req.branch.is_none());
        assert!(req.path_glob.is_none());
        assert!(req.path_prefix.is_none());
        assert_eq!(req.context_lines, 0);
        assert_eq!(req.max_results, 100);
    }

    #[test]
    fn search_request_with_all_optional_fields() {
        let req = TextSearchRequest {
            pattern: "TODO".to_string(),
            regex: false,
            case_sensitive: false,
            tenant_id: Some("abc123".to_string()),
            branch: Some("main".to_string()),
            path_glob: Some("**/*.rs".to_string()),
            path_prefix: Some("src/".to_string()),
            context_lines: 2,
            max_results: 50,
        };
        assert_eq!(req.tenant_id.as_deref(), Some("abc123"));
        assert_eq!(req.branch.as_deref(), Some("main"));
        assert_eq!(req.path_glob.as_deref(), Some("**/*.rs"));
        assert_eq!(req.path_prefix.as_deref(), Some("src/"));
        assert_eq!(req.context_lines, 2);
        assert_eq!(req.max_results, 50);
    }

    #[test]
    fn search_request_regex_flag() {
        let mut req = make_search_request(r"fn \w+");
        req.regex = true;
        assert!(req.regex);
    }

    // ── TextSearchMatch → TextSearchMatchResult mapping ───────────────────────

    #[test]
    fn match_result_from_proto_file_path() {
        let m = make_search_match("src/main.rs", 42, "fn main() {", "abc123");
        let r = TextSearchMatchResult::from(m);
        assert_eq!(r.file_path, "src/main.rs");
    }

    #[test]
    fn match_result_from_proto_line_number() {
        let m = make_search_match("src/main.rs", 42, "fn main() {", "abc123");
        let r = TextSearchMatchResult::from(m);
        assert_eq!(r.line_number, 42);
    }

    #[test]
    fn match_result_from_proto_content() {
        let m = make_search_match("src/lib.rs", 10, "pub fn foo()", "t1");
        let r = TextSearchMatchResult::from(m);
        assert_eq!(r.content, "pub fn foo()");
    }

    #[test]
    fn match_result_from_proto_tenant_id() {
        let m = make_search_match("src/lib.rs", 1, "x", "tenant42");
        let r = TextSearchMatchResult::from(m);
        assert_eq!(r.tenant_id, "tenant42");
    }

    #[test]
    fn match_result_from_proto_branch_none() {
        let m = make_search_match("src/lib.rs", 1, "x", "t1");
        let r = TextSearchMatchResult::from(m);
        assert!(r.branch.is_none());
    }

    #[test]
    fn match_result_from_proto_branch_some() {
        let mut m = make_search_match("src/lib.rs", 1, "x", "t1");
        m.branch = Some("feat/my-branch".to_string());
        let r = TextSearchMatchResult::from(m);
        assert_eq!(r.branch.as_deref(), Some("feat/my-branch"));
    }

    #[test]
    fn match_result_context_before_empty() {
        let m = make_search_match("src/lib.rs", 5, "y", "t1");
        let r = TextSearchMatchResult::from(m);
        assert!(r.context_before.is_empty());
    }

    #[test]
    fn match_result_context_before_populated() {
        let mut m = make_search_match("src/lib.rs", 5, "y", "t1");
        m.context_before = vec!["line3".to_string(), "line4".to_string()];
        m.context_after = vec!["line6".to_string()];
        let r = TextSearchMatchResult::from(m);
        assert_eq!(r.context_before, vec!["line3", "line4"]);
        assert_eq!(r.context_after, vec!["line6"]);
    }

    // ── TextSearchResponse field mapping ──────────────────────────────────────

    #[test]
    fn search_response_total_matches() {
        let m = make_search_match("a.rs", 1, "x", "t1");
        let resp = make_search_response(vec![m], false);
        assert_eq!(resp.total_matches, 1);
        assert!(!resp.truncated);
    }

    #[test]
    fn search_response_truncated_flag() {
        let resp = make_search_response(vec![], true);
        assert!(resp.truncated);
    }

    #[test]
    fn search_response_query_time_ms() {
        let resp = make_search_response(vec![], false);
        assert_eq!(resp.query_time_ms, 5);
    }

    #[test]
    fn search_response_matches_len() {
        let matches = vec![
            make_search_match("a.rs", 1, "x", "t1"),
            make_search_match("b.rs", 2, "y", "t1"),
        ];
        let resp = make_search_response(matches, false);
        assert_eq!(resp.matches.len(), 2);
        assert_eq!(resp.total_matches, 2);
    }

    // ── TextSearchCountResponse field mapping ─────────────────────────────────

    #[test]
    fn count_response_count_field() {
        let resp = make_count_response(42);
        assert_eq!(resp.count, 42);
    }

    #[test]
    fn count_response_query_time_ms() {
        let resp = make_count_response(0);
        assert_eq!(resp.query_time_ms, 2);
    }

    #[test]
    fn count_response_zero() {
        let resp = make_count_response(0);
        assert_eq!(resp.count, 0);
    }

    // ── Wire method name: "search" uses 10 s budget ───────────────────────────

    #[tokio::test]
    async fn text_search_wire_name_uses_10s_budget() {
        // "search" contains "search" → 10 s budget via resolve_timeout.
        // We verify the method name resolves as expected using a tiny override.
        use crate::grpc::timeouts::resolve_timeout;
        use std::time::Duration;
        let budget = resolve_timeout("search", None);
        assert_eq!(budget, Duration::from_secs(10));
    }

    #[tokio::test]
    async fn count_matches_method_uses_5s_budget() {
        use crate::grpc::timeouts::resolve_timeout;
        use std::time::Duration;
        // "countMatches" does not contain "search" → 5 s default.
        let budget = resolve_timeout("countMatches", None);
        assert_eq!(budget, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn text_search_call_times_out_correctly() {
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call("search", Some(Duration::from_millis(1)), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }

    // ── DaemonClient construction ─────────────────────────────────────────────

    #[tokio::test]
    async fn daemon_client_constructs_for_search_calls() {
        let result = DaemonClient::new("http://127.0.0.1:50051");
        assert!(result.is_ok());
    }
}
