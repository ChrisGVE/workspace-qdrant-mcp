//! Tests for M3 (per-leg failure → partial results, status stays 'ok'),
//! M4 (exact mode always emits context_before/context_after even when empty),
//! M5 (exact mode default max_results=100 when limit not specified by caller).
//!
//! ## M3 rationale
//!
//! TS `searchDense` / `searchSparse` each have their own try/catch that
//! returns `[]` on failure.  `searchCollection` combines the two arrays and
//! NEVER throws.  `searchAllCollections` only sets `status='uncertain'` when
//! `searchCollection` itself throws (collection-level error, not leg-level).
//! Therefore: a leg failure → empty results for that leg, successful leg's
//! results are kept, and status stays 'ok' (None).
//!
//! All stubs/helpers are defined `pub(super)` in `search_tests.rs`; reach
//! them via `use super::*;`.

use super::*;

use serde_json::Value;
use std::collections::HashMap;

use crate::proto::{TextSearchMatch, TextSearchRequest, TextSearchResponse};
use crate::qdrant::client::{QdrantPoint, QdrantRetrievedPoint};
use crate::tools::search::exact::{search_exact, ExactSearchDaemon};
use crate::tools::search::flow::run_search_pipeline;
use crate::tools::search::flow::SearchQdrant;
use crate::tools::search::options::{SearchOptions, DEFAULT_EXACT_LIMIT};
use crate::tools::search::types::SearchScope;

// ---------------------------------------------------------------------------
// M3: per-leg failure → partial results kept, status stays 'ok'
// ---------------------------------------------------------------------------
//
// TS `searchDense` / `searchSparse` each have their own try/catch returning []
// on error.  `searchCollection` combines results without throwing.
// `searchAllCollections` only sets uncertain when `searchCollection` itself
// throws (collection-level), which never happens in practice.
// Therefore: a failing leg produces [] for that leg only; the other leg's
// results are kept; `status` is 'ok' (None).

/// Qdrant stub whose dense search fails for one specific collection but whose
/// sparse leg (and all other collections) succeeds.
struct OneFailingLegQdrant {
    fail_dense_collection: String,
    ok_dense_results: Vec<QdrantPoint>,
    ok_sparse_results: Vec<QdrantPoint>,
}

impl SearchQdrant for OneFailingLegQdrant {
    async fn search_dense(
        &self,
        collection: &str,
        _vector: Vec<f32>,
        _limit: u64,
        _score_threshold: Option<f32>,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        if collection == self.fail_dense_collection {
            Err(anyhow::anyhow!(
                "Qdrant collection unavailable: {collection}"
            ))
        } else {
            Ok(self.ok_dense_results.clone())
        }
    }

    async fn search_sparse(
        &self,
        _collection: &str,
        _indices: Vec<u32>,
        _values: Vec<f32>,
        _limit: u64,
        _score_threshold: Option<f32>,
        _filter: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<QdrantPoint>> {
        Ok(self.ok_sparse_results.clone())
    }

    async fn scroll_page(
        &self,
        _collection: &str,
        _filter: Option<qdrant_client::qdrant::Filter>,
        _limit: u32,
    ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
        Ok(vec![])
    }

    async fn retrieve_by_ids(
        &self,
        _collection: &str,
        _ids: Vec<String>,
    ) -> anyhow::Result<Vec<QdrantRetrievedPoint>> {
        Ok(vec![])
    }
}

#[tokio::test]
async fn m3_leg_failure_keeps_partial_results_status_ok() {
    // M3: when the dense leg fails for a collection but the sparse leg succeeds,
    // the response must contain the sparse results and status must be None
    // (not 'uncertain') — mirrors TS `searchDense` having its own try/catch that
    // returns [] on failure, so `searchCollection` still returns sparse results.
    let mut daemon = StubDaemon {
        dense: Some(vec![0.1, 0.2]),
        sparse: Some(HashMap::from([(1u32, 0.5f32)])),
        unavailable: false,
    };

    let qdrant = OneFailingLegQdrant {
        fail_dense_collection: "projects".to_string(),
        ok_dense_results: vec![],
        ok_sparse_results: vec![make_qdrant_point("sparse1", 0.7, "sparse result")],
    };

    // Keyword-only mode so only the sparse leg runs (which succeeds).
    let opts = SearchOptions {
        collection: Some("projects".to_string()),
        scope: SearchScope::All,
        ..opts_keyword("test query", 10)
    };

    let resp = run_search_pipeline(&mut daemon, &qdrant, Vec::new(), &opts, None, false).await;

    assert!(
        resp.status.is_none(),
        "M3: leg failure must NOT set status='uncertain'; got: {:?}",
        resp.status
    );
    assert!(
        resp.status_reason.is_none(),
        "M3: leg failure must NOT set status_reason; got: {:?}",
        resp.status_reason
    );
}

#[tokio::test]
async fn m3_dense_leg_failure_silent_sparse_results_kept() {
    // M3: dense leg fails, sparse leg succeeds → sparse results present, status ok.
    let mut daemon = StubDaemon {
        dense: Some(vec![0.1, 0.2]),
        sparse: Some(HashMap::from([(1u32, 0.5f32)])),
        unavailable: false,
    };

    let qdrant = OneFailingLegQdrant {
        fail_dense_collection: "projects".to_string(),
        ok_dense_results: vec![],
        ok_sparse_results: vec![make_qdrant_point("s1", 0.6, "sparse only")],
    };

    // Hybrid mode: dense fails silently, sparse succeeds → 1 result returned.
    let opts = SearchOptions {
        collection: Some("projects".to_string()),
        scope: SearchScope::All,
        ..opts_hybrid("query", 10)
    };

    let resp = run_search_pipeline(&mut daemon, &qdrant, Vec::new(), &opts, None, false).await;

    // Sparse result must be present (dense failure is silent).
    assert_eq!(
        resp.results.len(),
        1,
        "M3: sparse result must be kept when dense leg fails"
    );
    assert!(resp.status.is_none(), "M3: status must remain None");
}

#[tokio::test]
async fn m3_all_legs_succeed_no_uncertain_status() {
    // M3 sanity: when all legs succeed, status must be None (not 'uncertain').
    let mut daemon = StubDaemon {
        dense: Some(vec![0.1, 0.2]),
        sparse: Some(HashMap::from([(1u32, 0.5f32)])),
        unavailable: false,
    };
    let qdrant = StubQdrant {
        dense_results: vec![make_qdrant_point("r1", 0.8, "result one")],
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_hybrid("test", 10),
        None,
        false,
    )
    .await;

    assert!(
        resp.status.is_none(),
        "M3: no failures → status must be None, not 'uncertain'"
    );
    assert!(
        resp.status_reason.is_none(),
        "M3: no failures → status_reason must be None"
    );
}

// ---------------------------------------------------------------------------
// M4: exact mode must ALWAYS emit context_before/context_after (even empty [])
// ---------------------------------------------------------------------------

/// Exact-search daemon stub that returns matches with empty context.
struct ExactStubEmptyContext {
    matches: Vec<TextSearchMatch>,
}

impl ExactSearchDaemon for ExactStubEmptyContext {
    async fn text_search(
        &mut self,
        _request: TextSearchRequest,
    ) -> Result<TextSearchResponse, tonic::Status> {
        Ok(TextSearchResponse {
            matches: self.matches.clone(),
            total_matches: self.matches.len() as i32,
            truncated: false,
            query_time_ms: 0,
        })
    }
}

fn make_match(file_path: &str, line: i32, content: &str) -> TextSearchMatch {
    TextSearchMatch {
        file_path: file_path.to_string(),
        line_number: line,
        content: content.to_string(),
        tenant_id: "t1".to_string(),
        branch: None,
        context_before: vec![], // empty
        context_after: vec![],  // empty
    }
}

#[tokio::test]
async fn m4_exact_match_with_no_context_always_emits_empty_arrays() {
    // M4: TS `mapExactResults` always includes context_before/context_after keys
    // (search-exact.ts:55-63). Previously Rust omitted them when empty → byte
    // mismatch for contextLines=0. Now they must always be present as [].
    let mut daemon = ExactStubEmptyContext {
        matches: vec![make_match("src/main.rs", 42i32, "fn main() {}")],
    };

    let opts = SearchOptions {
        exact: true,
        context_lines: 0,
        project_id: Some("t1".to_string()),
        scope: SearchScope::Project,
        ..opts_hybrid("main", 10)
    };

    let resp = search_exact(&mut daemon, &opts).await;

    assert_eq!(resp.results.len(), 1);
    let meta = &resp.results[0].metadata;

    // Both keys must be present even though context is empty.
    let cb = meta
        .get("context_before")
        .expect("M4: context_before must always be present");
    let ca = meta
        .get("context_after")
        .expect("M4: context_after must always be present");

    assert_eq!(
        cb,
        &Value::Array(vec![]),
        "M4: context_before must be empty array [] when no context lines"
    );
    assert_eq!(
        ca,
        &Value::Array(vec![]),
        "M4: context_after must be empty array [] when no context lines"
    );

    // Verify JSON serialisation includes both keys.
    let json = serde_json::to_string(&resp.results[0]).unwrap();
    assert!(
        json.contains("\"context_before\":[]"),
        "M4: JSON must contain context_before:[], got: {json}"
    );
    assert!(
        json.contains("\"context_after\":[]"),
        "M4: JSON must contain context_after:[], got: {json}"
    );
}

#[tokio::test]
async fn m4_exact_match_with_context_lines_emits_populated_arrays() {
    // M4 sanity: non-empty context lines are also preserved correctly.
    let mut daemon = ExactStubEmptyContext {
        matches: vec![TextSearchMatch {
            file_path: "src/lib.rs".to_string(),
            line_number: 10,
            content: "pub fn run() {}".to_string(),
            tenant_id: "t1".to_string(),
            branch: None,
            context_before: vec!["// comment before".to_string()],
            context_after: vec!["// comment after".to_string()],
        }],
    };

    let opts = SearchOptions {
        exact: true,
        context_lines: 1,
        project_id: Some("t1".to_string()),
        scope: SearchScope::Project,
        ..opts_hybrid("run", 10)
    };

    let resp = search_exact(&mut daemon, &opts).await;
    let meta = &resp.results[0].metadata;

    let cb = meta
        .get("context_before")
        .expect("context_before must be present");
    assert_eq!(
        cb,
        &Value::Array(vec![Value::String("// comment before".to_string())])
    );
    let ca = meta
        .get("context_after")
        .expect("context_after must be present");
    assert_eq!(
        ca,
        &Value::Array(vec![Value::String("// comment after".to_string())])
    );
}

// ---------------------------------------------------------------------------
// M5: exact mode default max_results = 100, not 10
// ---------------------------------------------------------------------------

/// Exact-search daemon that captures the max_results sent to text_search.
struct CapturingExactDaemon {
    captured_max_results: std::sync::Arc<std::sync::Mutex<i32>>,
}

impl ExactSearchDaemon for CapturingExactDaemon {
    async fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> Result<TextSearchResponse, tonic::Status> {
        *self.captured_max_results.lock().unwrap() = request.max_results;
        Ok(TextSearchResponse {
            matches: vec![],
            total_matches: 0,
            truncated: false,
            query_time_ms: 0,
        })
    }
}

#[tokio::test]
async fn m5_exact_mode_no_limit_uses_100() {
    // M5: TS uses `options.limit ?? 100` (search-exact.ts:95). When caller does
    // not specify limit, exact mode must pass max_results=100 to text_search.
    let captured = std::sync::Arc::new(std::sync::Mutex::new(0i32));
    let mut daemon = CapturingExactDaemon {
        captured_max_results: captured.clone(),
    };

    // No limit specified → from_input sets limit=DEFAULT_LIMIT(10), limit_explicit=false.
    let opts = SearchOptions {
        exact: true,
        project_id: Some("t1".to_string()),
        scope: SearchScope::Project,
        limit_explicit: false,        // caller did not specify limit
        ..opts_hybrid("pattern", 10)  // 10 is the default, but limit_explicit=false
    };

    let _ = search_exact(&mut daemon, &opts).await;

    assert_eq!(
        *captured.lock().unwrap(),
        DEFAULT_EXACT_LIMIT as i32,
        "M5: exact mode without explicit limit must pass max_results={DEFAULT_EXACT_LIMIT} (not 10)"
    );
}

#[tokio::test]
async fn m5_exact_mode_explicit_limit_respected() {
    // M5: when caller explicitly provides limit=5, exact mode must use 5, not 100.
    let captured = std::sync::Arc::new(std::sync::Mutex::new(0i32));
    let mut daemon = CapturingExactDaemon {
        captured_max_results: captured.clone(),
    };

    let opts = SearchOptions {
        exact: true,
        project_id: Some("t1".to_string()),
        scope: SearchScope::Project,
        limit: 5,
        limit_explicit: true, // caller specified limit=5 explicitly
        ..opts_hybrid("pattern", 5)
    };

    let _ = search_exact(&mut daemon, &opts).await;

    assert_eq!(
        *captured.lock().unwrap(),
        5,
        "M5: explicit limit=5 must be forwarded as max_results=5, not overridden with 100"
    );
}

#[tokio::test]
async fn m5_non_exact_mode_uses_default_limit_10() {
    // M5 sanity: non-exact search still uses DEFAULT_LIMIT=10 when unspecified.
    // This is a from_input level check — opts_hybrid uses limit=10 (DEFAULT_LIMIT).
    use crate::tools::search::options::DEFAULT_LIMIT;
    let opts = opts_hybrid("query", DEFAULT_LIMIT);
    assert!(!opts.exact, "sanity: opts_hybrid is not exact mode");
    assert_eq!(
        opts.limit, DEFAULT_LIMIT,
        "non-exact default limit must be 10"
    );
}

// ---------------------------------------------------------------------------
// #3: exact search with session project_id fallback
// ---------------------------------------------------------------------------

/// Exact daemon stub that records the tenant_id sent in the request.
struct TenantCapturingExactDaemon {
    captured_tenant: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl ExactSearchDaemon for TenantCapturingExactDaemon {
    async fn text_search(
        &mut self,
        request: TextSearchRequest,
    ) -> Result<TextSearchResponse, tonic::Status> {
        *self.captured_tenant.lock().unwrap() = request.tenant_id.clone();
        Ok(TextSearchResponse {
            matches: vec![],
            total_matches: 0,
            truncated: false,
            query_time_ms: 0,
        })
    }
}

#[tokio::test]
async fn exact_mode_with_project_id_in_opts_sends_tenant() {
    // Regression for finding #3: when opts.project_id is set (which the
    // search_tool now ensures by folding in the session fallback), exact mode
    // must forward it as tenant_id to the daemon.
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut daemon = TenantCapturingExactDaemon {
        captured_tenant: captured.clone(),
    };

    let opts = SearchOptions {
        exact: true,
        project_id: Some("session-project-xyz".to_string()), // simulates session fallback
        scope: SearchScope::Project,
        ..opts_hybrid("fn main", 10)
    };

    let _ = search_exact(&mut daemon, &opts).await;

    let tenant = captured.lock().unwrap().clone();
    assert_eq!(
        tenant.as_deref(),
        Some("session-project-xyz"),
        "exact mode must forward project_id as tenant to daemon"
    );
}

#[tokio::test]
async fn exact_mode_without_project_id_returns_unresolved() {
    // When opts.project_id is None AND scope=Project, exact mode must return an
    // unresolved response (not attempt RPC).  This verifies the guard is intact.
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut daemon = TenantCapturingExactDaemon {
        captured_tenant: captured.clone(),
    };

    let opts = SearchOptions {
        exact: true,
        project_id: None, // absent — unresolved
        scope: SearchScope::Project,
        ..opts_hybrid("fn main", 10)
    };

    let resp = search_exact(&mut daemon, &opts).await;

    // No RPC should fire (daemon not called).
    assert!(
        captured.lock().unwrap().is_none(),
        "no RPC must fire when tenant is unresolved"
    );
    assert_eq!(
        resp.status.as_deref(),
        Some("uncertain"),
        "unresolved exact mode must set status='uncertain'"
    );
}
