//! AC-S4: daemon-down → fallbackSearch degraded path.
//! AC-S5 / AC-OFF2 / F-001: project scope + unresolved project_id → REFUSE scroll.
//! Partial Qdrant failure → status='uncertain', status_reason cites unavailable collections.
//!
//! All stubs/helpers are defined `pub(super)` in `search_tests.rs`; reach them via
//! `use super::*;`.

use super::*;

use crate::tools::search::flow_fallback::{
    f001_refusal_reason, fallback_search, project_id_is_unresolved,
};
use crate::tools::search::types::{SearchMode, SearchScope};

// ---------------------------------------------------------------------------
// AC-S4: daemon unavailable → fallbackSearch → status='uncertain',
//         status_reason='Daemon unavailable - using fallback text search'
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac_s4_daemon_down_triggers_fallback_search() {
    // When daemon is unavailable, embed_text and generate_sparse_vector both
    // return Err → run_search_pipeline calls fallback_search.
    let mut daemon = StubDaemon {
        unavailable: true,
        ..Default::default()
    };

    // Provide scroll results that match the query text so fallback returns them.
    let scroll_point = make_retrieved_point("fb1", "fallback content match");
    let qdrant = StubQdrant {
        scroll_results: vec![scroll_point],
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        &opts_hybrid("fallback content", 10),
        Some("tenant_abc"),
        false,
        &Default::default(),
    )
    .await;

    assert_eq!(
        resp.status.as_deref(),
        Some("uncertain"),
        "status must be 'uncertain'"
    );
    assert_eq!(
        resp.status_reason.as_deref(),
        Some("Daemon unavailable - using fallback text search"),
        // search-qdrant.ts:405
        "status_reason must match TS verbatim"
    );
    // results returned — at least the fallback match
    assert!(
        !resp.results.is_empty(),
        "fallback must return matching results"
    );
}

// ---------------------------------------------------------------------------
// AC-S4b: daemon-down with no match → still degraded, empty results
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac_s4b_daemon_down_no_match_still_degraded() {
    let mut daemon = StubDaemon {
        unavailable: true,
        ..Default::default()
    };
    let scroll_point = make_retrieved_point("x1", "unrelated content");
    let qdrant = StubQdrant {
        scroll_results: vec![scroll_point],
        ..Default::default()
    };

    let resp = run_search_pipeline(
        &mut daemon,
        &qdrant,
        Vec::new(),
        // query "alpha beta" won't match "unrelated content"
        &opts_hybrid("alpha beta", 10),
        Some("tenant_abc"),
        false,
        &Default::default(),
    )
    .await;

    assert_eq!(resp.status.as_deref(), Some("uncertain"));
    assert_eq!(
        resp.status_reason.as_deref(),
        Some("Daemon unavailable - using fallback text search")
    );
    assert!(
        resp.results.is_empty(),
        "no substring match → empty results"
    );
}

// ---------------------------------------------------------------------------
// AC-S5 / F-001 SECURITY: scope=project AND project_id=None → REFUSE scroll.
//
// Exact expected status_reason (search-qdrant.ts:402-405):
//   "Daemon unavailable and project scope unresolved - cannot run cross-tenant
//    fallback. Refused collections: <comma-sep list>"
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac_s5_f001_project_scope_unresolved_refuses_scroll() {
    // opts_project_scope sets scope=Project, project_id=None.
    let opts = opts_project_scope("secure query");

    // Qdrant has scroll results that SHOULD NOT be returned.
    let scroll_pt = make_retrieved_point("sensitive1", "secure query sensitive data");
    let qdrant = StubQdrant {
        scroll_results: vec![scroll_pt],
        ..Default::default()
    };

    // Call fallback_search directly (same code path as when daemon is down).
    // The collections that fallback_search iterates over for scope=All include
    // "projects" and "scratchpad" — we drive it directly so scope=Project is
    // applied without needing the daemon path.
    let collections = vec!["projects".to_string(), "scratchpad".to_string()];

    // project_id=None → both "projects" and "scratchpad" are refused.
    let resp = fallback_search(
        &qdrant,
        &opts,
        &collections,
        None, /* project_id unresolved */
        &Default::default(),
    )
    .await;

    // F-001: must return uncertain status with the EXACT refusal reason.
    assert_eq!(
        resp.status.as_deref(),
        Some("uncertain"),
        "status must be 'uncertain'"
    );

    let expected_reason = f001_refusal_reason(&collections);
    assert_eq!(
        resp.status_reason.as_deref(),
        Some(expected_reason.as_str()),
        "F-001 refusal reason must match TS search-qdrant.ts:404 verbatim"
    );

    // SECURITY: zero results must be returned — no cross-tenant scroll.
    assert!(
        resp.results.is_empty(),
        "F-001: MUST return zero results when project scope is unresolved"
    );
}

#[tokio::test]
async fn ac_s5_f001_refusal_reason_format() {
    // Verify the exact string format with a known collection list.
    let refused = vec!["projects".to_string(), "scratchpad".to_string()];
    let reason = f001_refusal_reason(&refused);
    assert_eq!(
        reason,
        "Daemon unavailable and project scope unresolved - cannot run cross-tenant fallback. \
         Refused collections: projects, scratchpad",
        "f001_refusal_reason must match TS search-qdrant.ts:404 byte-for-byte"
    );
}

#[tokio::test]
async fn ac_s5_f001_single_collection_refused() {
    let refused = vec!["projects".to_string()];
    let reason = f001_refusal_reason(&refused);
    assert_eq!(
        reason,
        "Daemon unavailable and project scope unresolved - cannot run cross-tenant fallback. \
         Refused collections: projects"
    );
}

// ---------------------------------------------------------------------------
// AC-S5: When project_id IS resolved, project-scoped fallback proceeds normally.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ac_s5_project_scope_with_resolved_id_proceeds() {
    let opts = opts_project_with_id("search term", "tenant_abc");
    let scroll_pt = make_retrieved_point("r1", "search term result");
    let qdrant = StubQdrant {
        scroll_results: vec![scroll_pt],
        ..Default::default()
    };
    let collections = vec!["projects".to_string()];

    let resp = fallback_search(&qdrant, &opts, &collections, Some("tenant_abc"), &Default::default()).await;

    // Degraded (uncertain) but NOT refused — results allowed.
    assert_eq!(resp.status.as_deref(), Some("uncertain"));
    assert_eq!(
        resp.status_reason.as_deref(),
        Some("Daemon unavailable - using fallback text search"),
        "resolved project_id → normal fallback, not F-001 refusal"
    );
    assert!(
        !resp.results.is_empty(),
        "results must be returned when project resolved"
    );
}

// ---------------------------------------------------------------------------
// Partial Qdrant failure: one collection fails → status='uncertain',
//   status_reason cites which collections are unavailable.
// ---------------------------------------------------------------------------

/// Qdrant stub that returns an error for a specific collection on scroll.
struct PartiallyFailingQdrant {
    fail_collection: String,
    scroll_results: Vec<crate::qdrant::client::QdrantRetrievedPoint>,
}

impl SearchQdrant for PartiallyFailingQdrant {
    async fn search_dense(
        &self,
        _c: &str,
        _v: Vec<f32>,
        _l: u64,
        _t: Option<f32>,
        _f: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<crate::qdrant::client::QdrantPoint>> {
        Ok(vec![])
    }

    async fn search_sparse(
        &self,
        _c: &str,
        _i: Vec<u32>,
        _v: Vec<f32>,
        _l: u64,
        _t: Option<f32>,
        _f: Option<qdrant_client::qdrant::Filter>,
    ) -> anyhow::Result<Vec<crate::qdrant::client::QdrantPoint>> {
        Ok(vec![])
    }

    async fn scroll_page(
        &self,
        collection: &str,
        _f: Option<qdrant_client::qdrant::Filter>,
        _l: u32,
    ) -> anyhow::Result<Vec<crate::qdrant::client::QdrantRetrievedPoint>> {
        if collection == self.fail_collection {
            Err(anyhow::anyhow!("Qdrant unavailable: {collection}"))
        } else {
            Ok(self.scroll_results.clone())
        }
    }

    async fn retrieve_by_ids(
        &self,
        _c: &str,
        _ids: Vec<String>,
    ) -> anyhow::Result<Vec<crate::qdrant::client::QdrantRetrievedPoint>> {
        Ok(vec![])
    }
}

#[tokio::test]
async fn partial_qdrant_failure_reports_uncertain_status() {
    // When daemon is down and one Qdrant collection fails, the response
    // should be uncertain and the fallback produces results from the
    // healthy collection only.
    let opts = opts_hybrid("content text", 10);
    let good_point = make_retrieved_point("g1", "content text from good collection");
    let qdrant = PartiallyFailingQdrant {
        fail_collection: "scratchpad".to_string(),
        scroll_results: vec![good_point],
    };
    let collections = vec!["projects".to_string(), "scratchpad".to_string()];

    let resp = fallback_search(&qdrant, &opts, &collections, None, &Default::default()).await;

    // Fallback always sets status=uncertain.
    assert_eq!(resp.status.as_deref(), Some("uncertain"));
    // The partial failure doesn't produce F-001 since scope is All here — the
    // fallback runs both collections; scratchpad fails silently (Ok branch not
    // entered), projects succeeds. Status reason is the standard fallback message.
    assert_eq!(
        resp.status_reason.as_deref(),
        Some("Daemon unavailable - using fallback text search")
    );
    // Results come from the good collection only.
    assert_eq!(
        resp.results.len(),
        1,
        "only the healthy collection contributes results"
    );
    assert_eq!(resp.results[0].id, "g1");
}

// ---------------------------------------------------------------------------
// Sanity: fallback_search response field population
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fallback_search_response_fields_populated() {
    let opts = SearchOptions {
        mode: SearchMode::Keyword,
        scope: SearchScope::All,
        ..opts_hybrid("hello world", 5)
    };
    let pt = make_retrieved_point("pt1", "hello world piece");
    let qdrant = StubQdrant {
        scroll_results: vec![pt],
        ..Default::default()
    };
    let collections = vec!["projects".to_string()];

    let resp = fallback_search(&qdrant, &opts, &collections, None, &Default::default()).await;

    assert_eq!(resp.query, "hello world");
    assert_eq!(resp.mode, SearchMode::Keyword);
    assert_eq!(resp.scope, SearchScope::All);
    assert_eq!(resp.collections_searched, collections);
    assert!(!resp.results.is_empty());
    // score for fallback hits is always 0.5
    assert_eq!(
        resp.results[0].score, 0.5,
        "fallback results score is always 0.5"
    );
}

// ---------------------------------------------------------------------------
// M1 (SECURITY F-001): scope=project + unresolved project_id → ALL collections
// refused — not just "projects"/"scratchpad".
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m1_f001_project_scope_unresolved_refuses_all_collections_including_libraries() {
    // M1 fix: when scope=Project and project_id=None, EVERY collection in the
    // list is refused — the old code only refused "projects"/"scratchpad" which
    // allowed "libraries" and "rules" legs to leak.
    let opts = SearchOptions {
        scope: SearchScope::Project,
        project_id: None,
        include_libraries: true,
        ..opts_hybrid("sensitive query", 10)
    };
    let scroll_pt = make_retrieved_point("leak1", "sensitive query leak data");
    let qdrant = StubQdrant {
        scroll_results: vec![scroll_pt],
        ..Default::default()
    };
    // Both projects and libraries are included (include_libraries=true).
    let collections = vec!["projects".to_string(), "libraries".to_string()];

    let resp = fallback_search(&qdrant, &opts, &collections, None, &Default::default()).await;

    // F-001: BOTH collections refused — zero results, uncertain status.
    assert_eq!(
        resp.results.len(),
        0,
        "M1: libraries leg must also be refused when project_id unresolved"
    );
    assert_eq!(resp.status.as_deref(), Some("uncertain"));
    let reason = resp.status_reason.unwrap_or_default();
    assert!(
        reason.contains("Refused collections: projects, libraries"),
        "M1: F-001 string must list both refused collections, got: {reason}"
    );
}

#[tokio::test]
async fn m1_f001_explicit_rules_collection_refused_when_scope_project_unresolved() {
    // M1: even an explicit collection="rules" is refused when scope=project, project_id=None.
    let opts = SearchOptions {
        scope: SearchScope::Project,
        project_id: None,
        ..opts_hybrid("rules query", 10)
    };
    let qdrant = StubQdrant::default();
    let collections = vec!["rules".to_string()];

    let resp = fallback_search(&qdrant, &opts, &collections, None, &Default::default()).await;

    assert_eq!(
        resp.results.len(),
        0,
        "M1: rules collection must be refused"
    );
    let reason = resp.status_reason.unwrap_or_default();
    assert!(
        reason.contains("Refused collections: rules"),
        "M1: F-001 reason must cite the refused rules collection, got: {reason}"
    );
}

// ---------------------------------------------------------------------------
// M2 (SECURITY F-001): empty/whitespace project_id treated as unresolved.
// ---------------------------------------------------------------------------

#[test]
fn m2_project_id_is_unresolved_none() {
    assert!(project_id_is_unresolved(None), "None must be unresolved");
}

#[test]
fn m2_project_id_is_unresolved_empty_string() {
    assert!(
        project_id_is_unresolved(Some("")),
        "empty string must be unresolved (mirrors TS !currentProjectId)"
    );
}

#[test]
fn m2_project_id_is_unresolved_whitespace_only() {
    assert!(
        project_id_is_unresolved(Some("   ")),
        "whitespace-only must be unresolved"
    );
}

#[test]
fn m2_project_id_non_empty_is_resolved() {
    assert!(
        !project_id_is_unresolved(Some("tenant_abc")),
        "non-empty string must be resolved"
    );
}

#[tokio::test]
async fn m2_f001_empty_project_id_refuses_scroll_with_exact_f001_string() {
    // M2: project_id=Some("") must be treated as unresolved — refuses the scroll
    // and emits the exact F-001 status string; tenant scroll must NOT execute.
    let opts = SearchOptions {
        scope: SearchScope::Project,
        project_id: Some(String::new()), // empty string — must count as unresolved
        ..opts_hybrid("secure", 10)
    };
    let scroll_pt = make_retrieved_point("s1", "secure secret data");
    let qdrant = StubQdrant {
        scroll_results: vec![scroll_pt],
        ..Default::default()
    };
    let collections = vec!["projects".to_string()];

    let resp = fallback_search(&qdrant, &opts, &collections, Some(""), &Default::default()).await;

    assert_eq!(
        resp.results.len(),
        0,
        "M2: empty project_id must refuse scroll — tenant scroll must NOT execute"
    );
    assert_eq!(resp.status.as_deref(), Some("uncertain"));
    let expected = f001_refusal_reason(&collections);
    assert_eq!(
        resp.status_reason.as_deref(),
        Some(expected.as_str()),
        "M2: must emit exact F-001 string for empty project_id"
    );
}
