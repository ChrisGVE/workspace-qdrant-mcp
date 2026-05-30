//! Retrieve tool tests part 2: collection-not-found normalisation, daemon
//! independence, RetrieveInput::from_args, and JSON field-order / hasMore
//! parity tests.
//!
//! Included from `retrieve_tests.rs` via
//! `#[path = "retrieve_tests_part2.rs"] mod part2;`.

use super::{make_point, parse_response, result_text, StubQdrant};
use super::{retrieve_tool, RetrieveInput};

// ---------------------------------------------------------------------------
// Collection-not-found normalisation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn collection_not_found_normalised_to_success_empty() {
    let qdrant = StubQdrant {
        error: Some("collection not found".to_string()),
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(
        resp.success,
        "collection-not-found must normalise to success"
    );
    assert!(resp.documents.is_empty());
    assert_eq!(
        resp.message.as_deref(),
        Some("Collection not found or empty")
    );
}

#[tokio::test]
async fn collection_doesnt_exist_normalised_to_success_empty() {
    let qdrant = StubQdrant {
        error: Some("collection doesn't exist".to_string()),
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
}

#[tokio::test]
async fn qdrant_down_returns_failure() {
    let qdrant = StubQdrant {
        error: Some("connection refused".to_string()),
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(!resp.success);
    assert!(resp
        .message
        .as_deref()
        .unwrap_or("")
        .contains("Failed to retrieve documents"));
}

// ---------------------------------------------------------------------------
// Daemon-independence: tool works fine even if daemon is down
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retrieve_is_daemon_independent() {
    // This test verifies there's no DaemonClient dependency at all.
    // Passing a StubQdrant without any DaemonClient proves the contract.
    let qdrant = StubQdrant::default();
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 5,
        ..Default::default()
    };
    // Should complete without touching any gRPC endpoint
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
}

// ---------------------------------------------------------------------------
// RetrieveInput::from_args
// ---------------------------------------------------------------------------

#[test]
fn from_args_defaults() {
    let args = serde_json::json!({});
    let map = args.as_object().unwrap();
    let input = RetrieveInput::from_args(map);
    assert!(input.document_id.is_none());
    assert_eq!(input.collection, "projects");
    assert_eq!(input.limit, 10);
    assert_eq!(input.offset, 0);
    assert!(input.filter.is_none());
    assert!(input.project_id.is_none());
    assert!(input.library_name.is_none());
}

#[test]
fn from_args_all_fields() {
    let args = serde_json::json!({
        "documentId": "doc-abc",
        "collection": "rules",
        "filter": { "tag": "important" },
        "limit": 20,
        "offset": 5,
        "projectId": "proj-123",
        "libraryName": "mylib"
    });
    let map = args.as_object().unwrap();
    let input = RetrieveInput::from_args(map);
    assert_eq!(input.document_id.as_deref(), Some("doc-abc"));
    assert_eq!(input.collection, "rules");
    assert_eq!(input.limit, 20);
    assert_eq!(input.offset, 5);
    assert_eq!(input.project_id.as_deref(), Some("proj-123"));
    assert_eq!(input.library_name.as_deref(), Some("mylib"));
    let f = input.filter.unwrap();
    assert_eq!(f.get("tag").map(|s| s.as_str()), Some("important"));
}

// ---------------------------------------------------------------------------
// Golden: JSON field order
// ---------------------------------------------------------------------------

#[tokio::test]
async fn golden_success_has_more_is_camel_case() {
    // Verify hasMore is serialised as camelCase (not has_more)
    let qdrant = StubQdrant::default();
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let text = result_text(&r);
    assert!(
        text.contains("\"hasMore\""),
        "field must be camelCase 'hasMore'"
    );
    assert!(!text.contains("\"has_more\""));
}

// ---------------------------------------------------------------------------
// hasMore presence/absence parity with TS (MUST-FIX 1)
//
// TS omits hasMore entirely for: by-id not-found (retrieve.ts:202, 208),
// by-id catch-error (retrieve.ts:219-223), by-filter non-collection-not-found
// error (retrieve.ts:273-277).
// TS includes hasMore for: by-id success (retrieve.ts:217), by-filter success
// (retrieve.ts:261), collection-not-found (retrieve.ts:265-271), and
// unresolvedTenantResponse (retrieve.ts:44-55).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn by_id_not_found_json_has_no_has_more_key() {
    // TS retrieve.ts:202 — not-found returns {success, documents, message} with NO hasMore.
    let qdrant = StubQdrant::default(); // no points
    let input = RetrieveInput {
        document_id: Some("missing-doc".to_string()),
        collection: "projects".to_string(),
        project_id: Some("proj-x".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let text = result_text(&r);
    assert!(
        !text.contains("\"hasMore\""),
        "by-id not-found must NOT include hasMore key; got: {text}"
    );
}

#[tokio::test]
async fn by_id_error_json_has_no_has_more_key() {
    // TS retrieve.ts:219-223 — catch error returns {success, documents, message} NO hasMore.
    let qdrant = StubQdrant {
        error: Some("connection refused".to_string()),
        ..Default::default()
    };
    let input = RetrieveInput {
        document_id: Some("doc-1".to_string()),
        collection: "projects".to_string(),
        project_id: Some("proj-x".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let text = result_text(&r);
    assert!(
        !text.contains("\"hasMore\""),
        "by-id error must NOT include hasMore key; got: {text}"
    );
}

#[tokio::test]
async fn by_id_success_json_has_has_more_key() {
    // TS retrieve.ts:217 — success includes hasMore: false.
    let point = make_point("doc-1", "content", "proj-x");
    let qdrant = StubQdrant {
        points: vec![point],
        ..Default::default()
    };
    let input = RetrieveInput {
        document_id: Some("doc-1".to_string()),
        collection: "projects".to_string(),
        project_id: Some("proj-x".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let text = result_text(&r);
    assert!(
        text.contains("\"hasMore\""),
        "by-id success MUST include hasMore key; got: {text}"
    );
}

#[tokio::test]
async fn by_filter_error_json_has_no_has_more_key() {
    // TS retrieve.ts:273-277 — non-collection-not-found error omits hasMore.
    let qdrant = StubQdrant {
        error: Some("connection refused".to_string()),
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let text = result_text(&r);
    assert!(
        !text.contains("\"hasMore\""),
        "by-filter error must NOT include hasMore key; got: {text}"
    );
}

#[tokio::test]
async fn by_filter_success_json_has_has_more_key() {
    // TS retrieve.ts:261 — success includes hasMore.
    let qdrant = StubQdrant::default(); // empty scroll = success with hasMore: false
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let text = result_text(&r);
    assert!(
        text.contains("\"hasMore\""),
        "by-filter success MUST include hasMore key; got: {text}"
    );
}

// ---------------------------------------------------------------------------
// Offset passthrough (MUST-FIX 2)
//
// TS retrieve.ts:247: `if (offset > 0) scrollRequest.offset = offset`.
// We verify the tool accepts a non-zero offset without error (the stub
// discards the offset value — live integration testing is outside scope here).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn by_filter_with_nonzero_offset_succeeds() {
    let points = vec![make_point("id-5", "body", "proj-a")];
    let qdrant = StubQdrant {
        scroll_points: points,
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "projects".to_string(),
        project_id: Some("proj-a".to_string()),
        limit: 10,
        offset: 5, // non-zero — mirrors TS `if (offset > 0) scrollRequest.offset = offset`
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert_eq!(resp.documents.len(), 1);
}

#[tokio::test]
async fn by_filter_with_zero_offset_succeeds() {
    // offset=0 must behave identically to omitting offset (no cursor forwarded).
    let points = vec![make_point("id-0", "body", "proj-a")];
    let qdrant = StubQdrant {
        scroll_points: points,
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "projects".to_string(),
        project_id: Some("proj-a".to_string()),
        limit: 10,
        offset: 0,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert_eq!(resp.documents.len(), 1);
}
