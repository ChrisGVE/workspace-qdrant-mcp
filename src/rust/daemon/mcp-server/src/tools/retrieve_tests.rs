//! Unit tests for the `retrieve` tool.  All hermetic — no live Qdrant.

use std::collections::HashMap;

use serde_json::Value;

use super::{retrieve_tool, RetrieveFilter, RetrieveInput, RetrieveQdrant, RetrieveResponse};
use crate::qdrant::client::QdrantRetrievedPoint;

// ---------------------------------------------------------------------------
// Stub Qdrant
// ---------------------------------------------------------------------------

#[derive(Default)]
struct StubQdrant {
    /// Points returned by retrieve_by_ids.
    points: Vec<QdrantRetrievedPoint>,
    /// Points returned by scroll (per call — supports limit+1 over-fetch).
    scroll_points: Vec<QdrantRetrievedPoint>,
    /// If set, both methods return this error string.
    error: Option<String>,
}

impl RetrieveQdrant for StubQdrant {
    async fn retrieve_by_ids(
        &self,
        _collection: &str,
        _ids: Vec<String>,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        if let Some(ref e) = self.error {
            return Err(e.clone());
        }
        Ok(self.points.clone())
    }

    async fn scroll(
        &self,
        _collection: &str,
        _filter: Option<RetrieveFilter>,
        _limit: u32,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        if let Some(ref e) = self.error {
            return Err(e.clone());
        }
        Ok(self.scroll_points.clone())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_point(id: &str, content: &str, tenant_id: &str) -> QdrantRetrievedPoint {
    let mut payload = HashMap::new();
    payload.insert("content".to_string(), Value::String(content.to_string()));
    payload.insert(
        "tenant_id".to_string(),
        Value::String(tenant_id.to_string()),
    );
    payload.insert(
        "source_file".to_string(),
        Value::String("src/main.rs".to_string()),
    );
    QdrantRetrievedPoint {
        id: id.to_string(),
        payload,
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

fn parse_response(r: &rmcp::model::CallToolResult) -> RetrieveResponse {
    serde_json::from_str(result_text(r)).expect("valid JSON")
}

// ---------------------------------------------------------------------------
// by-id: scope guard (F-002)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn by_id_projects_no_project_id_returns_unresolved() {
    let qdrant = StubQdrant::default();
    let input = RetrieveInput {
        document_id: Some("doc-1".to_string()),
        collection: "projects".to_string(),
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
        .contains("without a resolvable scope"));
}

#[tokio::test]
async fn by_id_libraries_no_library_name_returns_unresolved() {
    let qdrant = StubQdrant::default();
    let input = RetrieveInput {
        document_id: Some("doc-1".to_string()),
        collection: "libraries".to_string(),
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
        .contains("without a resolvable scope"));
}

#[tokio::test]
async fn by_id_rules_no_scope_succeeds() {
    // rules = mixed-tenancy; no scope required
    let point = make_point("doc-1", "rule content", "ignored");
    let qdrant = StubQdrant {
        points: vec![point],
        ..Default::default()
    };
    let input = RetrieveInput {
        document_id: Some("doc-1".to_string()),
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert_eq!(resp.documents.len(), 1);
}

// ---------------------------------------------------------------------------
// by-id: happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn by_id_returns_single_document() {
    let point = make_point("abc-123", "hello world", "proj-x");
    let qdrant = StubQdrant {
        points: vec![point],
        ..Default::default()
    };
    let input = RetrieveInput {
        document_id: Some("abc-123".to_string()),
        collection: "projects".to_string(),
        project_id: Some("proj-x".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert_eq!(resp.documents.len(), 1);
    assert_eq!(resp.documents[0].id, "abc-123");
    assert_eq!(resp.documents[0].content, "hello world");
    assert_eq!(resp.total, Some(1));
    assert!(!resp.has_more);
}

#[tokio::test]
async fn by_id_metadata_excludes_content_and_vectors() {
    let point = make_point("doc-1", "body", "proj-a");
    // point.payload already has content + source_file; no vector keys
    let qdrant = StubQdrant {
        points: vec![point],
        ..Default::default()
    };
    let input = RetrieveInput {
        document_id: Some("doc-1".to_string()),
        collection: "projects".to_string(),
        project_id: Some("proj-a".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    let meta = resp.documents[0].metadata.as_object().unwrap();
    // 'content' must be excluded
    assert!(
        !meta.contains_key("content"),
        "content must be excluded from metadata"
    );
    // source_file must be present
    assert!(meta.contains_key("source_file"));
}

// ---------------------------------------------------------------------------
// by-id: ownership check (F-002)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn by_id_wrong_tenant_returns_not_found() {
    // Point has tenant_id="other-tenant" but caller scopes to "proj-x"
    let point = make_point("doc-1", "secret", "other-tenant");
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
    let resp = parse_response(&r);
    // Must NOT expose the foreign document
    assert!(!resp.success);
    assert!(resp
        .message
        .as_deref()
        .unwrap_or("")
        .contains("Document not found"));
}

#[tokio::test]
async fn by_id_not_found_returns_error() {
    let qdrant = StubQdrant::default(); // no points
    let input = RetrieveInput {
        document_id: Some("missing".to_string()),
        collection: "projects".to_string(),
        project_id: Some("proj-x".to_string()),
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
        .contains("Document not found: missing"));
}

// ---------------------------------------------------------------------------
// by-filter: scope guard (F-011)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn by_filter_projects_no_project_id_returns_unresolved() {
    let qdrant = StubQdrant::default();
    let input = RetrieveInput {
        collection: "projects".to_string(),
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
        .contains("without a resolvable scope"));
}

#[tokio::test]
async fn by_filter_libraries_no_library_name_returns_unresolved() {
    let qdrant = StubQdrant::default();
    let input = RetrieveInput {
        collection: "libraries".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(!resp.success);
}

#[tokio::test]
async fn by_filter_rules_no_scope_succeeds() {
    let qdrant = StubQdrant::default(); // empty scroll
    let input = RetrieveInput {
        collection: "rules".to_string(),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert!(resp.documents.is_empty());
}

// ---------------------------------------------------------------------------
// by-filter: pagination / hasMore
// ---------------------------------------------------------------------------

#[tokio::test]
async fn by_filter_has_more_when_limit_plus_1_returned() {
    // limit=2; scroll returns 3 points (limit+1) → hasMore=true, truncate to 2
    let points: Vec<QdrantRetrievedPoint> = (0..3)
        .map(|i| make_point(&format!("id-{i}"), "body", "proj-a"))
        .collect();
    let qdrant = StubQdrant {
        scroll_points: points,
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "projects".to_string(),
        project_id: Some("proj-a".to_string()),
        limit: 2,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert!(resp.has_more);
    assert_eq!(resp.documents.len(), 2);
}

#[tokio::test]
async fn by_filter_no_has_more_when_exact_limit_returned() {
    let points: Vec<QdrantRetrievedPoint> = (0..2)
        .map(|i| make_point(&format!("id-{i}"), "body", "proj-a"))
        .collect();
    let qdrant = StubQdrant {
        scroll_points: points,
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "projects".to_string(),
        project_id: Some("proj-a".to_string()),
        limit: 2,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(!resp.has_more);
    assert_eq!(resp.documents.len(), 2);
}

#[tokio::test]
async fn by_filter_fewer_than_limit_no_has_more() {
    let points = vec![make_point("id-0", "body", "proj-a")];
    let qdrant = StubQdrant {
        scroll_points: points,
        ..Default::default()
    };
    let input = RetrieveInput {
        collection: "projects".to_string(),
        project_id: Some("proj-a".to_string()),
        limit: 10,
        ..Default::default()
    };
    let r = retrieve_tool(input, &qdrant).await;
    let resp = parse_response(&r);
    assert!(!resp.has_more);
    assert_eq!(resp.documents.len(), 1);
}

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
