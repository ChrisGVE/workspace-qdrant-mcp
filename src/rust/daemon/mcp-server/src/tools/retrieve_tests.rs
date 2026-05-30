//! Unit tests for the `retrieve` tool.  All hermetic — no live Qdrant.

use std::collections::HashMap;

use serde_json::Value;

use super::{retrieve_tool, RetrieveFilter, RetrieveInput, RetrieveQdrant, RetrieveResponse};
use crate::qdrant::client::QdrantRetrievedPoint;

// ---------------------------------------------------------------------------
// Stub Qdrant
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(super) struct StubQdrant {
    /// Points returned by retrieve_by_ids.
    pub points: Vec<QdrantRetrievedPoint>,
    /// Points returned by scroll (per call — supports limit+1 over-fetch).
    pub scroll_points: Vec<QdrantRetrievedPoint>,
    /// If set, both methods return this error string.
    pub error: Option<String>,
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
        _offset: u32,
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

pub(super) fn make_point(id: &str, content: &str, tenant_id: &str) -> QdrantRetrievedPoint {
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

pub(super) fn result_text(r: &rmcp::model::CallToolResult) -> &str {
    r.content
        .first()
        .unwrap()
        .raw
        .as_text()
        .unwrap()
        .text
        .as_str()
}

pub(super) fn parse_response(r: &rmcp::model::CallToolResult) -> RetrieveResponse {
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert_eq!(resp.documents.len(), 1);
    assert_eq!(resp.documents[0].id, "abc-123");
    assert_eq!(resp.documents[0].content, "hello world");
    assert_eq!(resp.total, Some(1));
    assert_eq!(resp.has_more, Some(false));
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
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
    let r = retrieve_tool(input, &qdrant, None).await;
    let resp = parse_response(&r);
    assert!(resp.success);
    assert_eq!(resp.has_more, Some(true));
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
    let r = retrieve_tool(input, &qdrant, None).await;
    let resp = parse_response(&r);
    assert_eq!(resp.has_more, Some(false));
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
    let r = retrieve_tool(input, &qdrant, None).await;
    let resp = parse_response(&r);
    assert_eq!(resp.has_more, Some(false));
    assert_eq!(resp.documents.len(), 1);
}

// ---------------------------------------------------------------------------
// Collection-not-found normalisation and further tests — split into sibling
// ---------------------------------------------------------------------------

#[path = "retrieve_tests_part2.rs"]
mod part2;
