//! `retrieve` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/retrieve.ts` +
//! `src/typescript/mcp-server/src/tools/retrieve-types.ts`.
//!
//! Direct-Qdrant access — **daemon-independent** (works when daemon is down).
//!
//! # Result shape (field order matches TS `RetrieveResponse` declaration)
//!
//! ```json
//! // success by-id
//! { "success": true, "documents": [{ "id": "...", "content": "...",
//!   "metadata": {...} }], "total": 1, "hasMore": false }
//!
//! // success by-filter (empty)
//! { "success": true, "documents": [], "total": 0, "hasMore": false }
//!
//! // collection not found (normalised to success)
//! { "success": true, "documents": [], "total": 0, "hasMore": false,
//!   "message": "Collection not found or empty" }
//!
//! // unresolvable scope
//! { "success": false, "documents": [], "total": 0, "hasMore": false,
//!   "message": "Cannot retrieve from \"projects\" without a resolvable scope. ..." }
//!
//! // qdrant down
//! { "success": false, "documents": [],
//!   "message": "Failed to retrieve documents: ..." }
//! ```

use std::collections::HashMap;

use rmcp::model::CallToolResult;
use serde::Serialize;
use serde_json::Value;

use crate::qdrant::client::{QdrantReadClient, QdrantRetrievedPoint};
use crate::tools::envelope::ok_text;

// ---------------------------------------------------------------------------
// Constants — canonical Qdrant collection names (from wqm_common)
// ---------------------------------------------------------------------------

use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

/// Keys excluded from metadata — mirrors `extractMetadata` in retrieve-types.ts line 72.
const EXCLUDED_PAYLOAD_KEYS: &[&str] = &["content", "dense_vector", "sparse_vector"];

// ---------------------------------------------------------------------------
// Public trait — injectable for tests
// ---------------------------------------------------------------------------

/// Abstraction over Qdrant read operations needed by the retrieve tool.
pub trait RetrieveQdrant {
    fn retrieve_by_ids(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> impl std::future::Future<Output = Result<Vec<QdrantRetrievedPoint>, String>> + Send;

    fn scroll(
        &self,
        collection: &str,
        filter: Option<RetrieveFilter>,
        limit: u32,
    ) -> impl std::future::Future<Output = Result<Vec<QdrantRetrievedPoint>, String>> + Send;
}

/// Simple key=value filter for scroll operations.
#[derive(Debug, Clone)]
pub struct RetrieveFilter {
    pub must: Vec<(String, String)>,
}

impl RetrieveQdrant for QdrantReadClient {
    async fn retrieve_by_ids(
        &self,
        collection: &str,
        ids: Vec<String>,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        self.retrieve(collection, ids)
            .await
            .map_err(|e| e.to_string())
    }

    async fn scroll(
        &self,
        collection: &str,
        filter: Option<RetrieveFilter>,
        limit: u32,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        use qdrant_client::qdrant::{Condition, Filter};

        let qdrant_filter = filter.map(|f| {
            let conditions: Vec<_> = f
                .must
                .into_iter()
                .map(|(key, value)| Condition::matches(key, value))
                .collect();
            Filter::must(conditions)
        });

        self.scroll(collection, qdrant_filter, limit, None)
            .await
            .map(|(points, _next)| points)
            .map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Input struct
// ---------------------------------------------------------------------------

/// Input arguments for the `retrieve` tool.
#[derive(Debug, Default)]
pub struct RetrieveInput {
    pub document_id: Option<String>,
    /// "projects" | "libraries" | "rules" | "scratchpad" — default "projects"
    pub collection: String,
    pub filter: Option<HashMap<String, String>>,
    pub limit: u32,
    pub offset: u32,
    pub project_id: Option<String>,
    pub library_name: Option<String>,
}

impl RetrieveInput {
    /// Parse from the JSON `arguments` map of a `CallToolRequestParams`.
    ///
    /// Mirrors the destructuring defaults in retrieve.ts lines 110-119.
    pub fn from_args(args: &serde_json::Map<String, Value>) -> Self {
        let document_id = args
            .get("documentId")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let collection = args
            .get("collection")
            .and_then(|v| v.as_str())
            .unwrap_or("projects")
            .to_string();

        let filter = args.get("filter").and_then(|v| v.as_object()).map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        });

        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as u32;

        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        let project_id = args
            .get("projectId")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let library_name = args
            .get("libraryName")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        Self {
            document_id,
            collection,
            filter,
            limit,
            offset,
            project_id,
            library_name,
        }
    }
}

// ---------------------------------------------------------------------------
// Result structs — field ORDER matches TS declarations for JSON parity
// ---------------------------------------------------------------------------

/// Single retrieved document — mirrors TS `RetrievedDocument` (retrieve-types.ts lines 31-36).
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct RetrievedDocument {
    pub id: String,
    pub content: String,
    pub metadata: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// Tool response — mirrors TS `RetrieveResponse` (retrieve-types.ts lines 38-44).
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct RetrieveResponse {
    pub success: bool,
    pub documents: Vec<RetrievedDocument>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// camelCase per TS interface — use `#[serde(rename)]`.
    #[serde(rename = "hasMore")]
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map collection type string to canonical Qdrant collection name.
fn collection_name(collection: &str) -> &str {
    match collection {
        "projects" => COLLECTION_PROJECTS,
        "libraries" => COLLECTION_LIBRARIES,
        "rules" => COLLECTION_RULES,
        "scratchpad" => COLLECTION_SCRATCHPAD,
        _ => COLLECTION_PROJECTS,
    }
}

/// Extract metadata from a payload, excluding content and vector keys.
///
/// Mirrors `extractMetadata` in retrieve-types.ts lines 68-78.
fn extract_metadata(payload: &HashMap<String, Value>) -> Value {
    let map: serde_json::Map<String, Value> = payload
        .iter()
        .filter(|(k, _)| !EXCLUDED_PAYLOAD_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Value::Object(map)
}

/// Build an "unresolvable scope" error response — mirrors `unresolvedTenantResponse`.
fn unresolved_tenant_response(collection: &str) -> RetrieveResponse {
    RetrieveResponse {
        success: false,
        documents: vec![],
        total: Some(0),
        has_more: false,
        message: Some(format!(
            "Cannot retrieve from \"{collection}\" without a resolvable scope. \
             Pass `projectId` (for projects) or `libraryName` (for libraries), \
             or run from a registered project directory."
        )),
    }
}

/// Normalise a Qdrant error to success-empty when the collection is not found.
///
/// Mirrors retrieve.ts lines 264-272.
fn is_collection_not_found(err: &str) -> bool {
    err.contains("not found") || err.contains("doesn't exist")
}

fn extract_content(payload: &HashMap<String, Value>) -> String {
    payload
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn point_to_document(point: QdrantRetrievedPoint) -> RetrievedDocument {
    let content = extract_content(&point.payload);
    let metadata = extract_metadata(&point.payload);
    RetrievedDocument {
        id: point.id,
        content,
        metadata,
        score: None,
    }
}

// ---------------------------------------------------------------------------
// Tool function
// ---------------------------------------------------------------------------

/// Execute the `retrieve` tool.
///
/// Mirrors `RetrieveTool.retrieve()` in retrieve.ts. Daemon-independent.
pub async fn retrieve_tool<Q>(input: RetrieveInput, qdrant: &Q) -> CallToolResult
where
    Q: RetrieveQdrant,
{
    let coll = input.collection.as_str();
    let coll_name = collection_name(coll);

    // F-002 / F-011: resolve project scope up front for projects + scratchpad.
    let resolved_project_id = if coll == "projects" || coll == "scratchpad" {
        input.project_id.clone()
    } else {
        None
    };

    if let Some(doc_id) = input.document_id {
        return retrieve_by_id(
            coll,
            coll_name,
            &doc_id,
            resolved_project_id.as_deref(),
            input.library_name.as_deref(),
            qdrant,
        )
        .await;
    }

    // F-011: refuse broad scroll when scope cannot be resolved.
    if (coll == "projects" || coll == "scratchpad") && resolved_project_id.is_none() {
        return ok_text(&unresolved_tenant_response(coll));
    }
    if coll == "libraries" && input.library_name.is_none() {
        return ok_text(&unresolved_tenant_response(coll));
    }

    retrieve_by_filter(
        coll,
        coll_name,
        input.filter,
        input.limit,
        resolved_project_id.as_deref(),
        input.library_name.as_deref(),
        qdrant,
    )
    .await
}

async fn retrieve_by_id<Q>(
    collection: &str,
    coll_name: &str,
    doc_id: &str,
    resolved_project_id: Option<&str>,
    library_name: Option<&str>,
    qdrant: &Q,
) -> CallToolResult
where
    Q: RetrieveQdrant,
{
    // F-002: require scope for projects/scratchpad/libraries before reading.
    if (collection == "projects" || collection == "scratchpad") && resolved_project_id.is_none() {
        return ok_text(&unresolved_tenant_response(collection));
    }
    if collection == "libraries" && library_name.is_none() {
        return ok_text(&unresolved_tenant_response(collection));
    }

    match qdrant
        .retrieve_by_ids(coll_name, vec![doc_id.to_string()])
        .await
    {
        Ok(points) => {
            let point = match points.into_iter().next() {
                Some(p) => p,
                None => {
                    let resp = RetrieveResponse {
                        success: false,
                        documents: vec![],
                        total: None,
                        has_more: false,
                        message: Some(format!("Document not found: {doc_id}")),
                    };
                    return ok_text(&resp);
                }
            };

            // F-002: ownership check.
            if !payload_matches_scope(
                &point.payload,
                collection,
                resolved_project_id,
                library_name,
            ) {
                let resp = RetrieveResponse {
                    success: false,
                    documents: vec![],
                    total: None,
                    has_more: false,
                    message: Some(format!("Document not found: {doc_id}")),
                };
                return ok_text(&resp);
            }

            let document = point_to_document(point);
            ok_text(&RetrieveResponse {
                success: true,
                documents: vec![document],
                total: Some(1),
                has_more: false,
                message: None,
            })
        }
        Err(err) => ok_text(&RetrieveResponse {
            success: false,
            documents: vec![],
            total: None,
            has_more: false,
            message: Some(format!("Failed to retrieve document: {}", err)),
        }),
    }
}

async fn retrieve_by_filter<Q>(
    collection: &str,
    coll_name: &str,
    extra_filter: Option<HashMap<String, String>>,
    limit: u32,
    project_id: Option<&str>,
    library_name: Option<&str>,
    qdrant: &Q,
) -> CallToolResult
where
    Q: RetrieveQdrant,
{
    // Build filter conditions.
    let mut must: Vec<(String, String)> = vec![];

    match collection {
        "projects" | "scratchpad" => {
            if let Some(pid) = project_id {
                must.push(("tenant_id".to_string(), pid.to_string()));
            }
        }
        "libraries" => {
            if let Some(lib) = library_name {
                must.push(("tenant_id".to_string(), lib.to_string()));
            }
        }
        _ => {}
    }

    if let Some(f) = extra_filter {
        for (k, v) in f {
            must.push((k, v));
        }
    }

    let filter = if must.is_empty() {
        None
    } else {
        Some(RetrieveFilter { must })
    };

    // Over-fetch by 1 to detect hasMore — retrieve.ts lines 246/252.
    let fetch_limit = limit + 1;

    match qdrant.scroll(coll_name, filter, fetch_limit).await {
        Ok(mut points) => {
            let has_more = points.len() > limit as usize;
            if has_more {
                points.truncate(limit as usize);
            }
            let documents: Vec<RetrievedDocument> =
                points.into_iter().map(point_to_document).collect();
            let total = documents.len() as u64;
            ok_text(&RetrieveResponse {
                success: true,
                documents,
                total: Some(total),
                has_more,
                message: None,
            })
        }
        Err(err) => {
            if is_collection_not_found(&err) {
                // Normalise to success-empty — retrieve.ts:264-272
                ok_text(&RetrieveResponse {
                    success: true,
                    documents: vec![],
                    total: Some(0),
                    has_more: false,
                    message: Some("Collection not found or empty".to_string()),
                })
            } else {
                ok_text(&RetrieveResponse {
                    success: false,
                    documents: vec![],
                    total: None,
                    has_more: false,
                    message: Some(format!("Failed to retrieve documents: {err}")),
                })
            }
        }
    }
}

/// Check whether a point's payload matches the caller's scope.
///
/// Mirrors `payloadMatchesScope` in retrieve.ts lines 64-93.
fn payload_matches_scope(
    payload: &HashMap<String, Value>,
    collection: &str,
    project_id: Option<&str>,
    library_name: Option<&str>,
) -> bool {
    match collection {
        "projects" => {
            let Some(pid) = project_id else { return false };
            payload.get("tenant_id").and_then(|v| v.as_str()) == Some(pid)
        }
        "libraries" => {
            let Some(lib) = library_name else {
                return false;
            };
            payload.get("library_name").and_then(|v| v.as_str()) == Some(lib)
                || payload.get("tenant_id").and_then(|v| v.as_str()) == Some(lib)
        }
        "rules" => true, // intentionally mixed-tenancy
        "scratchpad" => {
            let Some(pid) = project_id else { return false };
            payload.get("tenant_id").and_then(|v| v.as_str()) == Some(pid)
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "retrieve_tests.rs"]
mod tests;
