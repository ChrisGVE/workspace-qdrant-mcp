//! `retrieve` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/retrieve.ts` +
//! `src/typescript/mcp-server/src/tools/retrieve-types.ts`.
//!
//! Direct-Qdrant access — **daemon-independent** (works when daemon is down).
//!
//! # Result shape (field order matches TS `RetrieveResponse` declaration)
//!
//! ```text
//! // success by-id
//! { "success": true, "documents": [...], "total": 1, "hasMore": false }
//! // by-id not-found / error — NO hasMore key
//! { "success": false, "documents": [], "message": "..." }
//! // by-filter success
//! { "success": true, "documents": [...], "total": N, "hasMore": bool }
//! // collection-not-found (normalised to success) — hasMore included
//! { "success": true, "documents": [], "total": 0, "hasMore": false, "message": "..." }
//! // non-collection-not-found error — NO hasMore key
//! { "success": false, "documents": [], "message": "..." }
//! ```

use std::collections::HashMap;

use rmcp::model::CallToolResult;
use serde_json::Value;

pub use self::retrieve_types::{
    RetrieveFilter, RetrieveInput, RetrieveQdrant, RetrieveResponse, RetrievedDocument,
};
use crate::qdrant::client::QdrantRetrievedPoint;
use crate::tools::envelope::ok_text;

#[path = "retrieve_types.rs"]
mod retrieve_types;

// ---------------------------------------------------------------------------
// Constants — canonical Qdrant collection names (from wqm_common)
// ---------------------------------------------------------------------------

use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

/// Keys excluded from metadata — mirrors `extractMetadata` in retrieve-types.ts line 72.
const EXCLUDED_PAYLOAD_KEYS: &[&str] = &["content", "dense_vector", "sparse_vector"];

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
///
/// NOTE: Key ordering is non-deterministic here (gRPC protobuf map has no
/// defined order) whereas TS reads Qdrant REST JSON which preserves insertion
/// order.  Normalisation of key order is deferred to task-33.
fn extract_metadata(payload: &HashMap<String, Value>) -> Value {
    let map: serde_json::Map<String, Value> = payload
        .iter()
        .filter(|(k, _)| !EXCLUDED_PAYLOAD_KEYS.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Value::Object(map)
}

/// Build an "unresolvable scope" error response — mirrors `unresolvedTenantResponse`
/// (retrieve.ts:44-55 includes `hasMore: false`).
fn unresolved_tenant_response(collection: &str) -> RetrieveResponse {
    RetrieveResponse {
        success: false,
        documents: vec![],
        total: Some(0),
        has_more: Some(false),
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
///
/// `session_project_id` is the fallback project ID from session state,
/// mirroring `projectId ?? (await this.resolveProjectId())` in TS retrieve.ts:124.
/// When the caller does not supply `projectId` in the tool arguments but a
/// session project is registered, the session ID is used instead of refusing.
pub async fn retrieve_tool<Q>(
    input: RetrieveInput,
    qdrant: &Q,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    Q: RetrieveQdrant,
{
    let coll = input.collection.as_str();
    let coll_name = collection_name(coll);

    // F-002 / F-011: resolve project scope up front for projects + scratchpad.
    // TS: `projectId ?? (await this.resolveProjectId())` (retrieve.ts:124).
    // The Rust equivalent of resolveProjectId() is session_project_id —
    // resolved once at session init; use it as the fallback when input.project_id
    // is absent.
    let resolved_project_id = if coll == "projects" || coll == "scratchpad" {
        input
            .project_id
            .clone()
            .or_else(|| session_project_id.map(str::to_string))
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
        input.offset,
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
                // Not found — TS retrieve.ts:202 omits `hasMore` entirely.
                None => {
                    return ok_text(&RetrieveResponse {
                        success: false,
                        documents: vec![],
                        total: None,
                        has_more: None,
                        message: Some(format!("Document not found: {doc_id}")),
                    });
                }
            };

            // F-002: ownership check. TS retrieve.ts:208 omits `hasMore` on mismatch.
            if !payload_matches_scope(
                &point.payload,
                collection,
                resolved_project_id,
                library_name,
            ) {
                return ok_text(&RetrieveResponse {
                    success: false,
                    documents: vec![],
                    total: None,
                    has_more: None,
                    message: Some(format!("Document not found: {doc_id}")),
                });
            }

            let document = point_to_document(point);
            // Success — TS retrieve.ts:217 includes `hasMore: false`.
            ok_text(&RetrieveResponse {
                success: true,
                documents: vec![document],
                total: Some(1),
                has_more: Some(false),
                message: None,
            })
        }
        // Catch error — TS retrieve.ts:219-223 omits `hasMore` entirely.
        Err(err) => ok_text(&RetrieveResponse {
            success: false,
            documents: vec![],
            total: None,
            has_more: None,
            message: Some(format!("Failed to retrieve document: {}", err)),
        }),
    }
}

async fn retrieve_by_filter<Q>(
    collection: &str,
    coll_name: &str,
    extra_filter: Option<HashMap<String, String>>,
    limit: u32,
    offset: u32,
    project_id: Option<&str>,
    library_name: Option<&str>,
    qdrant: &Q,
) -> CallToolResult
where
    Q: RetrieveQdrant,
{
    let filter = build_scroll_filter(collection, extra_filter, project_id, library_name);

    // Over-fetch by 1 to detect hasMore — retrieve.ts lines 246/252.
    let fetch_limit = limit + 1;

    match qdrant.scroll(coll_name, filter, fetch_limit, offset).await {
        Ok(mut points) => {
            let has_more = points.len() > limit as usize;
            if has_more {
                points.truncate(limit as usize);
            }
            let documents: Vec<RetrievedDocument> =
                points.into_iter().map(point_to_document).collect();
            let total = documents.len() as u64;
            // Success — TS retrieve.ts:261 includes `hasMore`.
            ok_text(&RetrieveResponse {
                success: true,
                documents,
                total: Some(total),
                has_more: Some(has_more),
                message: None,
            })
        }
        Err(err) if is_collection_not_found(&err) => {
            // Normalise to success-empty — TS retrieve.ts:265-271 includes `hasMore: false`.
            ok_text(&RetrieveResponse {
                success: true,
                documents: vec![],
                total: Some(0),
                has_more: Some(false),
                message: Some("Collection not found or empty".to_string()),
            })
        }
        // Non-collection-not-found error — TS retrieve.ts:273-277 omits `hasMore`.
        Err(err) => ok_text(&RetrieveResponse {
            success: false,
            documents: vec![],
            total: None,
            has_more: None,
            message: Some(format!("Failed to retrieve documents: {err}")),
        }),
    }
}

/// Build the `RetrieveFilter` for a scroll-based retrieve.
///
/// Extracted to keep `retrieve_by_filter` under 80 lines.
fn build_scroll_filter(
    collection: &str,
    extra_filter: Option<HashMap<String, String>>,
    project_id: Option<&str>,
    library_name: Option<&str>,
) -> Option<RetrieveFilter> {
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

    if must.is_empty() {
        None
    } else {
        Some(RetrieveFilter { must })
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
