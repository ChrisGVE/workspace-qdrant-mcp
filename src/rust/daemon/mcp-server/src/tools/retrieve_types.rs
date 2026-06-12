//! Types, constants, and the `RetrieveQdrant` trait for the `retrieve` tool.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/retrieve-types.ts`.

use std::collections::HashMap;

use serde::Serialize;
use serde_json::Value;

use crate::qdrant::client::{QdrantReadClient, QdrantRetrievedPoint};

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

    /// Scroll `collection` with an optional filter.
    ///
    /// `offset` mirrors TS retrieve.ts:247: `if (offset > 0) scrollRequest.offset = offset`.
    /// Pass `0` to start from the beginning (no offset forwarded to Qdrant).
    fn scroll(
        &self,
        collection: &str,
        filter: Option<RetrieveFilter>,
        limit: u32,
        offset: u32,
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
        offset: u32,
    ) -> Result<Vec<QdrantRetrievedPoint>, String> {
        use qdrant_client::qdrant::{Condition, Filter, PointId};

        let qdrant_filter = filter.map(|f| {
            let conditions: Vec<_> = f
                .must
                .into_iter()
                .map(|(key, value)| Condition::matches(key, value))
                .collect();
            Filter::must(conditions)
        });

        // Mirror TS retrieve.ts:247: `if (offset > 0) scrollRequest.offset = offset`.
        // Qdrant scroll offset is a PointId; a numeric offset maps to PointId::Num(u64).
        let qdrant_offset = if offset > 0 {
            Some(PointId::from(offset as u64))
        } else {
            None
        };

        self.scroll(collection, qdrant_filter, limit, qdrant_offset)
            .await
            .map(|(points, _next)| points)
            .map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Argument allowlist
// ---------------------------------------------------------------------------

/// The complete set of argument keys the `retrieve` tool accepts.
///
/// Kept in lock-step with `retrieve_schema()` in `tools/definitions.rs` (the
/// published `inputSchema`). `cwd` is NOT included: in this server the working
/// directory is carried on the session (`session.client_cwd`, #97), never as a
/// per-call argument, so a `cwd` key in the args map is genuinely unknown.
///
/// Salvaged from alkmimm PR #134 (`8e61a37cc`): the TS arg-builder silently
/// discarded unrecognized args, so a misdirected `retrieve({query: "..."})`
/// degraded into an unrelated "no resolvable scope" error. Rejecting unknown
/// args up front turns that confusing failure into an actionable one.
pub const RETRIEVE_ARG_KEYS: &[&str] = &[
    "documentId",
    "collection",
    "filter",
    "limit",
    "offset",
    "projectId",
    "libraryName",
];

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
    /// Mirrors the destructuring defaults in retrieve.ts lines 110-119, plus the
    /// unknown-argument rejection salvaged from alkmimm PR #134 (`8e61a37cc`):
    /// any key outside [`RETRIEVE_ARG_KEYS`] is rejected before scope resolution
    /// or any Qdrant call, with a hint that points a stray `query` at the
    /// `search` tool.
    ///
    /// # Errors
    /// Returns `Err(message)` when the args map carries any unrecognized key.
    pub fn from_args(args: &serde_json::Map<String, Value>) -> Result<Self, String> {
        if let Some(err) = Self::reject_unknown_args(args) {
            return Err(err);
        }

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

        Ok(Self {
            document_id,
            collection,
            filter,
            limit,
            offset,
            project_id,
            library_name,
        })
    }

    /// Return an error message if `args` contains any key outside
    /// [`RETRIEVE_ARG_KEYS`], else `None`.
    ///
    /// Unknown keys are listed in their map order. When `query` is among them the
    /// message also nudges the caller toward the `search` tool, since
    /// `retrieve({query: ...})` is the canonical mistake this guard catches.
    fn reject_unknown_args(args: &serde_json::Map<String, Value>) -> Option<String> {
        let unknown: Vec<&str> = args
            .keys()
            .map(String::as_str)
            .filter(|k| !RETRIEVE_ARG_KEYS.contains(k))
            .collect();

        if unknown.is_empty() {
            return None;
        }

        let mut message = format!(
            "Unknown argument(s) for retrieve: {}. Allowed: {}.",
            unknown.join(", "),
            RETRIEVE_ARG_KEYS.join(", "),
        );
        if unknown.contains(&"query") {
            message.push_str(
                " The `retrieve` tool fetches known documents by id or filter; \
                 use the `search` tool to run a query.",
            );
        }
        Some(message)
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
///
/// `has_more` is `Option<bool>` because TS omits the `hasMore` key entirely on
/// not-found and error responses (retrieve.ts:202, 208, 219-223, 273-277).
/// It is only included on success paths and `unresolvedTenantResponse`.
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct RetrieveResponse {
    pub success: bool,
    pub documents: Vec<RetrievedDocument>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// camelCase per TS interface.  Absent when TS omits the key entirely.
    #[serde(rename = "hasMore", skip_serializing_if = "Option::is_none")]
    pub has_more: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}
