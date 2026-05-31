//! Search tool response types.
//!
//! Field names and orderings mirror `SearchResult` / `SearchResponse` in
//! `src/typescript/mcp-server/src/tools/search-types.ts` exactly.
//!
//! ## Serialisation contract
//! - `SearchResult`: TS property order: id, score, collection, content, title?,
//!   metadata, provenance?, parent_context?, graph_context?
//! - `SearchResponse`: TS property order: results, total, query, mode, scope,
//!   collections_searched, status?, status_reason?, branch?, diversity_score?
//! - Optional fields are omitted when absent (`skip_serializing_if = "Option::is_none"`).
//! - Scores are raw `f64` — no rounding (task-33 golden suite handles format parity).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Search mode / scope enums
// ---------------------------------------------------------------------------

/// Mode of search — mirrors `SearchMode` in `search-types.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Hybrid,
    Semantic,
    Keyword,
}

impl Default for SearchMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Semantic => "semantic",
            Self::Keyword => "keyword",
        }
    }
}

/// Scope of search — mirrors `SearchScope` in `search-types.ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchScope {
    Project,
    Group,
    All,
}

impl Default for SearchScope {
    fn default() -> Self {
        Self::Project
    }
}

impl SearchScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Group => "group",
            Self::All => "all",
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-structures
// ---------------------------------------------------------------------------

/// Mirrors `Provenance` in `search-types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_project_id: Option<String>,
}

/// Mirrors `ParentContext` in `search-types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentContext {
    pub parent_unit_id: String,
    pub unit_type: String,
    pub unit_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locator: Option<HashMap<String, Value>>,
}

/// Mirrors `GraphContextNode` in `search-types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContextNode {
    pub symbol: String,
    pub file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<i32>,
}

/// Mirrors `GraphContext` in `search-types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContext {
    pub symbol: String,
    pub file_path: String,
    pub callers: Vec<GraphContextNode>,
    pub callees: Vec<GraphContextNode>,
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// One search result — field order mirrors `SearchResult` in `search-types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Point ID (string).
    pub id: String,
    /// Raw score — NOT rounded (task-33 handles float formatting).
    pub score: f64,
    /// Collection this result originates from.
    pub collection: String,
    /// Text content of the result.
    pub content: String,
    /// Optional document title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// All payload fields (raw spread), including `_search_type`.
    pub metadata: HashMap<String, Value>,
    /// Provenance information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// Parent unit context (fetched when expandContext=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_context: Option<ParentContext>,
    /// Graph context (1-hop callers/callees, fetched when includeGraphContext=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_context: Option<GraphContext>,
}

// ---------------------------------------------------------------------------
// SearchResponse
// ---------------------------------------------------------------------------

/// Search response — field order mirrors `SearchResponse` in `search-types.ts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: usize,
    pub query: String,
    pub mode: SearchMode,
    pub scope: SearchScope,
    pub collections_searched: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_reason: Option<String>,
    /// Branch filter applied to this search, absent when cross-branch ("*").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Source diversity score [0, 1] — absent when diversity disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diversity_score: Option<f64>,
}

impl SearchResponse {
    /// Convenience constructor for empty/degraded responses.
    pub fn empty(
        query: String,
        mode: SearchMode,
        scope: SearchScope,
        collections: Vec<String>,
        status: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            results: vec![],
            total: 0,
            query,
            mode,
            scope,
            collections_searched: collections,
            status: Some(status.into()),
            status_reason: Some(reason.into()),
            branch: None,
            diversity_score: None,
        }
    }
}
