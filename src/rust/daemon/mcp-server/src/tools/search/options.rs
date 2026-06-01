//! Search input parsing and option defaults.
//!
//! Mirrors `SearchOptions` from `search-types.ts` and `SearchTool.search()`
//! from `search.ts`.

use serde_json::Value;

use super::types::{SearchMode, SearchScope};

// ---------------------------------------------------------------------------
// Default constants (mirrors search-types.ts)
// ---------------------------------------------------------------------------

/// Default result limit — `DEFAULT_LIMIT = 10` in `search-types.ts`.
pub const DEFAULT_LIMIT: usize = 10;

/// Default max_results for exact (FTS5) mode when caller did not specify limit.
/// Mirrors TS `buildExactSearchRequest`: `options.limit ?? 100` (search-exact.ts:95).
pub const DEFAULT_EXACT_LIMIT: usize = 100;

/// Default score threshold applied at the Qdrant query level only —
/// `DEFAULT_SCORE_THRESHOLD = 0.3` in `search-types.ts`.
///
/// CRITICAL: this is NEVER applied to fused/post-RRF results. Only used
/// as the `score_threshold` parameter on individual Qdrant search calls.
/// See scratchpad note "MCP-Rust task 30 search wiring: do NOT apply
/// post-fusion score threshold".
pub const DEFAULT_SCORE_THRESHOLD: f64 = 0.3;

/// Default sparse expansion weight — `DEFAULT_EXPANSION_WEIGHT = 0.5`.
pub const DEFAULT_EXPANSION_WEIGHT: f64 = 0.5;

/// Default max expanded keywords — `DEFAULT_MAX_EXPANDED_KEYWORDS = 10`.
pub const DEFAULT_MAX_EXPANDED_KEYWORDS: usize = 10;

// ---------------------------------------------------------------------------
// SearchInput — deserialized from MCP tool arguments
// ---------------------------------------------------------------------------

/// Deserialized input from the `search` MCP tool call arguments.
///
/// All fields are optional to match the TS interface; defaults are applied
/// in [`SearchOptions::from_input`].
///
/// NOTE: `mode` and `scope` are stored as `Option<String>` internally so that
/// unrecognized enum values are silently dropped (matching TS
/// `buildSearchOptions` which only sets them when the value is recognised).
/// Conversion to typed enums happens in [`SearchOptions::from_input`].
#[derive(Debug, Clone, Default)]
pub struct SearchInput {
    pub query: String,
    pub collection: Option<String>,
    pub mode: Option<SearchMode>,
    pub limit: Option<usize>,
    pub score_threshold: Option<f64>,
    pub scope: Option<SearchScope>,
    pub branch: Option<String>,
    pub file_type: Option<String>,
    pub project_id: Option<String>,
    pub library_name: Option<String>,
    pub library_path: Option<String>,
    pub include_libraries: Option<bool>,
    pub tag: Option<String>,
    pub tags: Option<Vec<String>>,
    pub expand_context: Option<bool>,
    pub path_glob: Option<String>,
    pub component: Option<String>,
    pub exact: Option<bool>,
    pub context_lines: Option<usize>,
    pub include_graph_context: Option<bool>,
    pub diverse: Option<bool>,
}

// ---------------------------------------------------------------------------
// SearchOptions — resolved/defaulted options
// ---------------------------------------------------------------------------

/// Fully resolved search options with all defaults applied.
///
/// Created from a [`SearchInput`] with defaults matching `SearchTool.search()`
/// in `search.ts` lines 228-235.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub collection: Option<String>,
    pub mode: SearchMode,
    pub limit: usize,
    pub score_threshold: f64,
    pub scope: SearchScope,
    pub branch: Option<String>,
    pub file_type: Option<String>,
    pub project_id: Option<String>,
    pub library_name: Option<String>,
    pub library_path: Option<String>,
    pub include_libraries: bool,
    pub tag: Option<String>,
    pub tags: Option<Vec<String>>,
    pub expand_context: bool,
    pub path_glob: Option<String>,
    pub component: Option<String>,
    pub exact: bool,
    pub context_lines: usize,
    pub include_graph_context: bool,
    pub diverse: bool,
    /// True when the caller explicitly passed a `limit` value.
    /// Used by exact mode to default to `DEFAULT_EXACT_LIMIT` (100)
    /// instead of `DEFAULT_LIMIT` (10) when unset.
    pub limit_explicit: bool,
}

impl SearchOptions {
    /// Build resolved options from raw input, applying all defaults.
    ///
    /// Mirrors `SearchTool.search()` lines 228-235 and constructor defaults.
    pub fn from_input(input: SearchInput, current_branch: Option<&str>) -> Self {
        // Branch: use explicit branch from input, or fall back to session's current branch.
        let branch = input.branch.or_else(|| current_branch.map(str::to_string));
        let limit_explicit = input.limit.is_some();
        Self {
            query: input.query,
            collection: input.collection,
            mode: input.mode.unwrap_or(SearchMode::Hybrid),
            limit: input.limit.unwrap_or(DEFAULT_LIMIT),
            score_threshold: input.score_threshold.unwrap_or(DEFAULT_SCORE_THRESHOLD),
            scope: input.scope.unwrap_or(SearchScope::Project),
            branch,
            file_type: input.file_type,
            project_id: input.project_id,
            library_name: input.library_name,
            library_path: input.library_path,
            include_libraries: input.include_libraries.unwrap_or(false),
            tag: input.tag,
            tags: input.tags,
            expand_context: input.expand_context.unwrap_or(false),
            path_glob: input.path_glob,
            component: input.component,
            exact: input.exact.unwrap_or(false),
            context_lines: input.context_lines.unwrap_or(0),
            include_graph_context: input.include_graph_context.unwrap_or(false),
            diverse: input.diverse.unwrap_or(true),
            limit_explicit,
        }
    }

    /// Parse raw MCP tool arguments into a `SearchInput`.
    ///
    /// Matches the permissive TS `buildSearchOptions` (tool-builders/search.ts):
    /// - `query` defaults to `""` when absent (TS line 130: `?? ''`).
    /// - `mode` and `scope` are ONLY set when the value is a recognised string;
    ///   unknown values are silently dropped (not an error).
    /// - All other fields are extracted permissively (wrong type → None).
    pub fn parse_args(args: &serde_json::Map<String, Value>) -> Result<SearchInput, String> {
        // query: defaults to "" when absent (TS: `(args?.['query'] as string) ?? ''`)
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // mode: only set when value is a recognised string
        let mode = args.get("mode").and_then(|v| v.as_str()).and_then(|s| {
            match s {
                "hybrid" => Some(SearchMode::Hybrid),
                "semantic" => Some(SearchMode::Semantic),
                "keyword" => Some(SearchMode::Keyword),
                _ => None, // unrecognized → silently dropped (TS behaviour)
            }
        });

        // scope: only set when value is a recognised string
        let scope = args.get("scope").and_then(|v| v.as_str()).and_then(|s| {
            match s {
                "project" => Some(SearchScope::Project),
                "group" => Some(SearchScope::Group),
                "all" => Some(SearchScope::All),
                _ => None, // unrecognized → silently dropped (TS behaviour)
            }
        });

        Ok(SearchInput {
            query,
            collection: args
                .get("collection")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            mode,
            limit: args
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            score_threshold: args.get("scoreThreshold").and_then(|v| v.as_f64()),
            scope,
            branch: args
                .get("branch")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            file_type: args
                .get("fileType")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            project_id: args
                .get("projectId")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            library_name: args
                .get("libraryName")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            library_path: args
                .get("libraryPath")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            include_libraries: args.get("includeLibraries").and_then(|v| v.as_bool()),
            tag: args.get("tag").and_then(|v| v.as_str()).map(str::to_string),
            tags: args.get("tags").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|e| e.as_str().map(str::to_string))
                    .collect()
            }),
            expand_context: args.get("expandContext").and_then(|v| v.as_bool()),
            path_glob: args
                .get("pathGlob")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            component: args
                .get("component")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            exact: args.get("exact").and_then(|v| v.as_bool()),
            context_lines: args
                .get("contextLines")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            include_graph_context: args.get("includeGraphContext").and_then(|v| v.as_bool()),
            diverse: args.get("diverse").and_then(|v| v.as_bool()),
        })
    }
}
