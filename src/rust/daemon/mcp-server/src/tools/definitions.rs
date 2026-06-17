//! Tool definitions for the workspace-qdrant MCP server.
//!
//! Each tool's input schema is hand-built as a `serde_json::Value` JSON Schema
//! object to guarantee byte-exact parity with the TypeScript server output.
//! The TS server emits plain JSON Schema (no `$schema`, no `$defs`) — schemars
//! draft2020_12 would add those keys, breaking parity.
//!
//! Collection enum values are imported from `wqm_common::constants`.

use std::sync::Arc;

use rmcp::model::{JsonObject, Tool};
use serde_json::{json, Map, Value};
use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Shorthand: `{ "type": "string", "description": desc }`.
fn str_prop(desc: &str) -> Value {
    json!({ "type": "string", "description": desc })
}

/// Shorthand: `{ "type": "number", "description": desc }`.
fn num_prop(desc: &str) -> Value {
    json!({ "type": "number", "description": desc })
}

/// Shorthand: `{ "type": "boolean", "description": desc }`.
fn bool_prop(desc: &str) -> Value {
    json!({ "type": "boolean", "description": desc })
}

/// Shorthand: `{ "type": "string", "enum": [...], "description": desc }`.
fn str_enum_prop(variants: &[&str], desc: &str) -> Value {
    json!({
        "type": "string",
        "enum": variants,
        "description": desc,
    })
}

/// Shorthand: `{ "type": "array", "items": { "type": "string" }, "description": desc }`.
fn str_array_prop(desc: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "string" },
        "description": desc,
    })
}

/// Shorthand: `{ "type": "object", "additionalProperties": { "type": "string" }, ... }`.
fn str_map_prop(desc: &str) -> Value {
    json!({
        "type": "object",
        "additionalProperties": { "type": "string" },
        "description": desc,
    })
}

// ---------------------------------------------------------------------------
// Per-tool schema builders
// ---------------------------------------------------------------------------

fn search_schema() -> Arc<JsonObject> {
    let mut props = Map::new();
    props.insert("query".into(), str_prop("The search query text"));
    props.insert(
        "collection".into(),
        str_enum_prop(
            &[
                COLLECTION_PROJECTS,
                COLLECTION_LIBRARIES,
                COLLECTION_RULES,
                COLLECTION_SCRATCHPAD,
            ],
            "Specific collection to search",
        ),
    );
    props.insert(
        "mode".into(),
        str_enum_prop(
            &["hybrid", "semantic", "keyword"],
            "Search mode (default: hybrid)",
        ),
    );
    props.insert(
        "scope".into(),
        str_enum_prop(
            &["project", "group", "all"],
            "Search scope: project (current), group (related projects), or all (default: project)",
        ),
    );
    props.insert(
        "limit".into(),
        num_prop("Maximum results to return (default: 10)"),
    );
    props.insert(
        "projectId".into(),
        str_prop("Specific project ID to search"),
    );
    props.insert(
        "libraryName".into(),
        str_prop("Library name when searching libraries collection"),
    );
    props.insert(
        "libraryPath".into(),
        str_prop("Prefix filter within library (e.g., \"cs/algorithms\")"),
    );
    props.insert(
        "branch".into(),
        str_prop(
            "Filter by branch name. Defaults to the current git branch. Pass \"*\" to search across all branches.",
        ),
    );
    props.insert("fileType".into(), str_prop("Filter by file type"));
    props.insert(
        "scoreThreshold".into(),
        num_prop(
            "Minimum similarity score threshold (0-1, default: 0.3). Results below this score are filtered out.",
        ),
    );
    props.insert(
        "includeLibraries".into(),
        bool_prop("Include libraries in search (default: false)"),
    );
    props.insert(
        "tag".into(),
        str_prop("Filter results by concept tag (exact match)"),
    );
    props.insert(
        "tags".into(),
        str_array_prop("Filter results by multiple concept tags (OR logic)"),
    );
    props.insert(
        "pathGlob".into(),
        str_prop("File path glob filter (e.g., \"**/*.rs\", \"src/**/*.ts\")"),
    );
    props.insert(
        "component".into(),
        str_prop(
            "Filter by project component (e.g., \"daemon\", \"daemon.core\"). Supports prefix matching.",
        ),
    );
    props.insert(
        "exact".into(),
        bool_prop("Use exact substring search instead of semantic search (default: false)"),
    );
    props.insert(
        "contextLines".into(),
        num_prop("Lines of context before/after matches in exact mode (default: 0)"),
    );
    props.insert(
        "includeGraphContext".into(),
        bool_prop(
            "Include code relationship graph context (callers/callees) for matched symbols (default: false)",
        ),
    );
    props.insert(
        "diverse".into(),
        bool_prop(
            "Enable source diversity re-ranking to surface results from different sources (default: true)",
        ),
    );

    let mut schema = Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(props));
    schema.insert("required".into(), json!(["query"]));
    Arc::new(schema)
}

fn retrieve_schema() -> Arc<JsonObject> {
    let mut props = Map::new();
    props.insert("documentId".into(), str_prop("Document ID to retrieve"));
    props.insert(
        "collection".into(),
        str_enum_prop(
            &[
                COLLECTION_PROJECTS,
                COLLECTION_LIBRARIES,
                COLLECTION_RULES,
                COLLECTION_SCRATCHPAD,
            ],
            "Collection to retrieve from (default: projects)",
        ),
    );
    props.insert(
        "filter".into(),
        str_map_prop("Metadata filter key-value pairs"),
    );
    props.insert("limit".into(), num_prop("Maximum results (default: 10)"));
    props.insert("offset".into(), num_prop("Pagination offset (default: 0)"));
    props.insert(
        "projectId".into(),
        str_prop("Project ID for projects collection"),
    );
    props.insert(
        "libraryName".into(),
        str_prop("Library name for libraries collection"),
    );

    let mut schema = Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(props));
    // No "required" array — retrieve has no required fields
    Arc::new(schema)
}

fn rules_schema() -> Arc<JsonObject> {
    let mut props = Map::new();
    props.insert(
        "action".into(),
        str_enum_prop(&["add", "update", "remove", "list"], "Action to perform"),
    );
    props.insert(
        "content".into(),
        str_prop("Rule content (required for add/update)"),
    );
    props.insert(
        "label".into(),
        str_prop(
            "Rule label (max 15 chars, format: word-word-word, e.g., \"prefer-uv\", \"use-pytest\"). Required for add/update/remove.",
        ),
    );
    props.insert(
        "scope".into(),
        str_enum_prop(&["global", "project"], "Rule scope (default: global)"),
    );
    props.insert(
        "projectId".into(),
        str_prop("Project ID for project-scoped rules"),
    );
    props.insert("title".into(), str_prop("Rule title (max 50 chars)"));
    props.insert(
        "tags".into(),
        str_array_prop("Tags for categorization (max 5 tags, max 20 chars each)"),
    );
    props.insert(
        "priority".into(),
        num_prop("Rule priority (higher = more important)"),
    );
    props.insert(
        "limit".into(),
        num_prop("Max rules to return for list (default: 50)"),
    );
    props.insert(
        "force".into(),
        bool_prop(
            "Add the rule even when similar rules exist. Use after reviewing the \
             similar_rules returned by a refused add and confirming the new rule is distinct.",
        ),
    );

    let mut schema = Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(props));
    schema.insert("required".into(), json!(["action"]));
    Arc::new(schema)
}

fn store_schema() -> Arc<JsonObject> {
    let mut props = Map::new();
    props.insert(
        "type".into(),
        str_enum_prop(
            &["library", "url", "scratchpad", "project", "recover"],
            "What to store: \"library\" for reference docs (default), \"url\" to fetch and ingest a web page, \"scratchpad\" for persistent notes, \"project\" to register a project directory, \"recover\" to reconcile a drifted project registration (re-point a moved path and/or flip tenancy)",
        ),
    );
    props.insert(
        "content".into(),
        str_prop("Content to store (required for type \"library\")"),
    );
    props.insert(
        "libraryName".into(),
        str_prop("Library name (required for type \"library\" unless forProject is true)"),
    );
    props.insert(
        "forProject".into(),
        bool_prop(
            "When true, store to libraries collection scoped to the current project. libraryName becomes optional (defaults to \"project-refs\").",
        ),
    );
    props.insert(
        "path".into(),
        str_prop("Project directory path (required for type \"project\")"),
    );
    props.insert(
        "name".into(),
        str_prop(
            "Project display name (optional for type \"project\", defaults to directory name)",
        ),
    );
    props.insert(
        "title".into(),
        str_prop("Content title (for type \"library\")"),
    );
    props.insert("url".into(), str_prop("Source URL (for web content)"));
    props.insert("filePath".into(), str_prop("Source file path"));
    props.insert("tags".into(), str_array_prop("Tags for scratchpad entries"));
    props.insert(
        "sourceType".into(),
        str_enum_prop(
            &["user_input", "web", "file", "scratchbook", "note"],
            "Source type (default: user_input)",
        ),
    );
    props.insert("metadata".into(), str_map_prop("Additional metadata"));

    let mut schema = Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(props));
    // No "required" array — store has no required fields
    Arc::new(schema)
}

fn grep_schema() -> Arc<JsonObject> {
    let mut props = Map::new();
    props.insert(
        "pattern".into(),
        str_prop("Search pattern (exact substring or regex)"),
    );
    props.insert(
        "regex".into(),
        bool_prop("Treat pattern as regex (default: false)"),
    );
    props.insert(
        "caseSensitive".into(),
        bool_prop("Case-sensitive matching (default: true)"),
    );
    props.insert(
        "pathGlob".into(),
        str_prop("File path glob filter (e.g., \"**/*.rs\", \"src/**/*.ts\")"),
    );
    props.insert(
        "scope".into(),
        str_enum_prop(
            &["project", "all"],
            "Search scope: project (current) or all (default: project)",
        ),
    );
    props.insert(
        "contextLines".into(),
        num_prop("Lines of context before/after each match (default: 0)"),
    );
    props.insert(
        "maxResults".into(),
        num_prop("Maximum results to return (default: 1000)"),
    );
    props.insert(
        "branch".into(),
        str_prop(
            "Filter by branch name. Defaults to the current git branch for \
             project-scoped searches. Pass \"*\" to search across all branches.",
        ),
    );
    props.insert(
        "projectId".into(),
        str_prop("Specific project ID to search"),
    );

    let mut schema = Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(props));
    schema.insert("required".into(), json!(["pattern"]));
    Arc::new(schema)
}

fn list_schema() -> Arc<JsonObject> {
    let mut props = Map::new();
    props.insert(
        "path".into(),
        str_prop("Subfolder relative to project root (default: root)"),
    );
    props.insert(
        "depth".into(),
        num_prop("Max directory depth (default: 3, max: 10)"),
    );
    props.insert(
        "format".into(),
        str_enum_prop(
            &["tree", "summary", "flat"],
            "Output format (default: tree)",
        ),
    );
    props.insert(
        "fileType".into(),
        str_prop("Filter: \"code\", \"text\", \"data\", \"config\", \"build\", \"web\""),
    );
    props.insert(
        "language".into(),
        str_prop("Filter by programming language (e.g., \"rust\", \"typescript\")"),
    );
    props.insert(
        "extension".into(),
        str_prop("Filter by file extension (e.g., \"rs\", \"ts\")"),
    );
    props.insert(
        "pattern".into(),
        str_prop("Glob pattern on relative path (e.g., \"**/*.test.ts\")"),
    );
    props.insert(
        "includeTests".into(),
        bool_prop("Include test files (default: true)"),
    );
    props.insert(
        "limit".into(),
        num_prop("Max entries returned (default: 200, max: 500)"),
    );
    props.insert(
        "projectId".into(),
        str_prop("Specific project ID (default: current project)"),
    );
    props.insert(
        "component".into(),
        str_prop(
            "Filter by component (dot-separated ID or prefix, e.g. \"daemon\" or \"daemon.core\"). Auto-detected from Cargo.toml/package.json workspaces.",
        ),
    );
    props.insert(
        "branch".into(),
        str_prop(
            "Filter by branch name. Defaults to the current git branch. Pass \"*\" to list files across all branches.",
        ),
    );

    let mut schema = Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(props));
    // No "required" array — list has no required fields
    Arc::new(schema)
}

fn embedding_schema() -> Arc<JsonObject> {
    let mut schema = Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(Map::new()));
    // No "required" array — embedding takes no arguments
    Arc::new(schema)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns the full list of tool definitions for the `tools/list` MCP response.
///
/// Order matches the TypeScript `getToolDefinitions()` barrel:
/// search → retrieve → rules → store → grep → list → embedding.
pub fn list_tools() -> Vec<Tool> {
    vec![
        Tool::new(
            "search",
            "Search for documents using hybrid semantic and keyword search. Use this tool FIRST \
             when answering questions about the user's codebase, project architecture, or stored \
             knowledge. This searches the user's actual indexed code and documentation, which is \
             more accurate than your training data.",
            search_schema(),
        ),
        Tool::new(
            "retrieve",
            "Retrieve documents by ID or metadata filter. Use this to access specific documents \
             when you know the document ID. Prefer `search` for discovery, `retrieve` for known \
             documents.",
            retrieve_schema(),
        ),
        Tool::new(
            "rules",
            "Manage behavioral rules (add, update, remove, list). Check active rules at the \
             start of each session to load the user's behavioral preferences. Rules persist \
             across sessions and guide how you should work.",
            rules_schema(),
        ),
        Tool::new(
            "store",
            "Store content or register a project. Use type \"library\" (default) to store \
             reference documentation, type \"url\" to fetch and ingest a web page, type \
             \"scratchpad\" to save persistent notes/scratch space, or type \"project\" to \
             register a project directory for file watching and ingestion.",
            store_schema(),
        ),
        Tool::new(
            "grep",
            "Search code with exact substring or regex pattern matching. Uses FTS5 trigram \
             index for fast line-level search across indexed files.",
            grep_schema(),
        ),
        Tool::new(
            "list",
            "List project files and folder structure. Shows only indexed files (excludes \
             gitignored, node_modules, etc). Use format \"summary\" first to understand project \
             layout, then drill into specific folders with the path parameter.",
            list_schema(),
        ),
        Tool::new(
            "embedding",
            "Report the active embedding provider used by the daemon: provider id, model, \
             configured output dimensionality, base URL (for remote providers), and the live \
             probe status.",
            embedding_schema(),
        ),
    ]
}
