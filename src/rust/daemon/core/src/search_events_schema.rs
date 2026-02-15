//! Schema for the `search_events` table.
//!
//! Logs all search operations across tools (MCP, grep, ripgrep, etc.)
//! for pipeline instrumentation and behavior analysis.

/// SQL to create the search_events table
pub const CREATE_SEARCH_EVENTS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS search_events (
    id TEXT PRIMARY KEY NOT NULL,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    session_id TEXT,
    project_id TEXT,
    actor TEXT NOT NULL CHECK (actor IN ('claude', 'user', 'daemon')),
    tool TEXT NOT NULL CHECK (tool IN ('mcp_qdrant', 'rg', 'grep', 'ctags', 'lsp', 'filesearch')),
    op TEXT NOT NULL CHECK (op IN ('search', 'expand', 'open', 'followup')),
    query_text TEXT,
    filters TEXT,
    top_k INTEGER,
    result_count INTEGER,
    latency_ms INTEGER,
    top_result_refs TEXT,
    outcome TEXT,
    parent_event_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// Indexes for the search_events table
pub const CREATE_SEARCH_EVENTS_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_search_events_session ON search_events(session_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_search_events_tool ON search_events(tool, ts)",
    "CREATE INDEX IF NOT EXISTS idx_search_events_project ON search_events(project_id, ts)",
];

/// Record representing a row in the search_events table
#[derive(Debug, Clone)]
pub struct SearchEvent {
    pub id: String,
    pub ts: String,
    pub session_id: Option<String>,
    pub project_id: Option<String>,
    pub actor: String,
    pub tool: String,
    pub op: String,
    pub query_text: Option<String>,
    pub filters: Option<String>,
    pub top_k: Option<i32>,
    pub result_count: Option<i32>,
    pub latency_ms: Option<i32>,
    pub top_result_refs: Option<String>,
    pub outcome: Option<String>,
    pub parent_event_id: Option<String>,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sql_is_valid() {
        assert!(CREATE_SEARCH_EVENTS_SQL.contains("CREATE TABLE"));
        assert!(CREATE_SEARCH_EVENTS_SQL.contains("search_events"));
        assert!(CREATE_SEARCH_EVENTS_SQL.contains("actor TEXT NOT NULL"));
        assert!(CREATE_SEARCH_EVENTS_SQL.contains("tool TEXT NOT NULL"));
        assert!(CREATE_SEARCH_EVENTS_SQL.contains("op TEXT NOT NULL"));
    }

    #[test]
    fn test_indexes_count() {
        assert_eq!(CREATE_SEARCH_EVENTS_INDEXES_SQL.len(), 3);
    }

    #[test]
    fn test_indexes_are_idempotent() {
        for index_sql in CREATE_SEARCH_EVENTS_INDEXES_SQL {
            assert!(index_sql.contains("IF NOT EXISTS"));
        }
    }
}
