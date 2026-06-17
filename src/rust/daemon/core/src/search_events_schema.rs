//! Schema for the `search_events` table.
//!
//! Logs all search operations across tools (MCP, grep, ripgrep, etc.)
//! for pipeline instrumentation and behavior analysis.
//!
//! The `actor` column distinguishes who issued a search. The quality eval
//! harness (`wqm benchmark search-quality`, #135) tags its own traffic with
//! `actor = 'benchmark'` so organic-query mining can exclude it — see
//! migration v47, which relaxed the `actor` CHECK to admit that value on
//! databases created before #135. Fresh installs pick the relaxed CHECK up
//! directly from this constant via migration v12.

/// SQL to create the search_events table
pub const CREATE_SEARCH_EVENTS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS search_events (
    id TEXT PRIMARY KEY NOT NULL,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    session_id TEXT,
    project_id TEXT,
    actor TEXT NOT NULL CHECK (actor IN ('claude', 'user', 'daemon', 'benchmark')),
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

/// The canonical `actor` values the `search_events.actor` CHECK admits. The
/// server-side `LogSearchEvent` allow-list validation reuses these so a bad
/// value is rejected with a clear error instead of being silently dropped by the
/// SQLite CHECK (#135). Kept in lock-step with the CHECK clause above — the
/// `schema_check_lists_match_allowed_consts` test fails if they ever diverge.
pub const ALLOWED_ACTORS: &[&str] = &["claude", "user", "daemon", "benchmark"];

/// The canonical `tool` values the `search_events.tool` CHECK admits (#135).
pub const ALLOWED_TOOLS: &[&str] = &["mcp_qdrant", "rg", "grep", "ctags", "lsp", "filesearch"];

/// The canonical `op` values the `search_events.op` CHECK admits (#135).
pub const ALLOWED_OPS: &[&str] = &["search", "expand", "open", "followup"];

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
    fn test_actor_check_admits_benchmark() {
        // The eval harness (#135) tags its searches with actor='benchmark';
        // the CHECK must list it alongside the organic actors. Migration v47
        // relaxes the same constraint on pre-#135 databases.
        assert!(CREATE_SEARCH_EVENTS_SQL.contains("'benchmark'"));
        for actor in ["'claude'", "'user'", "'daemon'", "'benchmark'"] {
            assert!(
                CREATE_SEARCH_EVENTS_SQL.contains(actor),
                "actor CHECK must admit {actor}"
            );
        }
    }

    #[test]
    fn schema_check_lists_match_allowed_consts() {
        // Single-source guard (#135): the allow-list constants reused by the
        // gRPC LogSearchEvent validation must list exactly the same values the
        // SQLite CHECK clauses do. If a CHECK gains/loses a value, the matching
        // const must be updated in the same change or this fails.
        for (values, column) in [
            (ALLOWED_ACTORS, "actor"),
            (ALLOWED_TOOLS, "tool"),
            (ALLOWED_OPS, "op"),
        ] {
            for value in values {
                assert!(
                    CREATE_SEARCH_EVENTS_SQL.contains(&format!("'{value}'")),
                    "{column} CHECK must list '{value}' (allow-list/const drift)"
                );
            }
        }
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
