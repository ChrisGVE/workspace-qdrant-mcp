//! Schema for the `resolution_events` table.
//!
//! Tracks when search results led to useful outcomes (file opens,
//! patches applied, manual marks). Links to search_events via
//! foreign key for resolution rate analysis.

/// SQL to create the resolution_events table
pub const CREATE_RESOLUTION_EVENTS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS resolution_events (
    id TEXT PRIMARY KEY NOT NULL,
    search_event_id TEXT NOT NULL,
    resolution_type TEXT NOT NULL CHECK (resolution_type IN ('file_opened', 'patch_applied', 'manual_mark')),
    resolved_ref TEXT,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// Indexes for the resolution_events table
pub const CREATE_RESOLUTION_EVENTS_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_resolution_search ON resolution_events(search_event_id)",
    "CREATE INDEX IF NOT EXISTS idx_resolution_type ON resolution_events(resolution_type, ts)",
];

/// Record representing a row in the resolution_events table
#[derive(Debug, Clone)]
pub struct ResolutionEvent {
    pub id: String,
    pub search_event_id: String,
    pub resolution_type: String,
    pub resolved_ref: Option<String>,
    pub ts: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sql_is_valid() {
        assert!(CREATE_RESOLUTION_EVENTS_SQL.contains("CREATE TABLE"));
        assert!(CREATE_RESOLUTION_EVENTS_SQL.contains("resolution_events"));
        assert!(CREATE_RESOLUTION_EVENTS_SQL.contains("search_event_id TEXT NOT NULL"));
        assert!(CREATE_RESOLUTION_EVENTS_SQL.contains("resolution_type TEXT NOT NULL"));
    }

    #[test]
    fn test_indexes_count() {
        assert_eq!(CREATE_RESOLUTION_EVENTS_INDEXES_SQL.len(), 2);
    }

    #[test]
    fn test_indexes_are_idempotent() {
        for index_sql in CREATE_RESOLUTION_EVENTS_INDEXES_SQL {
            assert!(index_sql.contains("IF NOT EXISTS"));
        }
    }

    #[test]
    fn test_resolution_types_constrained() {
        assert!(CREATE_RESOLUTION_EVENTS_SQL.contains("file_opened"));
        assert!(CREATE_RESOLUTION_EVENTS_SQL.contains("patch_applied"));
        assert!(CREATE_RESOLUTION_EVENTS_SQL.contains("manual_mark"));
    }
}
