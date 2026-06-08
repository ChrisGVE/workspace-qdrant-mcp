//! Data types and SQLite fetching logic for the TUI queue browser.
//!
//! Separated from the view module to keep both files under the 500-line limit
//! and to allow unit-testing data logic independently from rendering.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::commands::queue::formatters::extract_object;
use crate::data::db::connect_readonly;
use crate::output::style::short_id;

/// Maximum items to fetch per query.
const FETCH_LIMIT: i64 = 200;

/// Status filter for the queue browser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusFilter {
    All,
    Pending,
    InProgress,
    Failed,
    Done,
}

impl StatusFilter {
    /// Cycle to the next filter value.
    pub fn next(self) -> Self {
        match self {
            Self::All => Self::Pending,
            Self::Pending => Self::InProgress,
            Self::InProgress => Self::Failed,
            Self::Failed => Self::Done,
            Self::Done => Self::All,
        }
    }

    /// Display label for the header.
    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Pending => "Pending",
            Self::InProgress => "In Progress",
            Self::Failed => "Failed",
            Self::Done => "Done",
        }
    }

    /// SQL value for the WHERE clause, or `None` for no filter.
    fn sql_value(self) -> Option<&'static str> {
        match self {
            Self::All => None,
            Self::Pending => Some("pending"),
            Self::InProgress => Some("in_progress"),
            Self::Failed => Some("failed"),
            Self::Done => Some("done"),
        }
    }
}

/// A single queue item ready for display in the TUI list.
#[derive(Debug, Clone)]
pub struct QueueRow {
    /// Full queue ID (for detail lookup).
    pub queue_id: String,
    /// Shortened ID for the list column.
    pub short_id: String,
    /// Resolved project name.
    pub project: String,
    /// Extracted object name (file name, URL, etc.).
    pub object: String,
    /// Item type (file, folder, url, etc.).
    pub item_type: String,
    /// Operation (add, update, delete, etc.).
    pub op: String,
    /// Raw status string for color coding.
    pub status: String,
    /// Human-readable relative age.
    pub age: String,
    /// Tenant kind: 'P' (project), 'L' (library), or '?' (unknown).
    pub kind: char,
}

/// Full detail of a single queue item for the popup view.
#[derive(Debug, Clone)]
pub struct QueueDetail {
    pub queue_id: String,
    pub idempotency_key: String,
    pub item_type: String,
    pub op: String,
    pub collection: String,
    pub status: String,
    pub project: String,
    pub tenant_id: String,
    pub object: String,
    pub payload_json: String,
    pub error_message: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub retry_count: i32,
}

/// Fetch queue rows from SQLite, applying the given status filter.
pub fn fetch_queue_rows(filter: StatusFilter) -> Vec<QueueRow> {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let tenant_names = build_tenant_name_map(&conn);
    let tenant_kinds = build_tenant_kind_map(&conn);

    let (query, params_vec) = build_query(filter);
    let params_slice: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

    let Ok(mut stmt) = conn.prepare(&query) else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map(params_slice.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?, // queue_id
            row.get::<_, String>(1)?, // item_type
            row.get::<_, String>(2)?, // op
            row.get::<_, String>(3)?, // status
            row.get::<_, String>(4)?, // created_at
            row.get::<_, String>(5)?, // tenant_id
            row.get::<_, String>(6)?, // payload_json
        ))
    }) else {
        return Vec::new();
    };

    rows.flatten()
        .map(
            |(queue_id, item_type, op, status, created_at, tenant_id, payload_json)| QueueRow {
                short_id: short_id(&queue_id),
                project: resolve_project_name(&tenant_id, &tenant_names),
                object: extract_object(&item_type, &payload_json),
                age: format_relative_time(&created_at),
                kind: tenant_kinds.get(&tenant_id).copied().unwrap_or('?'),
                queue_id,
                item_type,
                op,
                status,
            },
        )
        .collect()
}

/// Fetch full detail for a single queue item by its queue_id.
pub fn fetch_queue_detail(queue_id: &str) -> Option<QueueDetail> {
    let conn = connect_readonly().ok()?;
    let tenant_names = build_tenant_name_map(&conn);

    let mut stmt = conn
        .prepare(
            "SELECT queue_id, idempotency_key, item_type, op, collection, status, \
             tenant_id, COALESCE(payload_json, '{}'), error_message, \
             created_at, updated_at, retry_count \
             FROM unified_queue WHERE queue_id = ?1",
        )
        .ok()?;

    stmt.query_row(rusqlite::params![queue_id], |row| {
        let item_type: String = row.get(2)?;
        let tenant_id: String = row.get(6)?;
        let payload_json: String = row.get(7)?;
        let project = resolve_project_name(&tenant_id, &tenant_names);
        let object = extract_object(&item_type, &payload_json);

        Ok(QueueDetail {
            queue_id: row.get(0)?,
            idempotency_key: row.get(1)?,
            item_type,
            op: row.get(3)?,
            collection: row.get(4)?,
            status: row.get(5)?,
            tenant_id,
            project,
            object,
            payload_json,
            error_message: row.get(8)?,
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
            retry_count: row.get(11)?,
        })
    })
    .ok()
}

/// Build the SELECT query and parameters for the queue browser.
fn build_query(filter: StatusFilter) -> (String, Vec<Box<dyn rusqlite::ToSql>>) {
    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    let where_clause = if let Some(status) = filter.sql_value() {
        params_vec.push(Box::new(status.to_string()));
        "WHERE status = ?".to_string()
    } else {
        String::new()
    };

    let query = format!(
        "SELECT queue_id, item_type, op, status, created_at, tenant_id, \
         COALESCE(payload_json, '{{}}') \
         FROM unified_queue {} ORDER BY created_at DESC LIMIT ?",
        where_clause
    );
    params_vec.push(Box::new(FETCH_LIMIT));

    (query, params_vec)
}

/// Build a mapping from tenant_id to human-readable project name.
fn build_tenant_name_map(conn: &rusqlite::Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut name_count: HashMap<String, usize> = HashMap::new();
    let mut entries: Vec<(String, String)> = Vec::new();

    if let Ok(mut stmt) =
        conn.prepare("SELECT tenant_id, path FROM watch_folders WHERE parent_watch_id IS NULL")
    {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for r in rows.flatten() {
                let name =
                    r.1.rsplit('/')
                        .find(|s| !s.is_empty())
                        .unwrap_or(&r.0)
                        .to_string();
                *name_count.entry(name.clone()).or_default() += 1;
                entries.push((r.0, name));
            }
        }
    }

    for (tenant_id, name) in entries {
        let display = if name_count.get(&name).copied().unwrap_or(0) > 1 {
            format!("{} ({})", name, short_id(&tenant_id))
        } else {
            name
        };
        map.insert(tenant_id, display);
    }

    map
}

/// Map each tenant_id to its kind: 'P' (project) or 'L' (library).
fn build_tenant_kind_map(conn: &rusqlite::Connection) -> HashMap<String, char> {
    let mut map = HashMap::new();
    if let Ok(mut stmt) = conn
        .prepare("SELECT tenant_id, collection FROM watch_folders WHERE parent_watch_id IS NULL")
    {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            for (tid, collection) in rows.flatten() {
                let kind = if collection == "libraries" { 'L' } else { 'P' };
                map.insert(tid, kind);
            }
        }
    }
    map
}

/// Resolve a tenant_id to a project name, falling back to a short ID.
fn resolve_project_name(tenant_id: &str, tenant_names: &HashMap<String, String>) -> String {
    tenant_names
        .get(tenant_id)
        .cloned()
        .unwrap_or_else(|| short_id(tenant_id))
}

/// Format a UTC ISO timestamp as a human-readable relative time string.
fn format_relative_time(timestamp_str: &str) -> String {
    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
        let now = Utc::now();
        let duration = now.signed_duration_since(dt.with_timezone(&Utc));
        let secs = duration.num_seconds();

        if secs < 0 {
            return "future".to_string();
        }
        if secs < 60 {
            format!("{}s ago", secs)
        } else if secs < 3600 {
            format!("{}m ago", secs / 60)
        } else if secs < 86400 {
            format!("{}h ago", secs / 3600)
        } else {
            format!("{}d ago", secs / 86400)
        }
    } else {
        "unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_filter_cycle() {
        let f = StatusFilter::All;
        assert_eq!(f.next(), StatusFilter::Pending);
        assert_eq!(f.next().next(), StatusFilter::InProgress);
        assert_eq!(f.next().next().next(), StatusFilter::Failed);
        assert_eq!(f.next().next().next().next(), StatusFilter::Done);
        assert_eq!(f.next().next().next().next().next(), StatusFilter::All);
    }

    #[test]
    fn status_filter_labels() {
        assert_eq!(StatusFilter::All.label(), "All");
        assert_eq!(StatusFilter::Pending.label(), "Pending");
        assert_eq!(StatusFilter::InProgress.label(), "In Progress");
        assert_eq!(StatusFilter::Failed.label(), "Failed");
        assert_eq!(StatusFilter::Done.label(), "Done");
    }

    #[test]
    fn status_filter_sql_values() {
        assert_eq!(StatusFilter::All.sql_value(), None);
        assert_eq!(StatusFilter::Pending.sql_value(), Some("pending"));
        assert_eq!(StatusFilter::InProgress.sql_value(), Some("in_progress"));
        assert_eq!(StatusFilter::Failed.sql_value(), Some("failed"));
        assert_eq!(StatusFilter::Done.sql_value(), Some("done"));
    }

    #[test]
    fn build_query_no_filter() {
        let (query, params) = build_query(StatusFilter::All);
        assert!(!query.contains("WHERE"));
        assert_eq!(params.len(), 1); // just LIMIT
    }

    #[test]
    fn build_query_with_filter() {
        let (query, params) = build_query(StatusFilter::Failed);
        assert!(query.contains("WHERE status = ?"));
        assert_eq!(params.len(), 2); // status + LIMIT
    }

    #[test]
    fn format_relative_time_recent() {
        let now = Utc::now();
        let ts = now.to_rfc3339();
        let result = format_relative_time(&ts);
        assert!(result.contains("s ago") || result == "0s ago");
    }

    #[test]
    fn format_relative_time_invalid() {
        assert_eq!(format_relative_time("not-a-date"), "unknown");
    }

    #[test]
    fn queue_row_fields() {
        let row = QueueRow {
            queue_id: "abc123".to_string(),
            short_id: "abc1".to_string(),
            project: "my-project".to_string(),
            object: "main.rs".to_string(),
            item_type: "file".to_string(),
            op: "add".to_string(),
            status: "pending".to_string(),
            age: "5m ago".to_string(),
            kind: 'P',
        };
        assert_eq!(row.short_id, "abc1");
        assert_eq!(row.status, "pending");
    }
}
