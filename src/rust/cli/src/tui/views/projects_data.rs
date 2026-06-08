//! Data types and SQLite fetching logic for the TUI project browser.
//!
//! Separated from the view module to keep both files under the 500-line limit
//! and to allow unit-testing data logic independently from rendering.

use std::collections::HashMap;

use crate::data::db::connect_readonly;
use crate::output::style::home_to_tilde;

/// Maximum watch folders to fetch per query.
const FETCH_LIMIT: i64 = 200;

/// A registered project for display in the TUI project list.
#[derive(Debug, Clone)]
pub struct ProjectRow {
    /// Watch folder ID (primary key).
    pub watch_id: String,
    /// Human-readable project name (last path component).
    pub name: String,
    /// Full path with `~` substitution.
    pub display_path: String,
    /// Whether the project is currently active (reference count > 0).
    pub is_active: bool,
    /// Number of indexed documents (queue items completed for this tenant).
    pub doc_count: i64,
    /// Number of pending/in-progress queue items for this tenant.
    pub queue_count: i64,
}

/// Full detail for a single project, shown in the popup.
#[derive(Debug, Clone)]
pub struct ProjectDetail {
    pub watch_id: String,
    pub tenant_id: String,
    pub name: String,
    pub display_path: String,
    pub collection: String,
    pub is_active: bool,
    pub is_paused: bool,
    pub is_archived: bool,
    pub git_remote_url: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub last_scan: Option<String>,
    /// Watch folders that are children of this project (submodules).
    pub sub_watches: Vec<String>,
    /// Queue item counts by status.
    pub queue_by_status: HashMap<String, i64>,
}

/// Build a human-readable status string from project flags.
pub fn build_status_text(detail: &ProjectDetail) -> String {
    let mut parts = Vec::new();
    parts.push(if detail.is_active {
        "Active"
    } else {
        "Inactive"
    });
    if detail.is_paused {
        parts.push("Paused");
    }
    if detail.is_archived {
        parts.push("Archived");
    }
    parts.join(", ")
}

/// Format a UTC timestamp for local display.
pub fn format_local_time(utc_str: &str) -> String {
    wqm_common::timestamp_fmt::format_local(utc_str)
}

/// Fetch all registered projects from SQLite.
pub fn fetch_project_rows() -> Vec<ProjectRow> {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let queue_counts = build_queue_counts(&conn);
    let doc_counts = build_doc_counts(&conn);

    let Ok(mut stmt) = conn.prepare(
        "SELECT watch_id, tenant_id, path, is_active \
         FROM watch_folders \
         WHERE parent_watch_id IS NULL \
         ORDER BY is_active DESC, path ASC \
         LIMIT ?1",
    ) else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map(rusqlite::params![FETCH_LIMIT], |row| {
        Ok((
            row.get::<_, String>(0)?, // watch_id
            row.get::<_, String>(1)?, // tenant_id
            row.get::<_, String>(2)?, // path
            row.get::<_, i64>(3)?,    // is_active
        ))
    }) else {
        return Vec::new();
    };

    rows.flatten()
        .map(|(watch_id, tenant_id, path, is_active)| {
            let name = path
                .rsplit('/')
                .find(|s| !s.is_empty())
                .unwrap_or(&watch_id)
                .to_string();
            let display_path = home_to_tilde(&path);
            let q_count = queue_counts.get(&tenant_id).copied().unwrap_or(0);
            let d_count = doc_counts.get(&tenant_id).copied().unwrap_or(0);

            ProjectRow {
                watch_id,
                name,
                display_path,
                is_active: is_active > 0,
                doc_count: d_count,
                queue_count: q_count,
            }
        })
        .collect()
}

/// Fetch full detail for a single project by watch_id.
pub fn fetch_project_detail(watch_id: &str) -> Option<ProjectDetail> {
    let conn = connect_readonly().ok()?;

    let mut stmt = conn
        .prepare(
            "SELECT watch_id, tenant_id, path, collection, is_active, \
             is_paused, is_archived, git_remote_url, \
             created_at, updated_at, last_scan \
             FROM watch_folders WHERE watch_id = ?1",
        )
        .ok()?;

    let detail = stmt
        .query_row(rusqlite::params![watch_id], |row| {
            let path: String = row.get(2)?;
            let name = path
                .rsplit('/')
                .find(|s| !s.is_empty())
                .unwrap_or("unknown")
                .to_string();

            Ok(ProjectDetail {
                watch_id: row.get(0)?,
                tenant_id: row.get(1)?,
                name,
                display_path: home_to_tilde(&path),
                collection: row.get(3)?,
                is_active: row.get::<_, i64>(4)? > 0,
                is_paused: row.get::<_, i64>(5)? != 0,
                is_archived: row.get::<_, i64>(6)? != 0,
                git_remote_url: row.get(7)?,
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
                last_scan: row.get(10)?,
                sub_watches: Vec::new(),
                queue_by_status: HashMap::new(),
            })
        })
        .ok()?;

    // Enrich with sub-watches
    let sub_watches = fetch_sub_watches(&conn, watch_id);
    let queue_by_status = fetch_queue_by_status(&conn, &detail.tenant_id);

    Some(ProjectDetail {
        sub_watches,
        queue_by_status,
        ..detail
    })
}

/// Build a map of tenant_id -> count of pending/in_progress queue items.
fn build_queue_counts(conn: &rusqlite::Connection) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, COUNT(*) FROM unified_queue \
         WHERE status IN ('pending', 'in_progress') \
         GROUP BY tenant_id",
    ) else {
        return map;
    };

    if let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }) {
        for r in rows.flatten() {
            map.insert(r.0, r.1);
        }
    }
    map
}

/// Build a map of tenant_id -> count of indexed documents.
///
/// Counts rows in `tracked_files` (the authoritative record of indexed files),
/// not completed queue items. Completed queue rows are garbage-collected by the
/// daemon's cleanup task, so counting them reported 0 for every project once the
/// queue drained.
fn build_doc_counts(conn: &rusqlite::Connection) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    let Ok(mut stmt) = conn.prepare(
        "SELECT wf.tenant_id, COUNT(tf.file_id) \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         GROUP BY wf.tenant_id",
    ) else {
        return map;
    };

    if let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }) {
        for r in rows.flatten() {
            map.insert(r.0, r.1);
        }
    }
    map
}

/// Fetch sub-watch paths for a given parent watch_id.
fn fetch_sub_watches(conn: &rusqlite::Connection, parent_id: &str) -> Vec<String> {
    let Ok(mut stmt) =
        conn.prepare("SELECT path FROM watch_folders WHERE parent_watch_id = ?1 ORDER BY path")
    else {
        return Vec::new();
    };

    stmt.query_map(rusqlite::params![parent_id], |row| row.get::<_, String>(0))
        .map(|rows| rows.flatten().map(|p| home_to_tilde(&p)).collect())
        .unwrap_or_default()
}

/// Fetch queue item counts grouped by status for a given tenant.
fn fetch_queue_by_status(conn: &rusqlite::Connection, tenant_id: &str) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    let Ok(mut stmt) = conn.prepare(
        "SELECT status, COUNT(*) FROM unified_queue \
         WHERE tenant_id = ?1 GROUP BY status",
    ) else {
        return map;
    };

    if let Ok(rows) = stmt.query_map(rusqlite::params![tenant_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }) {
        for r in rows.flatten() {
            map.insert(r.0, r.1);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_row_fields() {
        let row = ProjectRow {
            watch_id: "abc123def456".to_string(),
            name: "my-project".to_string(),
            display_path: "~/dev/my-project".to_string(),
            is_active: true,
            doc_count: 42,
            queue_count: 3,
        };
        assert_eq!(row.name, "my-project");
        assert!(row.is_active);
        assert_eq!(row.doc_count, 42);
        assert_eq!(row.queue_count, 3);
    }

    #[test]
    fn project_detail_fields() {
        let detail = ProjectDetail {
            watch_id: "w1".to_string(),
            tenant_id: "t1".to_string(),
            name: "test-proj".to_string(),
            display_path: "~/test-proj".to_string(),
            collection: "projects".to_string(),
            is_active: true,
            is_paused: false,
            is_archived: false,
            git_remote_url: Some("https://github.com/user/repo".to_string()),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T12:00:00Z".to_string(),
            last_scan: Some("2025-01-01T11:00:00Z".to_string()),
            sub_watches: vec!["~/test-proj/sub".to_string()],
            queue_by_status: HashMap::from([("pending".to_string(), 2), ("done".to_string(), 10)]),
        };
        assert_eq!(detail.name, "test-proj");
        assert!(detail.is_active);
        assert!(!detail.is_paused);
        assert_eq!(detail.sub_watches.len(), 1);
        assert_eq!(detail.queue_by_status.get("pending"), Some(&2));
    }

    #[test]
    fn project_detail_default_empty_collections() {
        let detail = ProjectDetail {
            watch_id: String::new(),
            tenant_id: String::new(),
            name: String::new(),
            display_path: String::new(),
            collection: "projects".to_string(),
            is_active: false,
            is_paused: false,
            is_archived: false,
            git_remote_url: None,
            created_at: String::new(),
            updated_at: String::new(),
            last_scan: None,
            sub_watches: Vec::new(),
            queue_by_status: HashMap::new(),
        };
        assert!(detail.sub_watches.is_empty());
        assert!(detail.queue_by_status.is_empty());
    }

    #[test]
    fn doc_counts_from_tracked_files_not_queue() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (watch_id TEXT PRIMARY KEY, tenant_id TEXT);
             CREATE TABLE tracked_files (file_id INTEGER PRIMARY KEY, watch_folder_id TEXT);
             INSERT INTO watch_folders (watch_id, tenant_id) VALUES ('w1', 't1');
             INSERT INTO watch_folders (watch_id, tenant_id) VALUES ('w2', 't2');
             INSERT INTO tracked_files (file_id, watch_folder_id) VALUES (1, 'w1');
             INSERT INTO tracked_files (file_id, watch_folder_id) VALUES (2, 'w1');
             INSERT INTO tracked_files (file_id, watch_folder_id) VALUES (3, 'w2');",
        )
        .unwrap();
        let counts = build_doc_counts(&conn);
        assert_eq!(counts.get("t1"), Some(&2));
        assert_eq!(counts.get("t2"), Some(&1));
    }

    #[test]
    fn build_status_text_variants() {
        let mut detail = ProjectDetail {
            watch_id: String::new(),
            tenant_id: String::new(),
            name: String::new(),
            display_path: String::new(),
            collection: "projects".to_string(),
            is_active: true,
            is_paused: false,
            is_archived: false,
            git_remote_url: None,
            created_at: String::new(),
            updated_at: String::new(),
            last_scan: None,
            sub_watches: Vec::new(),
            queue_by_status: HashMap::new(),
        };
        assert_eq!(build_status_text(&detail), "Active");

        detail.is_active = false;
        detail.is_paused = true;
        assert_eq!(build_status_text(&detail), "Inactive, Paused");

        detail.is_active = true;
        detail.is_paused = false;
        detail.is_archived = true;
        assert_eq!(build_status_text(&detail), "Active, Archived");
    }
}
