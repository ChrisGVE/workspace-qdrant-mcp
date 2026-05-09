//! Project queries — registered watch folders in the 'projects' collection.

use anyhow::{Context, Result};
use rusqlite::Connection;

/// Basic project info from SQLite `watch_folders`.
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub tenant_id: String,
    pub path: String,
    pub is_active: bool,
    pub created_at: Option<String>,
    pub last_scan: Option<String>,
    pub last_activity_at: Option<String>,
}

/// Get all registered projects (parent watch folders in 'projects' collection).
pub fn get_projects(conn: &Connection) -> Result<Vec<ProjectInfo>> {
    let mut stmt = conn
        .prepare(
            "SELECT tenant_id, path, COALESCE(is_active, 0), \
                    created_at, last_scan, last_activity_at \
             FROM watch_folders \
             WHERE parent_watch_id IS NULL AND collection = 'projects' \
             ORDER BY is_active DESC, path ASC",
        )
        .context("Failed to query projects")?;

    let rows = stmt
        .query_map([], |row| {
            Ok(ProjectInfo {
                tenant_id: row.get(0)?,
                path: row.get(1)?,
                is_active: row.get::<_, i64>(2)? > 0,
                created_at: row.get(3)?,
                last_scan: row.get(4)?,
                last_activity_at: row.get(5)?,
            })
        })
        .context("Failed to read projects")?;

    rows.collect::<Result<Vec<_>, _>>()
        .context("Failed to parse project rows")
}

/// Count active projects (enabled watch folders).
pub fn get_active_project_count(conn: &Connection) -> Result<usize> {
    conn.query_row(
        "SELECT COUNT(*) FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects' AND is_active > 0",
        [],
        |row| row.get(0),
    )
    .context("Failed to count active projects")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                tenant_id TEXT,
                path TEXT,
                collection TEXT,
                parent_watch_id TEXT,
                is_active INTEGER DEFAULT 1,
                enabled INTEGER DEFAULT 1,
                library_mode TEXT,
                is_paused INTEGER DEFAULT 0,
                is_archived INTEGER DEFAULT 0,
                git_remote_url TEXT,
                created_at TEXT,
                updated_at TEXT,
                last_scan TEXT,
                last_activity_at TEXT,
                follow_symlinks INTEGER DEFAULT 0,
                cleanup_on_disable INTEGER DEFAULT 0
            );
            CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY,
                watch_folder_id TEXT,
                file_path TEXT,
                language TEXT,
                chunk_count INTEGER DEFAULT 0,
                needs_reconcile INTEGER DEFAULT 0,
                reconcile_reason TEXT,
                tenant_id TEXT,
                collection TEXT
            );
            CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT,
                item_type TEXT,
                op TEXT,
                collection TEXT,
                status TEXT,
                tenant_id TEXT,
                branch TEXT,
                payload_json TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                lease_until TEXT,
                worker_id TEXT,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                last_error_at TEXT,
                file_path TEXT
            );",
        )
        .unwrap();
        conn
    }

    #[test]
    fn active_projects_count() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w1', 't1', '/proj1', 'projects', 1);
             INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w2', 't2', '/proj2', 'projects', 0);
             INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w3', 't3', '/proj3', 'projects', 1);",
        )
        .unwrap();
        assert_eq!(get_active_project_count(&conn).unwrap(), 2);
        assert_eq!(get_projects(&conn).unwrap().len(), 3);
    }

    #[test]
    fn projects_with_null_fields() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active, last_activity_at, created_at)
             VALUES ('w1', 't1', '/proj1', 'projects', 1, NULL, NULL);",
        )
        .unwrap();
        let projects = get_projects(&conn).unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].tenant_id, "t1");
        assert!(projects[0].created_at.is_none());
        assert!(projects[0].last_activity_at.is_none());
    }
}
