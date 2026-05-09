//! Library queries — registered watch folders in the 'libraries' collection.

use anyhow::{Context, Result};
use rusqlite::Connection;

/// Count of collections known to have data (from watch_folders).
pub fn get_active_collection_count(conn: &Connection) -> Result<usize> {
    conn.query_row(
        "SELECT COUNT(DISTINCT collection) FROM watch_folders",
        [],
        |row| row.get(0),
    )
    .context("Failed to count collections")
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
    fn active_collection_count() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w1', 't1', '/proj1', 'projects', 1);
             INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w2', 't2', '/proj2', 'projects', 1);
             INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('lib1', 'lib-docs', '/docs', 'libraries', 1);
             INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('lib2', 'lib-api', '/api', 'libraries', 0);",
        )
        .unwrap();
        // COUNT(DISTINCT collection): 'projects' and 'libraries' = 2
        assert_eq!(get_active_collection_count(&conn).unwrap(), 2);
    }
}
