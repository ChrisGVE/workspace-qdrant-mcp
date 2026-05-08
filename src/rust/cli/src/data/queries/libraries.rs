//! Library queries — registered watch folders in the 'libraries' collection.

use anyhow::{Context, Result};
use rusqlite::Connection;

/// Basic library info from SQLite.
#[derive(Debug, Clone)]
pub struct LibraryInfo {
    pub watch_id: String,
    pub tenant_id: String,
    pub path: String,
    pub mode: String,
    pub enabled: bool,
    pub document_count: usize,
}

/// Get all registered libraries with document counts.
pub fn get_libraries(conn: &Connection) -> Result<Vec<LibraryInfo>> {
    let mut stmt = conn
        .prepare(
            "SELECT wf.watch_id, wf.tenant_id, wf.path, \
                    COALESCE(wf.library_mode, 'incremental'), \
                    COALESCE(wf.enabled, 1), \
                    COALESCE(tf_count.cnt, 0) \
             FROM watch_folders wf \
             LEFT JOIN ( \
                 SELECT watch_folder_id, COUNT(*) AS cnt \
                 FROM tracked_files GROUP BY watch_folder_id \
             ) tf_count ON tf_count.watch_folder_id = wf.watch_id \
             WHERE wf.collection = 'libraries' \
             ORDER BY wf.tenant_id",
        )
        .context("Failed to query libraries")?;

    let rows = stmt
        .query_map([], |row| {
            Ok(LibraryInfo {
                watch_id: row.get(0)?,
                tenant_id: row.get(1)?,
                path: row.get(2)?,
                mode: row.get(3)?,
                enabled: row.get::<_, i64>(4)? > 0,
                document_count: row.get(5)?,
            })
        })
        .context("Failed to read libraries")?;

    rows.collect::<Result<Vec<_>, _>>()
        .context("Failed to parse library rows")
}

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
    fn libraries_with_counts() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, library_mode)
             VALUES ('lib-docs', 'lib-docs', '/docs', 'libraries', 'sync');
             INSERT INTO tracked_files (file_id, watch_folder_id)
             VALUES (1, 'lib-docs');
             INSERT INTO tracked_files (file_id, watch_folder_id)
             VALUES (2, 'lib-docs');",
        )
        .unwrap();
        let libs = get_libraries(&conn).unwrap();
        assert_eq!(libs.len(), 1);
        assert_eq!(libs[0].document_count, 2);
        assert_eq!(libs[0].mode, "sync");
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
