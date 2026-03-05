//! Unit tests for queue command utilities

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rusqlite::{params, Connection};

    use crate::commands::queue::formatters::{format_relative_time, format_status};

    #[test]
    fn test_format_relative_time() {
        // Test with a timestamp 30 seconds ago
        let now = Utc::now();
        let timestamp = (now - chrono::Duration::seconds(30)).to_rfc3339();
        let result = format_relative_time(&timestamp);
        assert!(result.contains("s ago") || result.contains("0m ago"));

        // Test with a timestamp 2 hours ago
        let timestamp = (now - chrono::Duration::hours(2)).to_rfc3339();
        let result = format_relative_time(&timestamp);
        assert!(result.contains("h ago"));

        // Test with invalid timestamp
        let result = format_relative_time("invalid");
        assert_eq!(result, "unknown");
    }

    #[test]
    fn test_format_status() {
        // Just verify it doesn't panic
        let _ = format_status("pending");
        let _ = format_status("in_progress");
        let _ = format_status("done");
        let _ = format_status("failed");
        let _ = format_status("unknown");
    }

    /// Helper: create in-memory database with unified_queue schema
    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                branch TEXT,
                payload_json TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                last_error TEXT,
                leased_by TEXT,
                lease_expires_at TEXT,
                error_message TEXT,
                last_error_at TEXT,
                worker_id TEXT,
                lease_until TEXT
            )",
        )
        .unwrap();
        conn
    }

    /// Helper: insert a test queue item
    fn insert_test_item(conn: &Connection, id: &str, item_type: &str, op: &str, status: &str) {
        conn.execute(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?, ?, ?, ?, 'test-tenant', 'projects', ?, '{}', \
              datetime('now'), datetime('now'))",
            params![id, format!("key_{}", id), item_type, op, status],
        )
        .unwrap();
    }

    #[test]
    fn test_remove_exact_match() {
        let conn = setup_test_db();
        insert_test_item(&conn, "abc123def456", "file", "add", "pending");

        // Verify item exists
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'abc123def456'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        // Delete using exact match
        conn.execute(
            "DELETE FROM unified_queue WHERE queue_id = ?",
            params!["abc123def456"],
        )
        .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM unified_queue", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_remove_prefix_match() {
        let conn = setup_test_db();
        insert_test_item(&conn, "abc123def456", "file", "add", "done");

        // Find by prefix
        let prefix = "abc123%";
        let resolved_id: String = conn
            .query_row(
                "SELECT queue_id FROM unified_queue \
                 WHERE queue_id = ? OR queue_id LIKE ?",
                params!["abc123", prefix],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved_id, "abc123def456");

        // Delete the resolved item
        let deleted = conn
            .execute(
                "DELETE FROM unified_queue WHERE queue_id = ?",
                params![&resolved_id],
            )
            .unwrap();
        assert_eq!(deleted, 1);
    }

    #[test]
    fn test_remove_not_found() {
        let conn = setup_test_db();
        insert_test_item(&conn, "abc123def456", "file", "add", "pending");

        let prefix = "xyz%";
        let result = conn.query_row(
            "SELECT queue_id FROM unified_queue \
             WHERE queue_id = ? OR queue_id LIKE ?",
            params!["xyz", prefix],
            |r| r.get::<_, String>(0),
        );
        assert!(matches!(result, Err(rusqlite::Error::QueryReturnedNoRows)));
    }

    #[test]
    fn test_remove_no_cascade_to_other_tables() {
        let conn = setup_test_db();
        // Create tracked_files table to verify no cascade
        conn.execute_batch(
            "CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                file_path TEXT NOT NULL
            )",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO tracked_files (tenant_id, file_path) \
             VALUES ('test-tenant', '/foo/bar.rs')",
            [],
        )
        .unwrap();

        insert_test_item(&conn, "abc123def456", "file", "add", "pending");
        conn.execute(
            "DELETE FROM unified_queue WHERE queue_id = ?",
            params!["abc123def456"],
        )
        .unwrap();

        // tracked_files should be unaffected
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM tracked_files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
}
