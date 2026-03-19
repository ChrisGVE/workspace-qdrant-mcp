//! Unit tests for queue command utilities

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rusqlite::{params, Connection};

    use crate::commands::queue::formatters::{extract_object, format_relative_time, format_status};

    #[test]
    fn test_format_relative_time() {
        // Test with a timestamp 30 seconds ago
        let now = Utc::now();
        let timestamp = (now - chrono::Duration::seconds(30)).to_rfc3339();
        let result = format_relative_time(&timestamp);
        assert!(result.ends_with('s') || result.ends_with('m'));

        // Test with a timestamp 2 hours ago
        let timestamp = (now - chrono::Duration::hours(2)).to_rfc3339();
        let result = format_relative_time(&timestamp);
        assert!(result.ends_with('h'));

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

    // --- extract_object tests ---

    #[test]
    fn test_extract_object_file() {
        let payload = r#"{"file_path":"/home/user/project/src/main.rs"}"#;
        assert_eq!(extract_object("file", payload), "main.rs");
    }

    #[test]
    fn test_extract_object_file_nested_path() {
        let payload = r#"{"file_path":"/a/b/c/lib.rs","file_type":"code"}"#;
        assert_eq!(extract_object("file", payload), "lib.rs");
    }

    #[test]
    fn test_extract_object_file_trailing_slash() {
        // Edge case: file_path with trailing slash (shouldn't happen, but be robust)
        let payload = r#"{"file_path":"/a/b/c/"}"#;
        assert_eq!(extract_object("file", payload), "c");
    }

    #[test]
    fn test_extract_object_folder() {
        let payload = r#"{"folder_path":"/home/user/project/src"}"#;
        assert_eq!(extract_object("folder", payload), "src/");
    }

    #[test]
    fn test_extract_object_folder_trailing_slash() {
        let payload = r#"{"folder_path":"/home/user/project/src/"}"#;
        assert_eq!(extract_object("folder", payload), "src/");
    }

    #[test]
    fn test_extract_object_url() {
        let payload = r#"{"url":"https://docs.rs/tokio"}"#;
        assert_eq!(extract_object("url", payload), "https://docs.rs/tokio");
    }

    #[test]
    fn test_extract_object_website() {
        let payload = r#"{"url":"https://example.com","max_depth":3}"#;
        assert_eq!(extract_object("website", payload), "https://example.com");
    }

    #[test]
    fn test_extract_object_text_with_title() {
        let payload = r#"{"content":"long content here","title":"My Note"}"#;
        assert_eq!(extract_object("text", payload), "My Note");
    }

    #[test]
    fn test_extract_object_text_without_title() {
        let payload = r#"{"content":"short content","source_type":"cli"}"#;
        assert_eq!(extract_object("text", payload), "short content");
    }

    #[test]
    fn test_extract_object_text_truncated() {
        let long_content = "a".repeat(60);
        let payload = format!(r#"{{"content":"{}","source_type":"cli"}}"#, long_content);
        let result = extract_object("text", &payload);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 43); // 37 chars + "..."
    }

    #[test]
    fn test_extract_object_tenant_empty() {
        let payload = r#"{"some":"data"}"#;
        assert_eq!(extract_object("tenant", payload), "");
    }

    #[test]
    fn test_extract_object_collection_empty() {
        assert_eq!(extract_object("collection", r#"{}"#), "");
    }

    #[test]
    fn test_extract_object_unknown_type() {
        assert_eq!(extract_object("unknown", r#"{"foo":"bar"}"#), "");
    }

    #[test]
    fn test_extract_object_invalid_json() {
        assert_eq!(extract_object("file", "not json"), "");
    }

    #[test]
    fn test_extract_object_missing_field() {
        // file type but no file_path field
        assert_eq!(extract_object("file", r#"{"other":"value"}"#), "");
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
