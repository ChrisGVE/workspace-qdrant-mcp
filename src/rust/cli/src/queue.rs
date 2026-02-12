//! Unified Queue Module for CLI (Task 37.11)
//!
//! Provides SQLite direct access for enqueueing items to the unified_queue
//! when the daemon is unavailable. This enables CLI fallback to the queue
//! instead of failing when daemon is down.
//!
//! Note: Some types and methods are infrastructure for future gRPC integration
//! and queue inspection commands. They are tested but not yet used in the main CLI.

#![allow(dead_code)]

use anyhow::{Context, Result};
use wqm_common::timestamps;
use rusqlite::Connection;
use uuid::Uuid;

use crate::config::get_database_path;

// Re-export canonical types from wqm-common
pub use wqm_common::queue_types::{ItemType, QueueOperation};
pub use wqm_common::payloads::{ContentPayload, FilePayload};

/// Result of enqueue operation
#[derive(Debug)]
pub struct EnqueueResult {
    pub queue_id: String,
    pub idempotency_key: String,
    pub was_duplicate: bool,
}

/// Unified queue client for CLI
pub struct UnifiedQueueClient {
    conn: Connection,
}

impl UnifiedQueueClient {
    /// Connect to the state database
    ///
    /// Creates the database directory if needed (for write operations).
    /// Returns error with helpful message if database doesn't exist.
    pub fn connect() -> Result<Self> {
        let db_path = get_database_path()
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Ensure parent directory exists (needed for first write)
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }

        // Check if database exists - if not, provide helpful error
        if !db_path.exists() {
            anyhow::bail!(
                "Database not found at {}. Run daemon first: wqm service start",
                db_path.display()
            );
        }

        let conn = Connection::open(&db_path)
            .context(format!("Failed to open state database at {:?}", db_path))?;

        // Enable WAL mode for better concurrency
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .context("Failed to set SQLite pragmas")?;

        Ok(Self { conn })
    }

    /// Enqueue a content item (text ingestion)
    pub fn enqueue_content(
        &self,
        tenant_id: &str,
        collection: &str,
        payload: &ContentPayload,
        priority: i32,
        branch: &str,
    ) -> Result<EnqueueResult> {
        let payload_json = serde_json::to_string(payload)
            .context("Failed to serialize content payload")?;

        self.enqueue(
            ItemType::Content,
            QueueOperation::Ingest,
            tenant_id,
            collection,
            &payload_json,
            priority,
            branch,
            None,
        )
    }

    /// Enqueue a file item
    pub fn enqueue_file(
        &self,
        tenant_id: &str,
        collection: &str,
        payload: &FilePayload,
        op: QueueOperation,
        priority: i32,
        branch: &str,
    ) -> Result<EnqueueResult> {
        let payload_json = serde_json::to_string(payload)
            .context("Failed to serialize file payload")?;

        self.enqueue(
            ItemType::File,
            op,
            tenant_id,
            collection,
            &payload_json,
            priority,
            branch,
            None,
        )
    }

    /// Core enqueue method
    pub fn enqueue(
        &self,
        item_type: ItemType,
        op: QueueOperation,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        priority: i32,
        branch: &str,
        metadata: Option<&str>,
    ) -> Result<EnqueueResult> {
        let queue_id = Uuid::new_v4().to_string();
        let now = timestamps::now_utc();

        // Generate idempotency key using the canonical wqm-common implementation
        // (must match daemon and Python implementations)
        let idempotency_key = wqm_common::hashing::generate_idempotency_key(
            item_type,
            op,
            tenant_id,
            collection,
            payload_json,
        ).context("Failed to generate idempotency key")?;

        let item_type_str = item_type.to_string();
        let op_str = op.to_string();

        // Try to insert, checking for duplicates via idempotency_key
        let result = self.conn.execute(
            r#"
            INSERT OR IGNORE INTO unified_queue (
                queue_id, idempotency_key, item_type, op, tenant_id, collection,
                priority, status, branch, payload_json, metadata,
                created_at, updated_at, retry_count, max_retries
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending', ?8, ?9, ?10, ?11, ?11, 0, 3)
            "#,
            rusqlite::params![
                &queue_id,
                &idempotency_key,
                &item_type_str,
                &op_str,
                tenant_id,
                collection,
                priority,
                branch,
                payload_json,
                metadata,
                &now,
            ],
        ).context("Failed to insert into unified_queue")?;

        let was_duplicate = result == 0;

        Ok(EnqueueResult {
            queue_id,
            idempotency_key,
            was_duplicate,
        })
    }

    /// Get queue statistics
    pub fn get_stats(&self) -> Result<QueueStats> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT status, COUNT(*) as count
            FROM unified_queue
            GROUP BY status
            "#
        )?;

        let mut pending = 0;
        let mut in_progress = 0;
        let mut done = 0;
        let mut failed = 0;

        let rows = stmt.query_map([], |row| {
            let status: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((status, count))
        })?;

        for row in rows {
            let (status, count) = row?;
            match status.as_str() {
                "pending" => pending = count,
                "in_progress" => in_progress = count,
                "done" => done = count,
                "failed" => failed = count,
                _ => {}
            }
        }

        Ok(QueueStats {
            pending,
            in_progress,
            done,
            failed,
        })
    }

    /// Get queue statistics grouped by item type (Task 37.37)
    pub fn get_stats_by_type(&self) -> Result<std::collections::HashMap<String, QueueStats>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT item_type, status, COUNT(*) as count
            FROM unified_queue
            GROUP BY item_type, status
            "#
        )?;

        let mut result: std::collections::HashMap<String, QueueStats> = std::collections::HashMap::new();

        let rows = stmt.query_map([], |row| {
            let item_type: String = row.get(0)?;
            let status: String = row.get(1)?;
            let count: i64 = row.get(2)?;
            Ok((item_type, status, count))
        })?;

        for row in rows {
            let (item_type, status, count) = row?;
            let stats = result.entry(item_type).or_insert(QueueStats::default());
            match status.as_str() {
                "pending" => stats.pending = count,
                "in_progress" => stats.in_progress = count,
                "done" => stats.done = count,
                "failed" => stats.failed = count,
                _ => {}
            }
        }

        Ok(result)
    }
}

/// Queue statistics
#[derive(Debug, Default)]
pub struct QueueStats {
    pub pending: i64,
    pub in_progress: i64,
    pub done: i64,
    pub failed: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idempotency_key_generation() {
        // Use the canonical wqm-common implementation
        let key = wqm_common::hashing::generate_idempotency_key(
            ItemType::Content,
            QueueOperation::Ingest,
            "test-tenant",
            "test-collection",
            r#"{"content":"hello"}"#,
        ).unwrap();

        // Key should be 32 characters (hex)
        assert_eq!(key.len(), 32);

        // Same input should produce same key
        let key2 = wqm_common::hashing::generate_idempotency_key(
            ItemType::Content,
            QueueOperation::Ingest,
            "test-tenant",
            "test-collection",
            r#"{"content":"hello"}"#,
        ).unwrap();
        assert_eq!(key, key2);

        // Different input should produce different key
        let key3 = wqm_common::hashing::generate_idempotency_key(
            ItemType::Content,
            QueueOperation::Ingest,
            "test-tenant",
            "test-collection",
            r#"{"content":"world"}"#,
        ).unwrap();
        assert_ne!(key, key3);
    }

    #[test]
    fn test_content_payload_serialization() {
        let payload = ContentPayload {
            content: "test content".to_string(),
            source_type: "cli".to_string(),
            main_tag: Some("tag1".to_string()),
            full_tag: None,
        };

        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("test content"));
        assert!(json.contains("cli"));
        assert!(json.contains("tag1"));
    }

    #[test]
    fn test_item_type_display() {
        assert_eq!(ItemType::Content.to_string(), "content");
        assert_eq!(ItemType::File.to_string(), "file");
        assert_eq!(ItemType::Project.to_string(), "project");
    }

    #[test]
    fn test_queue_operation_display() {
        assert_eq!(QueueOperation::Ingest.to_string(), "ingest");
        assert_eq!(QueueOperation::Update.to_string(), "update");
        assert_eq!(QueueOperation::Delete.to_string(), "delete");
    }
}
