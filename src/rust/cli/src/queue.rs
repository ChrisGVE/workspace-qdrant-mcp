//! Unified Queue Module for CLI (Task 37.11)
//!
//! Provides SQLite direct access for enqueueing items to the unified_queue
//! when the daemon is unavailable. This enables CLI fallback to the queue
//! instead of failing when daemon is down.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::Connection;
use sha2::{Sha256, Digest};
use uuid::Uuid;

/// Item types that can be enqueued (mirrors daemon's ItemType)
#[derive(Debug, Clone, Copy)]
pub enum ItemType {
    Content,
    File,
    Folder,
    Project,
    Library,
    DeleteTenant,
    DeleteDocument,
    Rename,
}

impl ItemType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ItemType::Content => "content",
            ItemType::File => "file",
            ItemType::Folder => "folder",
            ItemType::Project => "project",
            ItemType::Library => "library",
            ItemType::DeleteTenant => "delete_tenant",
            ItemType::DeleteDocument => "delete_document",
            ItemType::Rename => "rename",
        }
    }
}

/// Queue operations (mirrors daemon's QueueOperation)
#[derive(Debug, Clone, Copy)]
pub enum QueueOperation {
    Ingest,
    Update,
    Delete,
    Scan,
}

impl QueueOperation {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueOperation::Ingest => "ingest",
            QueueOperation::Update => "update",
            QueueOperation::Delete => "delete",
            QueueOperation::Scan => "scan",
        }
    }
}

/// Content payload for text ingestion
#[derive(Debug)]
pub struct ContentPayload {
    pub content: String,
    pub source_type: String,
    pub main_tag: Option<String>,
    pub full_tag: Option<String>,
}

impl ContentPayload {
    /// Serialize to JSON for storage
    pub fn to_json(&self) -> Result<String> {
        let json = serde_json::json!({
            "content": self.content,
            "source_type": self.source_type,
            "main_tag": self.main_tag,
            "full_tag": self.full_tag,
        });
        serde_json::to_string(&json).context("Failed to serialize content payload")
    }
}

/// File payload for file ingestion
#[derive(Debug)]
pub struct FilePayload {
    pub file_path: String,
    pub file_type: Option<String>,
}

impl FilePayload {
    /// Serialize to JSON for storage
    pub fn to_json(&self) -> Result<String> {
        let json = serde_json::json!({
            "file_path": self.file_path,
            "file_type": self.file_type,
        });
        serde_json::to_string(&json).context("Failed to serialize file payload")
    }
}

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
    pub fn connect() -> Result<Self> {
        let db_path = get_state_db_path()?;

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
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
        let payload_json = payload.to_json()?;

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
        let payload_json = payload.to_json()?;

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
        let now = Utc::now().to_rfc3339();

        // Generate idempotency key (must match Python and daemon implementations)
        let idempotency_key = generate_idempotency_key(
            item_type.as_str(),
            op.as_str(),
            tenant_id,
            collection,
            payload_json,
        );

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
                item_type.as_str(),
                op.as_str(),
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

/// Generate idempotency key matching Python/daemon implementation
/// Format: SHA256(item_type|op|tenant_id|collection|payload_json) truncated to 32 chars
fn generate_idempotency_key(
    item_type: &str,
    op: &str,
    tenant_id: &str,
    collection: &str,
    payload_json: &str,
) -> String {
    let input = format!("{}|{}|{}|{}|{}", item_type, op, tenant_id, collection, payload_json);
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let hash = hasher.finalize();
    // Convert to hex and take first 32 characters
    format!("{:x}", hash)[..32].to_string()
}

/// Get the path to the state database
fn get_state_db_path() -> Result<PathBuf> {
    // First check HOME-based path (Unix standard)
    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(format!("{}/.workspace-qdrant/state.db", home)));
    }

    // Fallback to dirs crate
    if let Some(data_dir) = dirs::data_local_dir() {
        return Ok(data_dir.join("workspace-qdrant").join("state.db"));
    }

    // Last resort
    Ok(PathBuf::from("/tmp/.workspace-qdrant/state.db"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idempotency_key_generation() {
        let key = generate_idempotency_key(
            "content",
            "ingest",
            "test-tenant",
            "test-collection",
            r#"{"content":"hello"}"#,
        );

        // Key should be 32 characters (hex)
        assert_eq!(key.len(), 32);

        // Same input should produce same key
        let key2 = generate_idempotency_key(
            "content",
            "ingest",
            "test-tenant",
            "test-collection",
            r#"{"content":"hello"}"#,
        );
        assert_eq!(key, key2);

        // Different input should produce different key
        let key3 = generate_idempotency_key(
            "content",
            "ingest",
            "test-tenant",
            "test-collection",
            r#"{"content":"world"}"#,
        );
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

        let json = payload.to_json().unwrap();
        assert!(json.contains("test content"));
        assert!(json.contains("cli"));
        assert!(json.contains("tag1"));
    }

    #[test]
    fn test_item_type_as_str() {
        assert_eq!(ItemType::Content.as_str(), "content");
        assert_eq!(ItemType::File.as_str(), "file");
        assert_eq!(ItemType::Project.as_str(), "project");
    }

    #[test]
    fn test_queue_operation_as_str() {
        assert_eq!(QueueOperation::Ingest.as_str(), "ingest");
        assert_eq!(QueueOperation::Update.as_str(), "update");
        assert_eq!(QueueOperation::Delete.as_str(), "delete");
    }
}
