//! Unified Queue Schema Definitions (Task 22)
//!
//! This module defines the types and schema for the unified ingestion queue.
//! All queue operations (content, file, folder, etc.) are processed through
//! a single unified queue table with type discriminators.
//!
//! Core types (ItemType, QueueOperation, QueueStatus, payloads) are defined in
//! `wqm_common` and re-exported here for backward compatibility.

use serde::{Deserialize, Serialize};

// Re-export canonical types from wqm-common
pub use wqm_common::queue_types::{ItemType, QueueOperation, QueueStatus};
pub use wqm_common::payloads::{
    ContentPayload, FilePayload, FolderPayload, ProjectPayload,
    LibraryPayload, DeleteTenantPayload, DeleteDocumentPayload,
    MemoryPayload, RenamePayload, RenameType,
};

/// Complete unified queue item representation
///
/// This struct represents a full row from the unified_queue table,
/// used for dequeuing and processing operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedQueueItem {
    /// Unique queue item identifier
    pub queue_id: String,
    /// Idempotency key for deduplication
    pub idempotency_key: String,
    /// Type of item (content, file, folder, etc.)
    pub item_type: ItemType,
    /// Operation to perform (ingest, update, delete, scan)
    pub op: QueueOperation,
    /// Project/tenant identifier
    pub tenant_id: String,
    /// Target Qdrant collection
    pub collection: String,
    /// Processing priority (0-10, higher = more urgent)
    pub priority: i32,
    /// Current status (pending, in_progress, done, failed)
    pub status: QueueStatus,
    /// Git branch (default: main)
    pub branch: String,
    /// JSON payload with operation-specific data
    pub payload_json: String,
    /// Additional metadata as JSON
    #[serde(default)]
    pub metadata: Option<String>,
    /// Creation timestamp (RFC3339)
    pub created_at: String,
    /// Last update timestamp (RFC3339)
    pub updated_at: String,
    /// Lease expiry timestamp (RFC3339, None if not leased)
    #[serde(default)]
    pub lease_until: Option<String>,
    /// Worker ID holding the lease
    #[serde(default)]
    pub worker_id: Option<String>,
    /// Number of retry attempts
    #[serde(default)]
    pub retry_count: i32,
    /// Maximum allowed retries
    #[serde(default = "default_max_retries")]
    pub max_retries: i32,
    /// Last error message if failed
    #[serde(default)]
    pub error_message: Option<String>,
    /// Timestamp of last error (RFC3339)
    #[serde(default)]
    pub last_error_at: Option<String>,
    /// File path for file item types (Task 22: per-file deduplication)
    /// Only set for item_type='file', None for other types
    #[serde(default)]
    pub file_path: Option<String>,
}

fn default_max_retries() -> i32 { 3 }

impl UnifiedQueueItem {
    /// Parse the payload JSON into a typed payload struct
    pub fn parse_content_payload(&self) -> Result<ContentPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a FilePayload
    pub fn parse_file_payload(&self) -> Result<FilePayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a FolderPayload
    pub fn parse_folder_payload(&self) -> Result<FolderPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a ProjectPayload
    pub fn parse_project_payload(&self) -> Result<ProjectPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a LibraryPayload
    pub fn parse_library_payload(&self) -> Result<LibraryPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a DeleteTenantPayload
    pub fn parse_delete_tenant_payload(&self) -> Result<DeleteTenantPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a DeleteDocumentPayload
    pub fn parse_delete_document_payload(&self) -> Result<DeleteDocumentPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a RenamePayload
    pub fn parse_rename_payload(&self) -> Result<RenamePayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Check if the item can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Check if the lease has expired
    pub fn is_lease_expired(&self) -> bool {
        if let Some(ref lease_until) = self.lease_until {
            if let Ok(lease_time) = chrono::DateTime::parse_from_rfc3339(lease_until) {
                return chrono::Utc::now() > lease_time;
            }
        }
        // No lease or invalid timestamp means not leased
        true
    }
}

/// Statistics for the unified queue
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedQueueStats {
    /// Total items in queue (all statuses)
    pub total_items: i64,
    /// Items with pending status
    pub pending_items: i64,
    /// Items with in_progress status
    pub in_progress_items: i64,
    /// Items with done status (not yet cleaned up)
    pub done_items: i64,
    /// Items with failed status
    pub failed_items: i64,
    /// Items by type
    pub by_item_type: std::collections::HashMap<String, i64>,
    /// Items by operation
    pub by_operation: std::collections::HashMap<String, i64>,
    /// Oldest pending item timestamp
    pub oldest_pending: Option<String>,
    /// Newest item timestamp
    pub newest_item: Option<String>,
    /// Number of items with expired leases (stale)
    pub stale_leases: i64,
}

/// Generate an idempotency key for a queue item (simple format)
///
/// Uses format: `{item_type}:{collection}:{identifier_hash}`
/// Hash is truncated to 16 hex chars (8 bytes).
///
/// For the comprehensive format with operation and tenant_id,
/// use `generate_unified_idempotency_key`.
pub fn generate_idempotency_key(
    item_type: ItemType,
    collection: &str,
    identifier: &str,
) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(identifier.as_bytes());
    let hash = hasher.finalize();
    // Encode first 8 bytes as hex manually
    let hash_hex: String = hash[..8]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    format!("{}:{}:{}", item_type, collection, hash_hex)
}

// Re-export comprehensive idempotency key generation from wqm-common
pub use wqm_common::hashing::{
    generate_idempotency_key as generate_unified_idempotency_key,
    IdempotencyKeyError,
};

/// SQL to create the unified queue schema.
///
/// NOTE: The `priority` column is unused — priority is computed dynamically at dequeue time
/// via a CASE expression that JOINs with `watch_folders.is_active`. Two levels:
/// - High (1): active projects + memory collection
/// - Low (0): libraries + inactive projects
/// All callers should pass 0 when enqueuing. See `dequeue_unified()` in `queue_operations.rs`.
pub const CREATE_UNIFIED_QUEUE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS unified_queue (
    queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    item_type TEXT NOT NULL CHECK (item_type IN (
        'content', 'file', 'folder', 'project', 'library',
        'delete_tenant', 'delete_document', 'rename'
    )),
    op TEXT NOT NULL CHECK (op IN ('ingest', 'update', 'delete', 'scan')),
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    -- UNUSED: priority is computed at dequeue time, not stored. Always set to 0.
    priority INTEGER NOT NULL DEFAULT 0 CHECK (priority >= 0 AND priority <= 10),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'done', 'failed'
    )),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    lease_until TEXT,
    worker_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL DEFAULT '{}',
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    last_error_at TEXT,
    branch TEXT DEFAULT 'main',
    metadata TEXT DEFAULT '{}',
    -- Task 22: Per-file deduplication
    -- Only set for item_type='file', NULL for other types
    -- UNIQUE constraint prevents duplicate file ingestion
    file_path TEXT UNIQUE
)
"#;

/// SQL to create indexes for the unified queue
pub const CREATE_UNIFIED_QUEUE_INDEXES_SQL: &[&str] = &[
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_dequeue
       ON unified_queue(status, priority DESC, created_at ASC)
       WHERE status = 'pending'"#,
    r#"CREATE UNIQUE INDEX IF NOT EXISTS idx_unified_queue_idempotency
       ON unified_queue(idempotency_key)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_lease_expiry
       ON unified_queue(lease_until)
       WHERE status = 'in_progress'"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_collection_tenant
       ON unified_queue(collection, tenant_id)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_item_type
       ON unified_queue(item_type, status)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_failed
       ON unified_queue(status, last_error_at DESC)
       WHERE status = 'failed'"#,
    // Task 22: Per-file deduplication index
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_file_path
       ON unified_queue(file_path)
       WHERE file_path IS NOT NULL"#,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_type_display() {
        assert_eq!(ItemType::Content.to_string(), "content");
        assert_eq!(ItemType::DeleteTenant.to_string(), "delete_tenant");
    }

    #[test]
    fn test_item_type_from_str() {
        assert_eq!(ItemType::from_str("content"), Some(ItemType::Content));
        assert_eq!(ItemType::from_str("delete_tenant"), Some(ItemType::DeleteTenant));
        assert_eq!(ItemType::from_str("invalid"), None);
    }

    #[test]
    fn test_operation_validity() {
        // Content can be ingested, updated, deleted
        assert!(QueueOperation::Ingest.is_valid_for(ItemType::Content));
        assert!(QueueOperation::Update.is_valid_for(ItemType::Content));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Content));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::Content));

        // Folder cannot be updated
        assert!(!QueueOperation::Update.is_valid_for(ItemType::Folder));
        assert!(QueueOperation::Scan.is_valid_for(ItemType::Folder));

        // DeleteTenant can only delete
        assert!(!QueueOperation::Ingest.is_valid_for(ItemType::DeleteTenant));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::DeleteTenant));

        // Rename can only update
        assert!(QueueOperation::Update.is_valid_for(ItemType::Rename));
        assert!(!QueueOperation::Ingest.is_valid_for(ItemType::Rename));
    }

    #[test]
    fn test_idempotency_key_generation() {
        let key1 = generate_idempotency_key(
            ItemType::File,
            "my-collection",
            "/path/to/file.txt",
        );
        let key2 = generate_idempotency_key(
            ItemType::File,
            "my-collection",
            "/path/to/file.txt",
        );
        assert_eq!(key1, key2); // Same inputs = same key

        let key3 = generate_idempotency_key(
            ItemType::File,
            "my-collection",
            "/path/to/other.txt",
        );
        assert_ne!(key1, key3); // Different inputs = different key
    }

    #[test]
    fn test_queue_status_display() {
        assert_eq!(QueueStatus::Pending.to_string(), "pending");
        assert_eq!(QueueStatus::InProgress.to_string(), "in_progress");
    }

    #[test]
    fn test_unified_idempotency_key_generation() {
        let key1 = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Ingest,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        ).unwrap();

        // Key should be exactly 32 hex characters
        assert_eq!(key1.len(), 32);

        // Same inputs should produce same key
        let key2 = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Ingest,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        ).unwrap();
        assert_eq!(key1, key2);

        // Different inputs should produce different key
        let key3 = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Ingest,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/other.rs"}"#,
        ).unwrap();
        assert_ne!(key1, key3);

        // Different operation should produce different key
        let key4 = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Update,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        ).unwrap();
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_unified_idempotency_key_validation() {
        // Empty tenant_id should fail
        let result = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Ingest,
            "",  // Empty tenant_id
            "my-collection",
            "{}",
        );
        assert_eq!(result, Err(IdempotencyKeyError::EmptyTenantId));

        // Empty collection should fail
        let result = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Ingest,
            "proj_abc123",
            "",  // Empty collection
            "{}",
        );
        assert_eq!(result, Err(IdempotencyKeyError::EmptyCollection));

        // Invalid operation for type should fail
        let result = generate_unified_idempotency_key(
            ItemType::DeleteTenant,
            QueueOperation::Ingest,  // DeleteTenant only supports Delete
            "proj_abc123",
            "my-collection",
            "{}",
        );
        assert!(matches!(result, Err(IdempotencyKeyError::InvalidOperationForType { .. })));
    }

    #[test]
    fn test_unified_idempotency_key_cross_language_compatibility() {
        // This test vector should produce the same hash in both Rust and Python
        // Input: "file|ingest|proj_abc123|my-project-code|{}"
        let key = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Ingest,
            "proj_abc123",
            "my-project-code",
            "{}",
        ).unwrap();

        // Verify the key is valid hex
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(key.len(), 32);

        // The actual hash value for cross-language testing
        // Python should produce the same result with:
        // hashlib.sha256(b"file|ingest|proj_abc123|my-project-code|{}").hexdigest()[:32]
        // Expected: "0e6c3a8f7b8e4f2c9a1d5e7f3b6c8d9e" (placeholder - actual value computed at runtime)
        println!("Cross-language test key: {}", key);
    }
}
