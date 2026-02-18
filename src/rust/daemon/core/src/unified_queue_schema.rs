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
pub use wqm_common::queue_types::{ItemType, QueueOperation, QueueStatus, DestinationStatus, QueueDecision};
pub use wqm_common::payloads::{
    ContentPayload, FilePayload, FolderPayload, ProjectPayload,
    LibraryPayload, DeleteTenantPayload, DeleteDocumentPayload,
    MemoryPayload, UrlPayload, ScratchpadPayload, WebsitePayload, CollectionPayload,
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
    /// Per-destination status: Qdrant vector write
    #[serde(default)]
    pub qdrant_status: Option<DestinationStatus>,
    /// Per-destination status: search DB (FTS5) write
    #[serde(default)]
    pub search_status: Option<DestinationStatus>,
    /// Pre-computed processing decision (JSON-serialized QueueDecision)
    #[serde(default)]
    pub decision_json: Option<String>,
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

    /// Parse the payload JSON into a UrlPayload
    pub fn parse_url_payload(&self) -> Result<UrlPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a WebsitePayload
    pub fn parse_website_payload(&self) -> Result<WebsitePayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a CollectionPayload
    pub fn parse_collection_payload(&self) -> Result<CollectionPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the decision_json field into a QueueDecision
    pub fn parse_decision(&self) -> Option<Result<QueueDecision, serde_json::Error>> {
        self.decision_json.as_ref().map(|json| serde_json::from_str(json))
    }

    /// Determine the overall queue status from per-destination sub-statuses.
    ///
    /// Rules:
    /// - Both done → Done
    /// - Either failed (and neither pending/in_progress) → Failed
    /// - Otherwise → InProgress (still processing)
    pub fn check_completion(&self) -> QueueStatus {
        let qs = self.qdrant_status.unwrap_or(DestinationStatus::Pending);
        let ss = self.search_status.unwrap_or(DestinationStatus::Pending);

        match (qs, ss) {
            (DestinationStatus::Done, DestinationStatus::Done) => QueueStatus::Done,
            (DestinationStatus::Failed, DestinationStatus::Done)
            | (DestinationStatus::Done, DestinationStatus::Failed)
            | (DestinationStatus::Failed, DestinationStatus::Failed) => QueueStatus::Failed,
            _ => QueueStatus::InProgress,
        }
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
        'text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'
    )),
    op TEXT NOT NULL CHECK (op IN ('add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset')),
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
    file_path TEXT UNIQUE,
    -- State integrity: per-destination status tracking
    qdrant_status TEXT DEFAULT 'pending' CHECK (qdrant_status IN ('pending', 'in_progress', 'done', 'failed')),
    search_status TEXT DEFAULT 'pending' CHECK (search_status IN ('pending', 'in_progress', 'done', 'failed')),
    -- Pre-computed processing decision (JSON)
    decision_json TEXT
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

// ---------------------------------------------------------------------------
// Migration SQL — v20: qdrant_status, search_status, decision_json columns
// ---------------------------------------------------------------------------

/// SQL statements for migration v20: add per-destination status and decision columns
/// NOTE: SQLite ALTER TABLE ADD COLUMN supports CHECK constraints only if they
/// reference the added column alone. We omit CHECK here for maximum compatibility
/// and rely on daemon code to enforce valid values.
pub const MIGRATE_V20_ADD_COLUMNS_SQL: &[&str] = &[
    "ALTER TABLE unified_queue ADD COLUMN qdrant_status TEXT DEFAULT 'pending'",
    "ALTER TABLE unified_queue ADD COLUMN search_status TEXT DEFAULT 'pending'",
    "ALTER TABLE unified_queue ADD COLUMN decision_json TEXT",
];

/// Index for finding items with incomplete Qdrant writes
pub const CREATE_QDRANT_STATUS_INDEX_SQL: &str =
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_qdrant_status
       ON unified_queue(qdrant_status) WHERE qdrant_status != 'done'"#;

/// Index for finding items with incomplete search DB writes
pub const CREATE_SEARCH_STATUS_INDEX_SQL: &str =
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_search_status
       ON unified_queue(search_status) WHERE search_status != 'done'"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_type_display() {
        assert_eq!(ItemType::Text.to_string(), "text");
        assert_eq!(ItemType::Tenant.to_string(), "tenant");
    }

    #[test]
    fn test_item_type_from_str() {
        assert_eq!(ItemType::from_str("text"), Some(ItemType::Text));
        assert_eq!(ItemType::from_str("tenant"), Some(ItemType::Tenant));
        // Legacy values still work
        assert_eq!(ItemType::from_str("content"), Some(ItemType::Text));
        assert_eq!(ItemType::from_str("delete_tenant"), Some(ItemType::Tenant));
        assert_eq!(ItemType::from_str("invalid"), None);
    }

    #[test]
    fn test_operation_validity() {
        // Text: add, update, delete, uplift
        assert!(QueueOperation::Add.is_valid_for(ItemType::Text));
        assert!(QueueOperation::Update.is_valid_for(ItemType::Text));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Text));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::Text));

        // Folder: delete, scan, rename
        assert!(!QueueOperation::Update.is_valid_for(ItemType::Folder));
        assert!(QueueOperation::Scan.is_valid_for(ItemType::Folder));

        // Tenant: add, update, delete, scan, rename, uplift
        assert!(QueueOperation::Add.is_valid_for(ItemType::Tenant));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Tenant));

        // File: add, update, delete, rename, uplift
        assert!(QueueOperation::Rename.is_valid_for(ItemType::File));
        assert!(!QueueOperation::Rename.is_valid_for(ItemType::Text));
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
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        ).unwrap();

        assert_eq!(key1.len(), 32);

        let key2 = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/file.rs"}"#,
        ).unwrap();
        assert_eq!(key1, key2);

        let key3 = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            r#"{"file_path":"/path/to/other.rs"}"#,
        ).unwrap();
        assert_ne!(key1, key3);

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
        let result = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "",
            "my-collection",
            "{}",
        );
        assert_eq!(result, Err(IdempotencyKeyError::EmptyTenantId));

        let result = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "",
            "{}",
        );
        assert_eq!(result, Err(IdempotencyKeyError::EmptyCollection));

        let result = generate_unified_idempotency_key(
            ItemType::Collection,
            QueueOperation::Add,
            "proj_abc123",
            "my-collection",
            "{}",
        );
        assert!(matches!(result, Err(IdempotencyKeyError::InvalidOperationForType { .. })));
    }

    #[test]
    fn test_unified_idempotency_key_cross_language_compatibility() {
        let key = generate_unified_idempotency_key(
            ItemType::File,
            QueueOperation::Add,
            "proj_abc123",
            "my-project-code",
            "{}",
        ).unwrap();

        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_queue_decision_serde_roundtrip() {
        let decision = QueueDecision {
            delete_old: true,
            old_base_point: Some("abc123".to_string()),
            new_base_point: "def456".to_string(),
            old_file_hash: Some("oldhash".to_string()),
            new_file_hash: "newhash".to_string(),
        };

        let json = serde_json::to_string(&decision).unwrap();
        let parsed: QueueDecision = serde_json::from_str(&json).unwrap();

        assert!(parsed.delete_old);
        assert_eq!(parsed.old_base_point, Some("abc123".to_string()));
        assert_eq!(parsed.new_base_point, "def456");
        assert_eq!(parsed.old_file_hash, Some("oldhash".to_string()));
        assert_eq!(parsed.new_file_hash, "newhash");
    }

    #[test]
    fn test_queue_decision_no_old() {
        let decision = QueueDecision {
            delete_old: false,
            old_base_point: None,
            new_base_point: "def456".to_string(),
            old_file_hash: None,
            new_file_hash: "newhash".to_string(),
        };

        let json = serde_json::to_string(&decision).unwrap();
        let parsed: QueueDecision = serde_json::from_str(&json).unwrap();

        assert!(!parsed.delete_old);
        assert!(parsed.old_base_point.is_none());
    }

    #[test]
    fn test_check_completion_both_done() {
        let item = UnifiedQueueItem {
            queue_id: "q1".into(),
            idempotency_key: "k1".into(),
            item_type: ItemType::File,
            op: QueueOperation::Add,
            tenant_id: "t1".into(),
            collection: "projects".into(),
            priority: 0,
            status: QueueStatus::InProgress,
            branch: "main".into(),
            payload_json: "{}".into(),
            metadata: None,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T00:00:00Z".into(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            max_retries: 3,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: Some(DestinationStatus::Done),
            search_status: Some(DestinationStatus::Done),
            decision_json: None,
        };
        assert_eq!(item.check_completion(), QueueStatus::Done);
    }

    #[test]
    fn test_check_completion_partial_done() {
        let item = UnifiedQueueItem {
            queue_id: "q1".into(),
            idempotency_key: "k1".into(),
            item_type: ItemType::File,
            op: QueueOperation::Add,
            tenant_id: "t1".into(),
            collection: "projects".into(),
            priority: 0,
            status: QueueStatus::InProgress,
            branch: "main".into(),
            payload_json: "{}".into(),
            metadata: None,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T00:00:00Z".into(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            max_retries: 3,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: Some(DestinationStatus::Done),
            search_status: Some(DestinationStatus::Pending),
            decision_json: None,
        };
        assert_eq!(item.check_completion(), QueueStatus::InProgress);
    }

    #[test]
    fn test_check_completion_one_failed() {
        let item = UnifiedQueueItem {
            queue_id: "q1".into(),
            idempotency_key: "k1".into(),
            item_type: ItemType::File,
            op: QueueOperation::Add,
            tenant_id: "t1".into(),
            collection: "projects".into(),
            priority: 0,
            status: QueueStatus::InProgress,
            branch: "main".into(),
            payload_json: "{}".into(),
            metadata: None,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T00:00:00Z".into(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            max_retries: 3,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: Some(DestinationStatus::Done),
            search_status: Some(DestinationStatus::Failed),
            decision_json: None,
        };
        assert_eq!(item.check_completion(), QueueStatus::Failed);
    }

    #[test]
    fn test_check_completion_none_defaults_pending() {
        let item = UnifiedQueueItem {
            queue_id: "q1".into(),
            idempotency_key: "k1".into(),
            item_type: ItemType::File,
            op: QueueOperation::Add,
            tenant_id: "t1".into(),
            collection: "projects".into(),
            priority: 0,
            status: QueueStatus::Pending,
            branch: "main".into(),
            payload_json: "{}".into(),
            metadata: None,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T00:00:00Z".into(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            max_retries: 3,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: None,
            search_status: None,
            decision_json: None,
        };
        // Both None → treated as pending → InProgress
        assert_eq!(item.check_completion(), QueueStatus::InProgress);
    }

    #[test]
    fn test_destination_status_display() {
        assert_eq!(DestinationStatus::Pending.to_string(), "pending");
        assert_eq!(DestinationStatus::InProgress.to_string(), "in_progress");
        assert_eq!(DestinationStatus::Done.to_string(), "done");
        assert_eq!(DestinationStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn test_destination_status_from_str() {
        assert_eq!(DestinationStatus::from_str("pending"), Some(DestinationStatus::Pending));
        assert_eq!(DestinationStatus::from_str("done"), Some(DestinationStatus::Done));
        assert_eq!(DestinationStatus::from_str("invalid"), None);
    }
}
