//! Types for the unified queue schema.
//!
//! Contains [`UnifiedQueueItem`] (the full queue row representation) and
//! [`UnifiedQueueStats`] (aggregate statistics).

use serde::{Deserialize, Serialize};

pub use wqm_common::payloads::{
    CollectionPayload, ContentPayload, DeleteDocumentPayload, DeleteTenantPayload, FilePayload,
    FolderPayload, LibraryPayload, MemoryPayload, ProjectPayload, ScratchpadPayload, UrlPayload,
    WebsitePayload,
};
pub use wqm_common::queue_types::{
    DestinationStatus, ItemType, QueueDecision, QueueOperation, QueueStatus,
};

/// Complete unified queue item representation.
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
    /// Last error message if failed
    #[serde(default)]
    pub error_message: Option<String>,
    /// Timestamp of last error (RFC3339)
    #[serde(default)]
    pub last_error_at: Option<String>,
    /// File path for file item types (Task 22: per-file deduplication).
    /// Only set for item_type='file', None for other types.
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

impl UnifiedQueueItem {
    /// Parse the payload JSON into a typed payload struct.
    pub fn parse_content_payload(&self) -> Result<ContentPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a FilePayload.
    pub fn parse_file_payload(&self) -> Result<FilePayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a FolderPayload.
    pub fn parse_folder_payload(&self) -> Result<FolderPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a ProjectPayload.
    pub fn parse_project_payload(&self) -> Result<ProjectPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a LibraryPayload.
    pub fn parse_library_payload(&self) -> Result<LibraryPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a DeleteTenantPayload.
    pub fn parse_delete_tenant_payload(&self) -> Result<DeleteTenantPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a DeleteDocumentPayload.
    pub fn parse_delete_document_payload(
        &self,
    ) -> Result<DeleteDocumentPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a UrlPayload.
    pub fn parse_url_payload(&self) -> Result<UrlPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a WebsitePayload.
    pub fn parse_website_payload(&self) -> Result<WebsitePayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the payload JSON into a CollectionPayload.
    pub fn parse_collection_payload(&self) -> Result<CollectionPayload, serde_json::Error> {
        serde_json::from_str(&self.payload_json)
    }

    /// Parse the decision_json field into a QueueDecision.
    pub fn parse_decision(&self) -> Option<Result<QueueDecision, serde_json::Error>> {
        self.decision_json
            .as_ref()
            .map(|json| serde_json::from_str(json))
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

    /// Check if the lease has expired.
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

/// Statistics for the unified queue.
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
