//! Payload parsing helper for queue items.
//!
//! Replaces 10 copies of the `serde_json::from_str(&item.payload_json).map_err(...)`
//! boilerplate scattered across strategy modules.

use serde::de::DeserializeOwned;

use crate::unified_queue_processor::error::UnifiedProcessorError;
use crate::unified_queue_schema::UnifiedQueueItem;

/// Parse a typed payload from a queue item's `payload_json`.
///
/// The error message includes the short type name and queue_id for diagnostics.
pub fn parse_payload<T: DeserializeOwned>(
    item: &UnifiedQueueItem,
) -> Result<T, UnifiedProcessorError> {
    serde_json::from_str(&item.payload_json).map_err(|e| {
        let type_name = std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("Unknown");
        UnifiedProcessorError::InvalidPayload(format!(
            "Failed to parse {}: {} (queue_id={})",
            type_name, e, item.queue_id,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus};
    use wqm_common::payloads::FilePayload;

    fn make_item(payload_json: &str) -> UnifiedQueueItem {
        UnifiedQueueItem {
            queue_id: "test-q-1".to_string(),
            idempotency_key: "idem-1".to_string(),
            item_type: ItemType::File,
            op: QueueOperation::Add,
            tenant_id: "tenant-1".to_string(),
            collection: "projects".to_string(),
            status: QueueStatus::Pending,
            branch: "main".to_string(),
            payload_json: payload_json.to_string(),
            metadata: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: None,
            search_status: None,
            decision_json: None,
        }
    }

    #[test]
    fn test_parse_valid_payload() {
        let json = r#"{"file_path": "/tmp/test.rs", "file_hash": "abc123"}"#;
        let item = make_item(json);
        let payload: FilePayload = parse_payload(&item).unwrap();
        assert_eq!(payload.file_path, "/tmp/test.rs");
    }

    #[test]
    fn test_parse_invalid_json() {
        let item = make_item("not json at all");
        let result = parse_payload::<FilePayload>(&item);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid payload"), "got: {}", err);
    }

    #[test]
    fn test_parse_wrong_type() {
        // Valid JSON but missing required `file_path` field
        let item = make_item(r#"{"content": "hello"}"#);
        let result = parse_payload::<FilePayload>(&item);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_includes_type_name() {
        let item = make_item("{}");
        let result = parse_payload::<FilePayload>(&item);
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("FilePayload"),
            "error should mention type name, got: {}",
            err
        );
        assert!(
            err.contains("test-q-1"),
            "error should mention queue_id, got: {}",
            err
        );
    }
}
