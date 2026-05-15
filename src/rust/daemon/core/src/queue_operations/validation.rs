//! Payload validation for queue items.

use tracing::error;

use crate::unified_queue_schema::{ItemType, QueueOperation as UnifiedOp};

use super::{QueueError, QueueManager, QueueResult};

/// Check that `payload[field]` is a non-empty string.
fn has_non_empty_string(payload: &serde_json::Value, field: &str) -> bool {
    payload.get(field).map_or(false, |v| {
        v.is_string() && !v.as_str().unwrap_or("").is_empty()
    })
}

/// Check that `payload[field]` is a string (may be empty).
fn has_string(payload: &serde_json::Value, field: &str) -> bool {
    payload.get(field).map_or(false, |v| v.is_string())
}

/// Validate a required non-empty string field, returning an error if missing.
fn require_non_empty(payload: &serde_json::Value, item_type: &str, field: &str) -> QueueResult<()> {
    if !has_non_empty_string(payload, field) {
        error!(
            "Queue validation failed: {} item missing '{}' in payload",
            item_type, field
        );
        return Err(QueueError::MissingPayloadField {
            item_type: item_type.to_string(),
            field: field.to_string(),
        });
    }
    Ok(())
}

/// Validate a required string field (may be empty), returning an error if missing.
fn require_string(payload: &serde_json::Value, item_type: &str, field: &str) -> QueueResult<()> {
    if !has_string(payload, field) {
        error!(
            "Queue validation failed: {} item missing '{}' in payload",
            item_type, field
        );
        return Err(QueueError::MissingPayloadField {
            item_type: item_type.to_string(),
            field: field.to_string(),
        });
    }
    Ok(())
}

impl QueueManager {
    /// Validate payload contains required fields for the given item type (Task 46)
    ///
    /// Returns Ok(()) if valid, or an error describing the missing field.
    pub(super) fn validate_payload_for_type(
        item_type: ItemType,
        op: UnifiedOp,
        payload: &serde_json::Value,
    ) -> QueueResult<()> {
        match item_type {
            ItemType::File => {
                require_non_empty(payload, "file", "file_path")?;
                if op == UnifiedOp::Rename {
                    require_non_empty(payload, "file", "old_path")?;
                }
            }
            ItemType::Folder => {
                // folder_path is Option<RelativePath>; None signals
                // "scan the watch_folder root itself" (library rescan/watch).
                // The strategy handler validates anchoring against the
                // watch_folders row at processing time.
                if op == UnifiedOp::Rename {
                    require_non_empty(payload, "folder", "old_path")?;
                }
            }
            ItemType::Tenant => {
                // Tenant validation depends on collection context
                // Projects need project_root, libraries need library_name
                // For delete ops, tenant_id is sufficient (already in queue item)
                // For rename ops, need old_tenant_id
                if op == UnifiedOp::Rename {
                    require_non_empty(payload, "tenant", "old_tenant_id")?;
                }
                // For add/scan ops on projects, project_root is needed
                // For add ops on libraries, library_name is needed
                // These are validated contextually in the processor
            }
            ItemType::Doc => {
                require_non_empty(payload, "doc", "document_id")?;
            }
            ItemType::Text => {
                // Text items must have a 'content' field (can be empty string for some operations)
                require_string(payload, "text", "content")?;
            }
            ItemType::Website => {
                require_non_empty(payload, "website", "url")?;
            }
            ItemType::Collection => {
                require_non_empty(payload, "collection", "collection_name")?;
            }
            ItemType::Url => {
                require_non_empty(payload, "url", "url")?;
            }
        }
        Ok(())
    }
}
