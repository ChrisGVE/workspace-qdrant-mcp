//! Payload validation for queue items.

use tracing::error;

use crate::unified_queue_schema::{ItemType, QueueOperation as UnifiedOp};

use super::{QueueError, QueueManager, QueueResult};

impl QueueManager {
    /// Validate payload contains required fields for the given item type (Task 46)
    ///
    /// Returns Ok(()) if valid, or an error describing the missing field.
    pub(super) fn validate_payload_for_type(item_type: ItemType, op: UnifiedOp, payload: &serde_json::Value) -> QueueResult<()> {
        match item_type {
            ItemType::File => {
                if !payload.get("file_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: file item missing 'file_path' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "file".to_string(),
                        field: "file_path".to_string(),
                    });
                }
                if op == UnifiedOp::Rename {
                    if !payload.get("old_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                        error!("Queue validation failed: file rename missing 'old_path' in payload");
                        return Err(QueueError::MissingPayloadField {
                            item_type: "file".to_string(),
                            field: "old_path".to_string(),
                        });
                    }
                }
            }
            ItemType::Folder => {
                if !payload.get("folder_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: folder item missing 'folder_path' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "folder".to_string(),
                        field: "folder_path".to_string(),
                    });
                }
                if op == UnifiedOp::Rename {
                    if !payload.get("old_path").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                        error!("Queue validation failed: folder rename missing 'old_path' in payload");
                        return Err(QueueError::MissingPayloadField {
                            item_type: "folder".to_string(),
                            field: "old_path".to_string(),
                        });
                    }
                }
            }
            ItemType::Tenant => {
                // Tenant validation depends on collection context
                // Projects need project_root, libraries need library_name
                // For delete ops, tenant_id is sufficient (already in queue item)
                // For rename ops, need old_tenant_id
                if op == UnifiedOp::Rename {
                    if !payload.get("old_tenant_id").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                        error!("Queue validation failed: tenant rename missing 'old_tenant_id'");
                        return Err(QueueError::MissingPayloadField {
                            item_type: "tenant".to_string(),
                            field: "old_tenant_id".to_string(),
                        });
                    }
                }
                // For add/scan ops on projects, project_root is needed
                // For add ops on libraries, library_name is needed
                // These are validated contextually in the processor
            }
            ItemType::Doc => {
                if !payload.get("document_id").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: doc item missing 'document_id' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "doc".to_string(),
                        field: "document_id".to_string(),
                    });
                }
            }
            ItemType::Text => {
                // Text items must have a 'content' field (can be empty string for some operations)
                if !payload.get("content").map_or(false, |v| v.is_string()) {
                    error!("Queue validation failed: text item missing 'content' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "text".to_string(),
                        field: "content".to_string(),
                    });
                }
            }
            ItemType::Website => {
                if !payload.get("url").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: website item missing 'url' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "website".to_string(),
                        field: "url".to_string(),
                    });
                }
            }
            ItemType::Collection => {
                if !payload.get("collection_name").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: collection item missing 'collection_name' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "collection".to_string(),
                        field: "collection_name".to_string(),
                    });
                }
            }
            ItemType::Url => {
                if !payload.get("url").map_or(false, |v| v.is_string() && !v.as_str().unwrap_or("").is_empty()) {
                    error!("Queue validation failed: url item missing 'url' in payload");
                    return Err(QueueError::MissingPayloadField {
                        item_type: "url".to_string(),
                        field: "url".to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}
