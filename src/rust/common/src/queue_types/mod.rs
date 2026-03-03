//! Unified queue type definitions shared between daemon and CLI
//!
//! This module provides the canonical enum types for queue items, operations,
//! and statuses. Both the Rust daemon and CLI import these to ensure consistency.
//!
//! ## Taxonomy
//!
//! `ItemType` describes WHAT the object is (text, file, folder, tenant, etc.).
//! `QueueOperation` describes WHAT TO DO with it (add, update, delete, scan, etc.).
//!
//! Not all combinations are valid — use `QueueOperation::is_valid_for()` to check.

mod decision;
mod item_type;
mod operation;
mod status;

pub use decision::QueueDecision;
pub use item_type::ItemType;
pub use operation::QueueOperation;
pub use status::{DestinationStatus, QueueStatus};

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ItemType tests
    // ========================================================================

    #[test]
    fn test_item_type_display() {
        assert_eq!(ItemType::Text.to_string(), "text");
        assert_eq!(ItemType::File.to_string(), "file");
        assert_eq!(ItemType::Url.to_string(), "url");
        assert_eq!(ItemType::Website.to_string(), "website");
        assert_eq!(ItemType::Doc.to_string(), "doc");
        assert_eq!(ItemType::Folder.to_string(), "folder");
        assert_eq!(ItemType::Tenant.to_string(), "tenant");
        assert_eq!(ItemType::Collection.to_string(), "collection");
    }

    #[test]
    fn test_item_type_from_str_new_values() {
        assert_eq!(ItemType::from_str("text"), Some(ItemType::Text));
        assert_eq!(ItemType::from_str("file"), Some(ItemType::File));
        assert_eq!(ItemType::from_str("url"), Some(ItemType::Url));
        assert_eq!(ItemType::from_str("website"), Some(ItemType::Website));
        assert_eq!(ItemType::from_str("doc"), Some(ItemType::Doc));
        assert_eq!(ItemType::from_str("folder"), Some(ItemType::Folder));
        assert_eq!(ItemType::from_str("tenant"), Some(ItemType::Tenant));
        assert_eq!(ItemType::from_str("collection"), Some(ItemType::Collection));
        assert_eq!(ItemType::from_str("invalid"), None);
    }

    #[test]
    fn test_item_type_from_str_legacy_values() {
        // "content" → Text
        assert_eq!(ItemType::from_str("content"), Some(ItemType::Text));
        // "project" / "library" / "delete_tenant" → Tenant
        assert_eq!(ItemType::from_str("project"), Some(ItemType::Tenant));
        assert_eq!(ItemType::from_str("library"), Some(ItemType::Tenant));
        assert_eq!(ItemType::from_str("delete_tenant"), Some(ItemType::Tenant));
        // "delete_document" → Doc
        assert_eq!(ItemType::from_str("delete_document"), Some(ItemType::Doc));
        // "rename" was an item type, now it's not
        assert_eq!(ItemType::from_str("rename"), None);
    }

    #[test]
    fn test_item_type_as_str() {
        for it in ItemType::all() {
            // as_str round-trips through from_str
            assert_eq!(ItemType::from_str(it.as_str()), Some(*it));
        }
    }

    #[test]
    fn test_item_type_all_count() {
        assert_eq!(ItemType::all().len(), 8);
    }

    #[test]
    fn test_item_type_dequeue_priority() {
        // Content types: priority 2
        assert_eq!(ItemType::Text.dequeue_priority(), 2);
        assert_eq!(ItemType::File.dequeue_priority(), 2);
        assert_eq!(ItemType::Url.dequeue_priority(), 2);
        assert_eq!(ItemType::Website.dequeue_priority(), 2);
        assert_eq!(ItemType::Doc.dequeue_priority(), 2);
        // Structural: priority 1
        assert_eq!(ItemType::Folder.dequeue_priority(), 1);
        // Administrative: priority 0
        assert_eq!(ItemType::Tenant.dequeue_priority(), 0);
        assert_eq!(ItemType::Collection.dequeue_priority(), 0);
    }

    // ========================================================================
    // QueueOperation tests
    // ========================================================================

    #[test]
    fn test_queue_operation_display() {
        assert_eq!(QueueOperation::Add.to_string(), "add");
        assert_eq!(QueueOperation::Update.to_string(), "update");
        assert_eq!(QueueOperation::Delete.to_string(), "delete");
        assert_eq!(QueueOperation::Scan.to_string(), "scan");
        assert_eq!(QueueOperation::Rename.to_string(), "rename");
        assert_eq!(QueueOperation::Uplift.to_string(), "uplift");
        assert_eq!(QueueOperation::Reset.to_string(), "reset");
    }

    #[test]
    fn test_queue_operation_from_str_new_values() {
        assert_eq!(QueueOperation::from_str("add"), Some(QueueOperation::Add));
        assert_eq!(QueueOperation::from_str("update"), Some(QueueOperation::Update));
        assert_eq!(QueueOperation::from_str("delete"), Some(QueueOperation::Delete));
        assert_eq!(QueueOperation::from_str("scan"), Some(QueueOperation::Scan));
        assert_eq!(QueueOperation::from_str("rename"), Some(QueueOperation::Rename));
        assert_eq!(QueueOperation::from_str("uplift"), Some(QueueOperation::Uplift));
        assert_eq!(QueueOperation::from_str("reset"), Some(QueueOperation::Reset));
        assert_eq!(QueueOperation::from_str("invalid"), None);
    }

    #[test]
    fn test_queue_operation_from_str_legacy() {
        // "ingest" → Add
        assert_eq!(QueueOperation::from_str("ingest"), Some(QueueOperation::Add));
    }

    #[test]
    fn test_queue_operation_as_str() {
        for op in QueueOperation::all() {
            assert_eq!(QueueOperation::from_str(op.as_str()), Some(*op));
        }
    }

    #[test]
    fn test_queue_operation_all_count() {
        assert_eq!(QueueOperation::all().len(), 7);
    }

    // ========================================================================
    // Validity matrix tests
    // ========================================================================

    #[test]
    fn test_text_valid_ops() {
        assert!(QueueOperation::Add.is_valid_for(ItemType::Text));
        assert!(QueueOperation::Update.is_valid_for(ItemType::Text));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Text));
        assert!(QueueOperation::Uplift.is_valid_for(ItemType::Text));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::Text));
        assert!(!QueueOperation::Rename.is_valid_for(ItemType::Text));
        assert!(!QueueOperation::Reset.is_valid_for(ItemType::Text));
    }

    #[test]
    fn test_file_valid_ops() {
        assert!(QueueOperation::Add.is_valid_for(ItemType::File));
        assert!(QueueOperation::Update.is_valid_for(ItemType::File));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::File));
        assert!(QueueOperation::Rename.is_valid_for(ItemType::File));
        assert!(QueueOperation::Uplift.is_valid_for(ItemType::File));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::File));
        assert!(!QueueOperation::Reset.is_valid_for(ItemType::File));
    }

    #[test]
    fn test_url_valid_ops() {
        assert!(QueueOperation::Add.is_valid_for(ItemType::Url));
        assert!(QueueOperation::Update.is_valid_for(ItemType::Url));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Url));
        assert!(QueueOperation::Uplift.is_valid_for(ItemType::Url));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::Url));
        assert!(!QueueOperation::Rename.is_valid_for(ItemType::Url));
        assert!(!QueueOperation::Reset.is_valid_for(ItemType::Url));
    }

    #[test]
    fn test_website_valid_ops() {
        assert!(QueueOperation::Add.is_valid_for(ItemType::Website));
        assert!(QueueOperation::Update.is_valid_for(ItemType::Website));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Website));
        assert!(QueueOperation::Scan.is_valid_for(ItemType::Website));
        assert!(QueueOperation::Uplift.is_valid_for(ItemType::Website));
        assert!(!QueueOperation::Rename.is_valid_for(ItemType::Website));
        assert!(!QueueOperation::Reset.is_valid_for(ItemType::Website));
    }

    #[test]
    fn test_doc_valid_ops() {
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Doc));
        assert!(QueueOperation::Uplift.is_valid_for(ItemType::Doc));
        assert!(!QueueOperation::Add.is_valid_for(ItemType::Doc));
        assert!(!QueueOperation::Update.is_valid_for(ItemType::Doc));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::Doc));
        assert!(!QueueOperation::Rename.is_valid_for(ItemType::Doc));
        assert!(!QueueOperation::Reset.is_valid_for(ItemType::Doc));
    }

    #[test]
    fn test_folder_valid_ops() {
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Folder));
        assert!(QueueOperation::Scan.is_valid_for(ItemType::Folder));
        assert!(QueueOperation::Rename.is_valid_for(ItemType::Folder));
        assert!(!QueueOperation::Add.is_valid_for(ItemType::Folder));
        assert!(!QueueOperation::Update.is_valid_for(ItemType::Folder));
        assert!(!QueueOperation::Uplift.is_valid_for(ItemType::Folder));
        assert!(!QueueOperation::Reset.is_valid_for(ItemType::Folder));
    }

    #[test]
    fn test_tenant_valid_ops() {
        assert!(QueueOperation::Add.is_valid_for(ItemType::Tenant));
        assert!(QueueOperation::Update.is_valid_for(ItemType::Tenant));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Tenant));
        assert!(QueueOperation::Scan.is_valid_for(ItemType::Tenant));
        assert!(QueueOperation::Rename.is_valid_for(ItemType::Tenant));
        assert!(QueueOperation::Uplift.is_valid_for(ItemType::Tenant));
        assert!(!QueueOperation::Reset.is_valid_for(ItemType::Tenant));
    }

    #[test]
    fn test_collection_valid_ops() {
        assert!(QueueOperation::Uplift.is_valid_for(ItemType::Collection));
        assert!(QueueOperation::Reset.is_valid_for(ItemType::Collection));
        assert!(!QueueOperation::Add.is_valid_for(ItemType::Collection));
        assert!(!QueueOperation::Update.is_valid_for(ItemType::Collection));
        assert!(!QueueOperation::Delete.is_valid_for(ItemType::Collection));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::Collection));
        assert!(!QueueOperation::Rename.is_valid_for(ItemType::Collection));
    }

    #[test]
    fn test_full_validity_matrix_coverage() {
        // Ensure every (ItemType, QueueOperation) pair is covered
        let mut valid_count = 0;
        let mut invalid_count = 0;
        for it in ItemType::all() {
            for op in QueueOperation::all() {
                if op.is_valid_for(*it) {
                    valid_count += 1;
                } else {
                    invalid_count += 1;
                }
            }
        }
        // 8 item types * 7 ops = 56 total
        assert_eq!(valid_count + invalid_count, 56);
        // Expected valid: text(4) + file(5) + url(4) + website(5) + doc(2) + folder(3) + tenant(6) + collection(2) = 31
        assert_eq!(valid_count, 31);
    }

    // ========================================================================
    // QueueStatus tests
    // ========================================================================

    #[test]
    fn test_queue_status_display() {
        assert_eq!(QueueStatus::Pending.to_string(), "pending");
        assert_eq!(QueueStatus::InProgress.to_string(), "in_progress");
        assert_eq!(QueueStatus::Done.to_string(), "done");
        assert_eq!(QueueStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn test_queue_status_from_str() {
        assert_eq!(QueueStatus::from_str("pending"), Some(QueueStatus::Pending));
        assert_eq!(QueueStatus::from_str("in_progress"), Some(QueueStatus::InProgress));
        assert_eq!(QueueStatus::from_str("done"), Some(QueueStatus::Done));
        assert_eq!(QueueStatus::from_str("failed"), Some(QueueStatus::Failed));
        assert_eq!(QueueStatus::from_str("invalid"), None);
    }
}
