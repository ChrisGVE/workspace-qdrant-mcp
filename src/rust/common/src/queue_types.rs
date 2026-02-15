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

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::constants;

/// Item types that can be enqueued for processing.
///
/// Describes WHAT the object is, not what action to perform on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemType {
    /// Direct text content (scratchbook, notes, memory rules)
    Text,
    /// Single file with path reference
    File,
    /// URL fetch (individual web page)
    Url,
    /// Entire website (multi-page crawl)
    Website,
    /// Generalized document reference (for delete-by-ID, uplift-by-ID)
    Doc,
    /// Directory
    Folder,
    /// Project or library tenant (collection field disambiguates)
    Tenant,
    /// A Qdrant collection itself
    Collection,
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ItemType {
    /// Parse item type from string.
    ///
    /// Accepts both new and legacy values for migration robustness:
    /// - "content" → Text
    /// - "project" / "library" / "delete_tenant" → Tenant
    /// - "delete_document" → Doc
    /// - "rename" → None (rename is an operation, not an item type)
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            constants::item_type::TEXT => Some(ItemType::Text),
            constants::item_type::FILE => Some(ItemType::File),
            constants::item_type::URL => Some(ItemType::Url),
            constants::item_type::WEBSITE => Some(ItemType::Website),
            constants::item_type::DOC => Some(ItemType::Doc),
            constants::item_type::FOLDER => Some(ItemType::Folder),
            constants::item_type::TENANT => Some(ItemType::Tenant),
            constants::item_type::COLLECTION => Some(ItemType::Collection),
            // Legacy mappings for migration robustness
            "content" => Some(ItemType::Text),
            "project" | "library" | "delete_tenant" => Some(ItemType::Tenant),
            "delete_document" => Some(ItemType::Doc),
            // "rename" was an item type before; it's now an operation
            _ => None,
        }
    }

    /// Get all valid item types
    pub fn all() -> &'static [ItemType] {
        &[
            ItemType::Text,
            ItemType::File,
            ItemType::Url,
            ItemType::Website,
            ItemType::Doc,
            ItemType::Folder,
            ItemType::Tenant,
            ItemType::Collection,
        ]
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ItemType::Text => constants::item_type::TEXT,
            ItemType::File => constants::item_type::FILE,
            ItemType::Url => constants::item_type::URL,
            ItemType::Website => constants::item_type::WEBSITE,
            ItemType::Doc => constants::item_type::DOC,
            ItemType::Folder => constants::item_type::FOLDER,
            ItemType::Tenant => constants::item_type::TENANT,
            ItemType::Collection => constants::item_type::COLLECTION,
        }
    }

    /// Dequeue priority tier for this item type.
    ///
    /// Higher number = dequeued first (DESC sort).
    /// Content types (2) before structural (1) before administrative (0).
    pub fn dequeue_priority(&self) -> u8 {
        match self {
            ItemType::Text
            | ItemType::File
            | ItemType::Url
            | ItemType::Website
            | ItemType::Doc => 2,
            ItemType::Folder => 1,
            ItemType::Tenant | ItemType::Collection => 0,
        }
    }
}

/// Operation types for queue items.
///
/// Describes WHAT TO DO with the item, not what the item is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueOperation {
    /// Create/add new content
    Add,
    /// Modify existing content; tenant re-sync
    Update,
    /// Remove content
    Delete,
    /// Enumerate contents (folders, tenants, websites)
    Scan,
    /// Move/rename path or tenant_id
    Rename,
    /// Metadata enrichment cascade (collection → tenant → doc)
    Uplift,
    /// Clear all content in a collection (not delete the collection)
    Reset,
}

impl fmt::Display for QueueOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl QueueOperation {
    /// Parse operation from string.
    ///
    /// Accepts legacy "ingest" → Add for migration robustness.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            constants::operation::ADD => Some(QueueOperation::Add),
            constants::operation::UPDATE => Some(QueueOperation::Update),
            constants::operation::DELETE => Some(QueueOperation::Delete),
            constants::operation::SCAN => Some(QueueOperation::Scan),
            constants::operation::RENAME => Some(QueueOperation::Rename),
            constants::operation::UPLIFT => Some(QueueOperation::Uplift),
            constants::operation::RESET => Some(QueueOperation::Reset),
            // Legacy mapping
            "ingest" => Some(QueueOperation::Add),
            _ => None,
        }
    }

    /// Check if this operation is valid for the given item type.
    ///
    /// Valid combinations matrix:
    /// ```text
    /// item_type \ op | add | update | delete | scan | rename | uplift | reset
    /// text           |  Y  |   Y    |   Y    |  -   |   -    |   Y    |   -
    /// file           |  Y  |   Y    |   Y    |  -   |   Y    |   Y    |   -
    /// url            |  Y  |   Y    |   Y    |  -   |   -    |   Y    |   -
    /// website        |  Y  |   Y    |   Y    |  Y   |   -    |   Y    |   -
    /// doc            |  -  |   -    |   Y    |  -   |   -    |   Y    |   -
    /// folder         |  -  |   -    |   Y    |  Y   |   Y    |   -    |   -
    /// tenant         |  Y  |   Y    |   Y    |  Y   |   Y    |   Y    |   -
    /// collection     |  -  |   -    |   -    |  -   |   -    |   Y    |   Y
    /// ```
    pub fn is_valid_for(&self, item_type: ItemType) -> bool {
        match (item_type, self) {
            // Text: add, update, delete, uplift
            (ItemType::Text, QueueOperation::Add) => true,
            (ItemType::Text, QueueOperation::Update) => true,
            (ItemType::Text, QueueOperation::Delete) => true,
            (ItemType::Text, QueueOperation::Uplift) => true,
            (ItemType::Text, _) => false,

            // File: add, update, delete, rename, uplift
            (ItemType::File, QueueOperation::Add) => true,
            (ItemType::File, QueueOperation::Update) => true,
            (ItemType::File, QueueOperation::Delete) => true,
            (ItemType::File, QueueOperation::Rename) => true,
            (ItemType::File, QueueOperation::Uplift) => true,
            (ItemType::File, _) => false,

            // Url: add, update, delete, uplift
            (ItemType::Url, QueueOperation::Add) => true,
            (ItemType::Url, QueueOperation::Update) => true,
            (ItemType::Url, QueueOperation::Delete) => true,
            (ItemType::Url, QueueOperation::Uplift) => true,
            (ItemType::Url, _) => false,

            // Website: add, update, delete, scan, uplift
            (ItemType::Website, QueueOperation::Add) => true,
            (ItemType::Website, QueueOperation::Update) => true,
            (ItemType::Website, QueueOperation::Delete) => true,
            (ItemType::Website, QueueOperation::Scan) => true,
            (ItemType::Website, QueueOperation::Uplift) => true,
            (ItemType::Website, _) => false,

            // Doc: delete, uplift only
            (ItemType::Doc, QueueOperation::Delete) => true,
            (ItemType::Doc, QueueOperation::Uplift) => true,
            (ItemType::Doc, _) => false,

            // Folder: delete, scan, rename
            (ItemType::Folder, QueueOperation::Delete) => true,
            (ItemType::Folder, QueueOperation::Scan) => true,
            (ItemType::Folder, QueueOperation::Rename) => true,
            (ItemType::Folder, _) => false,

            // Tenant: add, update, delete, scan, rename, uplift
            (ItemType::Tenant, QueueOperation::Add) => true,
            (ItemType::Tenant, QueueOperation::Update) => true,
            (ItemType::Tenant, QueueOperation::Delete) => true,
            (ItemType::Tenant, QueueOperation::Scan) => true,
            (ItemType::Tenant, QueueOperation::Rename) => true,
            (ItemType::Tenant, QueueOperation::Uplift) => true,
            (ItemType::Tenant, _) => false,

            // Collection: uplift, reset only
            (ItemType::Collection, QueueOperation::Uplift) => true,
            (ItemType::Collection, QueueOperation::Reset) => true,
            (ItemType::Collection, _) => false,
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueOperation::Add => constants::operation::ADD,
            QueueOperation::Update => constants::operation::UPDATE,
            QueueOperation::Delete => constants::operation::DELETE,
            QueueOperation::Scan => constants::operation::SCAN,
            QueueOperation::Rename => constants::operation::RENAME,
            QueueOperation::Uplift => constants::operation::UPLIFT,
            QueueOperation::Reset => constants::operation::RESET,
        }
    }

    /// Get all valid operations
    pub fn all() -> &'static [QueueOperation] {
        &[
            QueueOperation::Add,
            QueueOperation::Update,
            QueueOperation::Delete,
            QueueOperation::Scan,
            QueueOperation::Rename,
            QueueOperation::Uplift,
            QueueOperation::Reset,
        ]
    }
}

/// Queue item status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueStatus {
    /// Ready to be picked up by processor
    Pending,
    /// Currently being processed (lease acquired)
    InProgress,
    /// Successfully completed
    Done,
    /// Max retries exceeded
    Failed,
}

impl fmt::Display for QueueStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueueStatus::Pending => write!(f, "pending"),
            QueueStatus::InProgress => write!(f, "in_progress"),
            QueueStatus::Done => write!(f, "done"),
            QueueStatus::Failed => write!(f, "failed"),
        }
    }
}

impl QueueStatus {
    /// Parse status from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(QueueStatus::Pending),
            "in_progress" => Some(QueueStatus::InProgress),
            "done" => Some(QueueStatus::Done),
            "failed" => Some(QueueStatus::Failed),
            _ => None,
        }
    }
}

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
