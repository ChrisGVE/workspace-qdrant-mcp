//! `QueueOperation` enum — describes WHAT TO DO with a queued item.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::constants;

use super::item_type::ItemType;

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
    pub fn parse_str(s: &str) -> Option<Self> {
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
