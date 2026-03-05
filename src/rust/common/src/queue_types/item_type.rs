//! `ItemType` enum — describes WHAT the queued object is.

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
            ItemType::Text | ItemType::File | ItemType::Url | ItemType::Website | ItemType::Doc => {
                2
            }
            ItemType::Folder => 1,
            ItemType::Tenant | ItemType::Collection => 0,
        }
    }
}
