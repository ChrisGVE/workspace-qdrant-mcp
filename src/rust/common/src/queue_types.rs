//! Unified queue type definitions shared between daemon and CLI
//!
//! This module provides the canonical enum types for queue items, operations,
//! and statuses. Both the Rust daemon and CLI import these to ensure consistency.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Item types that can be enqueued for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemType {
    /// Direct text content (scratchbook, notes, clipboard)
    Content,
    /// Single file ingestion with path reference
    File,
    /// Folder scan operation (generates child file items)
    Folder,
    /// Project initialization/scan (top-level container)
    Project,
    /// Library documentation ingestion
    Library,
    /// Tenant-wide deletion operation
    DeleteTenant,
    /// Single document deletion by ID
    DeleteDocument,
    /// File/folder rename tracking
    Rename,
    /// URL fetch and ingestion (web pages, documents at URLs)
    Url,
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ItemType::Content => write!(f, "content"),
            ItemType::File => write!(f, "file"),
            ItemType::Folder => write!(f, "folder"),
            ItemType::Project => write!(f, "project"),
            ItemType::Library => write!(f, "library"),
            ItemType::DeleteTenant => write!(f, "delete_tenant"),
            ItemType::DeleteDocument => write!(f, "delete_document"),
            ItemType::Rename => write!(f, "rename"),
            ItemType::Url => write!(f, "url"),
        }
    }
}

impl ItemType {
    /// Parse item type from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "content" => Some(ItemType::Content),
            "file" => Some(ItemType::File),
            "folder" => Some(ItemType::Folder),
            "project" => Some(ItemType::Project),
            "library" => Some(ItemType::Library),
            "delete_tenant" => Some(ItemType::DeleteTenant),
            "delete_document" => Some(ItemType::DeleteDocument),
            "rename" => Some(ItemType::Rename),
            "url" => Some(ItemType::Url),
            _ => None,
        }
    }

    /// Get all valid item types
    pub fn all() -> &'static [ItemType] {
        &[
            ItemType::Content,
            ItemType::File,
            ItemType::Folder,
            ItemType::Project,
            ItemType::Library,
            ItemType::DeleteTenant,
            ItemType::DeleteDocument,
            ItemType::Rename,
            ItemType::Url,
        ]
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ItemType::Content => "content",
            ItemType::File => "file",
            ItemType::Folder => "folder",
            ItemType::Project => "project",
            ItemType::Library => "library",
            ItemType::DeleteTenant => "delete_tenant",
            ItemType::DeleteDocument => "delete_document",
            ItemType::Rename => "rename",
            ItemType::Url => "url",
        }
    }
}

/// Operation types for queue items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueOperation {
    /// Initial ingestion or re-ingestion of content
    Ingest,
    /// Update existing content (delete + reingest)
    Update,
    /// Remove content from vector database
    Delete,
    /// Scan directory/project without immediate ingestion
    Scan,
}

impl fmt::Display for QueueOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueueOperation::Ingest => write!(f, "ingest"),
            QueueOperation::Update => write!(f, "update"),
            QueueOperation::Delete => write!(f, "delete"),
            QueueOperation::Scan => write!(f, "scan"),
        }
    }
}

impl QueueOperation {
    /// Parse operation from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "ingest" => Some(QueueOperation::Ingest),
            "update" => Some(QueueOperation::Update),
            "delete" => Some(QueueOperation::Delete),
            "scan" => Some(QueueOperation::Scan),
            _ => None,
        }
    }

    /// Check if this operation is valid for the given item type
    pub fn is_valid_for(&self, item_type: ItemType) -> bool {
        match (item_type, self) {
            // Content: ingest, update, delete
            (ItemType::Content, QueueOperation::Ingest) => true,
            (ItemType::Content, QueueOperation::Update) => true,
            (ItemType::Content, QueueOperation::Delete) => true,
            (ItemType::Content, QueueOperation::Scan) => false,

            // File: ingest, update, delete
            (ItemType::File, QueueOperation::Ingest) => true,
            (ItemType::File, QueueOperation::Update) => true,
            (ItemType::File, QueueOperation::Delete) => true,
            (ItemType::File, QueueOperation::Scan) => false,

            // Folder: ingest, delete, scan (no update)
            (ItemType::Folder, QueueOperation::Ingest) => true,
            (ItemType::Folder, QueueOperation::Update) => false,
            (ItemType::Folder, QueueOperation::Delete) => true,
            (ItemType::Folder, QueueOperation::Scan) => true,

            // Project: ingest, delete, scan (no update)
            (ItemType::Project, QueueOperation::Ingest) => true,
            (ItemType::Project, QueueOperation::Update) => false,
            (ItemType::Project, QueueOperation::Delete) => true,
            (ItemType::Project, QueueOperation::Scan) => true,

            // Library: ingest, update, delete
            (ItemType::Library, QueueOperation::Ingest) => true,
            (ItemType::Library, QueueOperation::Update) => true,
            (ItemType::Library, QueueOperation::Delete) => true,
            (ItemType::Library, QueueOperation::Scan) => false,

            // DeleteTenant: only delete
            (ItemType::DeleteTenant, QueueOperation::Delete) => true,
            (ItemType::DeleteTenant, _) => false,

            // DeleteDocument: only delete
            (ItemType::DeleteDocument, QueueOperation::Delete) => true,
            (ItemType::DeleteDocument, _) => false,

            // Rename: only update
            (ItemType::Rename, QueueOperation::Update) => true,
            (ItemType::Rename, _) => false,

            // Url: ingest, update, delete
            (ItemType::Url, QueueOperation::Ingest) => true,
            (ItemType::Url, QueueOperation::Update) => true,
            (ItemType::Url, QueueOperation::Delete) => true,
            (ItemType::Url, QueueOperation::Scan) => true, // Crawl operation
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueOperation::Ingest => "ingest",
            QueueOperation::Update => "update",
            QueueOperation::Delete => "delete",
            QueueOperation::Scan => "scan",
        }
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
    fn test_item_type_as_str() {
        assert_eq!(ItemType::Content.as_str(), "content");
        assert_eq!(ItemType::File.as_str(), "file");
        assert_eq!(ItemType::Project.as_str(), "project");
    }

    #[test]
    fn test_operation_validity() {
        assert!(QueueOperation::Ingest.is_valid_for(ItemType::Content));
        assert!(QueueOperation::Update.is_valid_for(ItemType::Content));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::Content));
        assert!(!QueueOperation::Scan.is_valid_for(ItemType::Content));

        assert!(!QueueOperation::Update.is_valid_for(ItemType::Folder));
        assert!(QueueOperation::Scan.is_valid_for(ItemType::Folder));

        assert!(!QueueOperation::Ingest.is_valid_for(ItemType::DeleteTenant));
        assert!(QueueOperation::Delete.is_valid_for(ItemType::DeleteTenant));

        assert!(QueueOperation::Update.is_valid_for(ItemType::Rename));
        assert!(!QueueOperation::Ingest.is_valid_for(ItemType::Rename));
    }

    #[test]
    fn test_queue_operation_as_str() {
        assert_eq!(QueueOperation::Ingest.as_str(), "ingest");
        assert_eq!(QueueOperation::Update.as_str(), "update");
        assert_eq!(QueueOperation::Delete.as_str(), "delete");
    }

    #[test]
    fn test_queue_status_display() {
        assert_eq!(QueueStatus::Pending.to_string(), "pending");
        assert_eq!(QueueStatus::InProgress.to_string(), "in_progress");
    }
}
