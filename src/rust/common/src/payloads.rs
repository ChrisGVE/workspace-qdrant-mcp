//! Queue payload structs shared between daemon and CLI
//!
//! These structs represent the JSON payloads for different queue item types.

use serde::{Deserialize, Serialize};

/// Payload for content items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPayload {
    /// The actual text content
    pub content: String,
    /// Source type: scratchbook, mcp, clipboard
    pub source_type: String,
    /// Primary categorization tag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_tag: Option<String>,
    /// Full hierarchical tag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_tag: Option<String>,
}

/// Payload for file items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePayload {
    /// Absolute path to the file
    pub file_path: String,
    /// File type classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_type: Option<String>,
    /// SHA256 hash for change detection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,
    /// File size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

/// Payload for folder items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderPayload {
    /// Absolute path to the folder
    pub folder_path: String,
    /// Whether to scan recursively
    #[serde(default = "default_true")]
    pub recursive: bool,
    /// Maximum recursion depth
    #[serde(default = "default_recursive_depth")]
    pub recursive_depth: u32,
    /// File patterns to include
    #[serde(default)]
    pub patterns: Vec<String>,
    /// Patterns to ignore
    #[serde(default)]
    pub ignore_patterns: Vec<String>,
}

fn default_true() -> bool { true }
fn default_recursive_depth() -> u32 { 10 }

/// Payload for project items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectPayload {
    /// Absolute path to project root
    pub project_root: String,
    /// Git remote URL (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_remote: Option<String>,
    /// Project type classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_type: Option<String>,
}

/// Payload for library items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryPayload {
    /// Library name
    pub library_name: String,
    /// Library version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library_version: Option<String>,
    /// Source URL for documentation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
}

/// Payload for delete_tenant items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteTenantPayload {
    /// Tenant ID to delete
    pub tenant_id_to_delete: String,
    /// Reason for deletion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Payload for delete_document items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteDocumentPayload {
    /// Document identifier (UUID or path)
    pub document_id: String,
    /// Specific point IDs to delete (optional)
    #[serde(default)]
    pub point_ids: Vec<String>,
}

/// Discriminator for rename payload types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RenameType {
    /// File or folder path rename (delete old + re-ingest new)
    PathRename,
    /// Tenant ID cascade rename (update Qdrant payloads in-place)
    TenantIdRename,
}

impl Default for RenameType {
    fn default() -> Self {
        RenameType::PathRename
    }
}

/// Payload for rename items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenamePayload {
    /// Type of rename operation
    #[serde(default)]
    pub rename_type: RenameType,
    /// Original path (for PathRename)
    #[serde(default)]
    pub old_path: Option<String>,
    /// New path (for PathRename)
    #[serde(default)]
    pub new_path: Option<String>,
    /// Whether this is a folder rename (for PathRename)
    #[serde(default)]
    pub is_folder: bool,
    /// Old tenant ID (for TenantIdRename)
    #[serde(default)]
    pub old_tenant_id: Option<String>,
    /// New tenant ID (for TenantIdRename)
    #[serde(default)]
    pub new_tenant_id: Option<String>,
    /// Reason for the rename (e.g., "remote_url_changed", "manual_rename")
    #[serde(default)]
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_payload_serde() {
        let payload = ContentPayload {
            content: "test content".to_string(),
            source_type: "cli".to_string(),
            main_tag: Some("tag1".to_string()),
            full_tag: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("test content"));
        assert!(json.contains("cli"));
        assert!(json.contains("tag1"));
        assert!(!json.contains("full_tag"));

        let back: ContentPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "test content");
    }

    #[test]
    fn test_file_payload_serde() {
        let payload = FilePayload {
            file_path: "/src/main.rs".to_string(),
            file_type: Some("code".to_string()),
            file_hash: None,
            size_bytes: Some(1024),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("/src/main.rs"));
        assert!(!json.contains("file_hash"));

        let back: FilePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.file_path, "/src/main.rs");
        assert_eq!(back.size_bytes, Some(1024));
    }

    #[test]
    fn test_rename_type_default() {
        assert_eq!(RenameType::default(), RenameType::PathRename);
    }
}
