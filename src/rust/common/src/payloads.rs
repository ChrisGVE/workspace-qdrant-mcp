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

/// Payload for memory rule items (queued via MCP memory tool)
///
/// Memory rules have their own payload type because they carry metadata
/// (label, scope, title, tags, priority) that must be persisted in the
/// Qdrant point payload for filtering and display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPayload {
    /// Rule content text
    pub content: String,
    /// Source type (always "memory_rule")
    pub source_type: String,
    /// Rule label (identifier, max 15 chars, e.g. "prefer-uv")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Action: add, update, remove
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Scope: global or project
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Project ID for project-scoped rules
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Rule title (max 50 chars)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tags for categorization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Priority (higher = more important)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
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

    #[test]
    fn test_memory_payload_full_serde() {
        let payload = MemoryPayload {
            content: "always use bun".to_string(),
            source_type: "memory_rule".to_string(),
            label: Some("prefer-bun".to_string()),
            action: Some("add".to_string()),
            scope: Some("global".to_string()),
            project_id: None,
            title: Some("Prefer bun over npm".to_string()),
            tags: Some(vec!["tooling".to_string(), "workflow".to_string()]),
            priority: Some(8),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("prefer-bun"));
        assert!(json.contains("global"));
        assert!(json.contains("tooling"));
        assert!(!json.contains("project_id"));

        let back: MemoryPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.label, Some("prefer-bun".to_string()));
        assert_eq!(back.tags, Some(vec!["tooling".to_string(), "workflow".to_string()]));
        assert_eq!(back.priority, Some(8));
    }

    #[test]
    fn test_memory_payload_minimal_serde() {
        let json = r#"{"content":"test rule","source_type":"memory_rule"}"#;
        let payload: MemoryPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.content, "test rule");
        assert_eq!(payload.label, None);
        assert_eq!(payload.scope, None);
        assert_eq!(payload.tags, None);
    }

    #[test]
    fn test_memory_payload_from_mcp_json() {
        // Simulate the JSON the MCP server actually sends
        let json = r#"{
            "content": "deploy after build",
            "source_type": "memory_rule",
            "label": "deploy-after-build",
            "action": "add",
            "scope": "project",
            "project_id": "abc123",
            "title": "Deploy binaries after changes",
            "tags": ["workflow", "deployment"],
            "priority": 9
        }"#;
        let payload: MemoryPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.scope, Some("project".to_string()));
        assert_eq!(payload.project_id, Some("abc123".to_string()));
        assert_eq!(payload.priority, Some(9));
    }
}
