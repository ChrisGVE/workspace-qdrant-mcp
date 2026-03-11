//! Payloads for filesystem items: files and folders

use serde::{Deserialize, Serialize};

pub(super) fn default_true() -> bool {
    true
}

pub(super) fn default_recursive_depth() -> u32 {
    10
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
    /// Previous path before rename (used when op=Rename)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_path: Option<String>,
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
    /// Previous path before rename (used when op=Rename)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_path: Option<String>,
    /// Baseline ISO 8601 timestamp for mtime-based pruning.
    ///
    /// Files and subdirectory scans with mtime ≤ this value are skipped —
    /// they haven't changed since the last full scan. `None` on the first
    /// scan (no baseline → scan everything).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_scan: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_payload_serde() {
        let payload = FilePayload {
            file_path: "/src/main.rs".to_string(),
            file_type: Some("code".to_string()),
            file_hash: None,
            size_bytes: Some(1024),
            old_path: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("/src/main.rs"));
        assert!(!json.contains("file_hash"));
        assert!(!json.contains("old_path"));

        let back: FilePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.file_path, "/src/main.rs");
        assert_eq!(back.size_bytes, Some(1024));
    }

    #[test]
    fn test_file_payload_with_rename() {
        let payload = FilePayload {
            file_path: "/src/new_name.rs".to_string(),
            file_type: None,
            file_hash: None,
            size_bytes: None,
            old_path: Some("/src/old_name.rs".to_string()),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("old_path"));
        assert!(json.contains("old_name.rs"));

        let back: FilePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.old_path, Some("/src/old_name.rs".to_string()));
    }

    #[test]
    fn test_folder_payload_with_rename() {
        let payload = FolderPayload {
            folder_path: "/src/new_dir".to_string(),
            recursive: true,
            recursive_depth: 10,
            patterns: vec![],
            ignore_patterns: vec![],
            old_path: Some("/src/old_dir".to_string()),
            last_scan: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("old_dir"));

        let back: FolderPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.old_path, Some("/src/old_dir".to_string()));
    }
}
