//! Payloads for filesystem items: files and folders

use serde::{Deserialize, Serialize};

use crate::paths::RelativePath;

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

/// Payload for folder items.
///
/// All path fields are anchored to the owning `watch_folders.path` root
/// per docs/specs/16-path-abstraction.md §3.3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderPayload {
    /// Folder path relative to the owning watch_folder root.
    ///
    /// `None` represents the watch_folder root itself — used by library
    /// rescan / watch flows that enqueue a `Folder/Scan` against the
    /// registered library root. Strict subdirectory enqueues from the
    /// progressive scanner carry `Some(rel)` where `rel` is relative
    /// to the same root and never empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folder_path: Option<RelativePath>,
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
    /// Previous folder path before rename, relative to the watch_folder
    /// root (used when op=Rename).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_path: Option<RelativePath>,
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
            folder_path: Some(RelativePath::from_user_input("src/new_dir").unwrap()),
            recursive: true,
            recursive_depth: 10,
            patterns: vec![],
            ignore_patterns: vec![],
            old_path: Some(RelativePath::from_user_input("src/old_dir").unwrap()),
            last_scan: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("old_dir"));

        let back: FolderPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.old_path,
            Some(RelativePath::from_user_input("src/old_dir").unwrap())
        );
    }

    #[test]
    fn test_folder_payload_root_scan_serializes_without_folder_path() {
        // A library root scan (None folder_path) omits the field from JSON
        // entirely so the daemon's lookup-from-watch_folders fallback fires.
        let payload = FolderPayload {
            folder_path: None,
            recursive: true,
            recursive_depth: 10,
            patterns: vec![],
            ignore_patterns: vec![],
            old_path: None,
            last_scan: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(!json.contains("folder_path"));

        let back: FolderPayload = serde_json::from_str(&json).unwrap();
        assert!(back.folder_path.is_none());
    }
}
